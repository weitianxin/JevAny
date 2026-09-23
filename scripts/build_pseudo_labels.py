#!/usr/bin/env python3
"""Build a ground-truth-free test-time training suite from repeated JevAny samples."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from jevany.checkpoint import LoadOptions
from jevany.device import default_device
from jevany.predictors import LocalPredictor
from jevany.self_train import blind_record, permute_choices, pseudo_label, sample_answer
from jevany.suite import digest, load_split, read_json, read_jsonl, read_manifest, semantic_hash, write_json, write_jsonl


def select_groups(rows, limit):
    """Select whole record groups in source order so paired variants stay together."""
    if not limit:
        return rows
    selected_groups, seen = [], set()
    for row in rows:
        group = row["_meta"]["group_id"]
        if group not in seen:
            if len(selected_groups) == limit:
                break
            selected_groups.append(group)
            seen.add(group)
    selected = set(selected_groups)
    return [row for row in rows if row["_meta"]["group_id"] in selected]


def shard_groups(rows, shard_index, num_shards):
    assignments = {}
    for row in rows:
        group = row["_meta"]["group_id"]
        if group not in assignments:
            assignments[group] = len(assignments) % num_shards
    return [(index, row) for index, row in enumerate(rows)
            if assignments[row["_meta"]["group_id"]] == shard_index]


def atomic_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def sequence_digest(values):
    return hashlib.sha256("\n".join(str(value) for value in values).encode()).hexdigest()


def checkpoint_digests(run):
    path = Path(run)
    if not path.is_dir():
        return {}
    names = ("head.pt", "adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    return {name: digest(path / name) for name in names if (path / name).exists()}


def merge_parts(output, num_shards):
    if (output / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed suite {output}")
    parts = output / ".parts"
    metadata = [read_json(parts / f"part-{index:05d}.json") for index in range(num_shards)]
    for index, item in enumerate(metadata):
        part_path = parts / f"part-{index:05d}.jsonl"
        if item.get("shard_index") != index or item.get("num_shards") != num_shards:
            raise ValueError(f"invalid metadata for pseudo-label shard {index}")
        if digest(part_path) != item.get("sha256"):
            raise ValueError(f"pseudo-label shard {index} checksum mismatch")
        rows = read_jsonl(part_path)
        if len(rows) != item.get("accepted"):
            raise ValueError(f"pseudo-label shard {index} record count mismatch")
        if item.get("processed") != item.get("accepted", 0) + item.get("rejected", 0):
            raise ValueError(f"pseudo-label shard {index} processed count mismatch")
        if len(item.get("processed_indices", [])) != item.get("processed"):
            raise ValueError(f"pseudo-label shard {index} processed indices mismatch")
        if sequence_digest(row["_meta"]["pseudo_parent_id"] for row in rows) != item.get("accepted_ids_sha256"):
            raise ValueError(f"pseudo-label shard {index} accepted ids mismatch")
    configs = [item["config"] for item in metadata]
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("pseudo-label shard configurations do not match")
    pseudo = [row for index in range(num_shards)
              for row in read_jsonl(parts / f"part-{index:05d}.jsonl")]
    pseudo.sort(key=lambda row: row["_meta"]["pseudo_input_index"])
    if len({row["_meta"]["pseudo_parent_id"] for row in pseudo}) != len(pseudo):
        raise ValueError("pseudo-label shards contain duplicate parent ids")
    expected = metadata[0]["manifest"]["pseudo_labeling"]["selected_records"]
    rejected = sum(item["rejected"] for item in metadata)
    if len(pseudo) + rejected != expected:
        raise ValueError(f"pseudo-label shards cover {len(pseudo) + rejected} of {expected} selected records")
    if not pseudo:
        raise ValueError("all pseudo labels were rejected")
    processed_indices = [index for item in metadata for index in item["processed_indices"]]
    if sorted(processed_indices) != list(range(expected)):
        raise ValueError("pseudo-label shards do not cover every selected input exactly once")
    path = output / "train.jsonl"
    atomic_jsonl(path, pseudo)
    template = metadata[0]["manifest"]
    template["trainable_sources"] = sorted({row["_meta"]["source"] for row in pseudo})
    template["files"] = {"train.jsonl": {
        "records": len(pseudo),
        "questions": sum(len(row["questions"]) for row in pseudo),
        "sha256": digest(path),
    }}
    template["pseudo_labeling"]["rejected"] = rejected
    template["counts"] = {
        "source": dict(Counter(row["_meta"]["source"] for row in pseudo)),
        "agreement": dict(Counter(f"{row['_meta']['pseudo_agreement']:.2f}" for row in pseudo)),
        "unique_content": len({semantic_hash(row) for row in pseudo}),
    }
    atomic_json(output / "manifest.json", template)
    return len(pseudo), rejected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="")
    parser.add_argument("--suite", default="")
    parser.add_argument("--split", default="development")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--min-agreement", type=float, default=0.5,
                        help="minimum vote share; ties are always rejected")
    parser.add_argument("--limit-groups", type=int, default=0,
                        help="use the first N complete groups; zero uses the full split")
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge", action="store_true", help="merge completed deterministic shards")
    parser.add_argument("--resume", action="store_true", help="keep an already completed shard")
    args = parser.parse_args()
    output = Path(args.out)
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")
    if args.limit_groups < 0:
        parser.error("--limit-groups must be nonnegative")
    if args.merge:
        records, rejected = merge_parts(output, args.num_shards)
        print(json.dumps({"records": records, "rejected": rejected}, indent=2))
        return
    if not args.run or not args.suite:
        parser.error("--run and --suite are required unless --merge is used")
    if args.samples < 1 or not 0.5 <= args.min_agreement <= 1:
        parser.error("--samples must be positive and --min-agreement must be in [0.5, 1]")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    if (output / "manifest.json").exists():
        raise FileExistsError(f"refusing to overwrite completed suite {output}")
    output.mkdir(parents=True, exist_ok=True)
    parts = output / ".parts"
    parts.mkdir(exist_ok=True)
    part_path = parts / f"part-{args.shard_index:05d}.jsonl"
    part_metadata_path = parts / f"part-{args.shard_index:05d}.json"
    rows = load_split(args.suite, args.split)
    if args.source:
        rows = [row for row in rows if row["_meta"]["source"] in set(args.source)]
    rows = select_groups(rows, args.limit_groups)
    selected_ids = [row["_meta"]["id"] for row in rows]
    selected_groups = sorted({row["_meta"]["group_id"] for row in rows})
    if not selected_ids:
        raise ValueError("selection produced no records")
    checkpoint = str(Path(args.run).resolve()) if Path(args.run).exists() else args.run
    config = {
        "checkpoint": checkpoint,
        "checkpoint_files": checkpoint_digests(args.run),
        "source_suite": str(Path(args.suite).resolve()),
        "source_manifest_sha256": digest(Path(args.suite) / "manifest.json"),
        "source_split": args.split,
        "source_filter": sorted(args.source),
        "limit_groups": args.limit_groups,
        "samples": args.samples,
        "temperature": args.temperature,
        "min_agreement": args.min_agreement,
        "seed": args.seed,
        "selected_ids_sha256": sequence_digest(selected_ids),
        "generation_shards": args.num_shards,
    }
    indexed_rows = shard_groups(rows, args.shard_index, args.num_shards)
    if args.resume and part_path.exists() and part_metadata_path.exists():
        existing = read_json(part_metadata_path)
        processed_ids_sha256 = sequence_digest(row["_meta"]["id"] for _, row in indexed_rows)
        if (existing.get("config") != config
                or digest(part_path) != existing.get("sha256")
                or existing.get("processed_ids_sha256") != processed_ids_sha256):
            raise ValueError(f"completed shard {args.shard_index} does not match this request")
        print(f"shard {args.shard_index} already complete", flush=True)
        return
    if args.resume and (part_path.exists() or part_metadata_path.exists()):
        part_path.unlink(missing_ok=True)
        part_metadata_path.unlink(missing_ok=True)
    if part_path.exists() or part_metadata_path.exists():
        raise FileExistsError(f"shard {args.shard_index} already exists; pass --resume to validate and keep it")
    rows = [(index, blind_record(row)) for index, row in indexed_rows]

    predictor = LocalPredictor(args.run, args.device or default_device(), LoadOptions(temperature=1.0))
    pseudo, rejected, rejected_ids = [], 0, []
    for progress, (input_index, blinded) in enumerate(rows):
        rng = random.Random(f"{args.seed}:{blinded['_meta']['id']}")
        samples = []
        for _ in range(args.samples):
            prediction = predictor(permute_choices(blinded, rng))["probabilities"]
            samples.append({question_id: sample_answer(probabilities, rng, args.temperature)
                            for question_id, probabilities in prediction.items()})
        record = pseudo_label(blinded, samples, samples=args.samples, min_agreement=args.min_agreement)
        if record is None:
            rejected += 1
            rejected_ids.append(blinded["_meta"]["id"])
        else:
            metadata = record["_meta"]
            parent_id, parent_group = metadata["id"], metadata["group_id"]
            metadata.update({
                "id": f"pseudo/{parent_id}",
                "group_id": f"pseudo/{parent_group}",
                "split": "train",
                "variant": "pseudo",
                "pseudo_parent_id": parent_id,
                "pseudo_parent_group_id": parent_group,
                "pseudo_source_split": args.split,
                "pseudo_source_suite": str(Path(args.suite).resolve()),
                "pseudo_input_index": input_index,
            })
            pseudo.append(record)
        if (progress + 1) % 25 == 0:
            print(f"shard {args.shard_index}: pseudo-labelled {progress + 1}/{len(rows)}", flush=True)

    parent = read_manifest(args.suite)
    manifest = {
        "name": "jevany-test-time-training",
        "base_revisions": parent.get("base_revisions", {}),
        "trainable_sources": sorted({row["_meta"]["source"] for row in pseudo}),
        "eval_only_sources": [],
        "holdout_sources": [],
        "pseudo_labeling": {
            **config,
            "ground_truth_used_for_adaptation": False,
            "target": "strict_majority_hard_label",
            "protocol": "transductive input adaptation; gold labels remain in the immutable source suite",
            "rejected": rejected,
            "selected_records": len(selected_ids),
            "selected_groups": len(selected_groups),
            "selected_group_ids_sha256": sequence_digest(selected_groups),
        },
    }
    atomic_jsonl(part_path, pseudo)
    atomic_json(part_metadata_path, {
        "config": config,
        "manifest": manifest,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "processed": len(rows),
        "processed_indices": [index for index, _ in rows],
        "processed_ids_sha256": sequence_digest(row["_meta"]["id"] for _, row in rows),
        "accepted": len(pseudo),
        "accepted_ids_sha256": sequence_digest(row["_meta"]["pseudo_parent_id"] for row in pseudo),
        "rejected": rejected,
        "rejected_ids_sha256": sequence_digest(rejected_ids),
        "sha256": digest(part_path),
    })
    if args.num_shards == 1:
        records, rejected = merge_parts(output, 1)
        print(json.dumps({"records": records, "rejected": rejected}, indent=2))
    else:
        print(json.dumps({"shard": args.shard_index, "records": len(pseudo), "rejected": rejected}, indent=2))


if __name__ == "__main__":
    main()
