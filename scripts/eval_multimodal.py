#!/usr/bin/env python3
"""Evaluate one deterministic shard of a multimodal suite or media ablation."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from jevany.benchmark import evaluate_records
from jevany.checkpoint import LoadOptions
from jevany.predictors import LocalPredictor
from jevany.suite import digest, load_split, read_json


def ablate(records, mode, blank_image=None, blank_video=None, seed=17):
    if mode == "full":
        return records
    rows = [{**row, "media": [dict(item) for item in row["media"]]} for row in records]
    if mode == "blank":
        blanks = {"image": blank_image, "video": blank_video}
        for row in rows:
            for item in row["media"]:
                if not blanks[item["type"]]:
                    raise ValueError(f"missing blank {item['type']} path")
                item["uri"] = str(Path(blanks[item["type"]]).resolve())
        return rows
    if mode != "shuffled":
        raise ValueError(f"unknown ablation: {mode}")
    rng = random.Random(seed)
    by_source = {}
    for row in rows:
        by_source.setdefault(row["_meta"]["source"], {}).setdefault(row["_meta"]["group_id"], row)
    mapping = {}
    for source, groups in by_source.items():
        group_ids = sorted(groups)
        if len(group_ids) < 2:
            raise ValueError("shuffled-media ablation needs at least two media groups per source")
        offset = rng.randrange(1, len(group_ids))
        replacements = group_ids[offset:] + group_ids[:offset]
        for original, replacement in zip(group_ids, replacements):
            original_hashes = groups[original]["_meta"]["media_sha256"]
            replacement_hashes = groups[replacement]["_meta"]["media_sha256"]
            if original_hashes == replacement_hashes:
                raise ValueError(f"shuffled media did not change content for {source}/{original}")
            mapping[(source, original)] = (replacement, groups[replacement]["media"], replacement_hashes)
    for row in rows:
        replacement, media, hashes = mapping[(row["_meta"]["source"], row["_meta"]["group_id"])]
        row["media"] = [dict(item) for item in media]
        row["_meta"] = {**row["_meta"], "ablation_media_group": replacement, "ablation_media_sha256": hashes}
    return rows


def verify_media(records):
    checked = {}
    for row in records:
        expected = row["_meta"].get("media_sha256", [])
        if len(expected) != len(row.get("media", [])):
            raise ValueError(f"media hash count mismatch for {row['_meta']['id']}")
        for item, sha256 in zip(row["media"], expected):
            path = Path(item["uri"])
            if path not in checked:
                checked[path] = digest(path)
            actual = checked[path]
            if actual != sha256:
                raise ValueError(f"media hash mismatch for {row['_meta']['id']}: {path}")
    return len(checked)


def ablation_provenance(records, mode, seed, blank_image, blank_video):
    value = {"mode": mode, "seed": seed if mode == "shuffled" else None}
    if mode == "blank":
        value["blank_sha256"] = {
            kind: digest(Path(path)) for kind, path in (("image", blank_image), ("video", blank_video)) if path
        }
    if mode == "shuffled":
        mapping = sorted(
            (row["_meta"]["source"], row["_meta"]["group_id"], row["_meta"]["ablation_media_group"])
            for row in records
        )
        value["mapping_sha256"] = hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode()).hexdigest()
    return value


def verify_suite_artifacts(suite):
    manifest = read_json(Path(suite) / "manifest.json")
    for name, expected in manifest.get("artifacts", {}).items():
        path = Path(suite) / name
        if not path.is_file() or digest(path) != expected["sha256"]:
            raise ValueError(f"suite artifact hash mismatch: {path}")
    audit = manifest.get("media_audit", {})
    if audit.get("exact_reference_overlap", 0) or audit.get("perceptual_reference_overlap", 0):
        raise ValueError("suite media overlaps prior model data")
    if audit.get("evaluation_decode_errors", 0):
        raise ValueError("suite media audit contains decode errors")
    return sorted(manifest.get("artifacts", {}))


def checkpoint_artifacts(directory):
    directory = Path(directory)
    names = ("adapter_model.safetensors", "adapter_model.bin", "head.pt", "adapter_config.json")
    return {name: digest(directory / name) for name in names if (directory / name).is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("full", "blank", "shuffled"), default="full")
    parser.add_argument("--blank-image")
    parser.add_argument("--blank-video")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")
    suite_artifacts = verify_suite_artifacts(args.suite)
    records = load_split(args.suite, "development")
    media_files = verify_media(records)
    records = ablate(records, args.mode, args.blank_image, args.blank_video)
    provenance = ablation_provenance(records, args.mode, 17, args.blank_image, args.blank_video)
    records = records[args.shard_index::args.num_shards]
    options = LoadOptions.from_env()
    predictor = LocalPredictor(args.run, args.device, options)
    report, _ = evaluate_records(records, predictor, args.out)
    report.update(
        suite=str(args.suite), mode=args.mode,
        num_shards=args.num_shards, shard_index=args.shard_index,
        suite_manifest_sha256=digest(Path(args.suite) / "manifest.json"),
        split_sha256=digest(Path(args.suite) / "development.jsonl"),
        media_files_verified=media_files,
        suite_artifacts_verified=suite_artifacts,
        checkpoint={
            "requested": args.run, "resolved": str(predictor.run),
            "artifacts": checkpoint_artifacts(predictor.run),
        },
        base_loading=predictor.base_loading,
        load_options={
            "dtype": str(options.dtype) if options.dtype else None,
            "merge": options.merge, "attn": options.attn,
            "lora_scale": options.lora_scale, "temperature": options.temperature,
            "base_load_path_used": bool(options.base_load_path),
        },
        ablation=provenance,
    )
    from jevany.suite import write_json
    write_json(Path(args.out) / "report.json", report)


if __name__ == "__main__":
    main()
