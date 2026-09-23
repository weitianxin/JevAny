#!/usr/bin/env python3
"""Freeze MMStar image and MVBench video decisions with media-level provenance."""
import argparse
import hashlib
import math
import re
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import Image, load_dataset
from huggingface_hub import hf_hub_download

from jevany.suite import digest, semantic_hash, write_json, write_jsonl


MMSTAR_REPO = "Lin-Chen/MMStar"
MMSTAR_REVISION = "bc98d668301da7b14f648724866e57302778ab27"
MVBENCH_REPO = "OpenGVLab/MVBench"
MVBENCH_REVISION = "230a2d4fac8900333c61754641c7a13e069ac9c6"
MVBENCH_TASKS = {
    "action_antonym": ("video/ssv2_video.zip", "ssv2_video"),
    "fine_grained_action": ("video/Moments_in_Time_Raw.zip", "Moments_in_Time_Raw/videos"),
    "egocentric_navigation": ("video/vlnqa.zip", "vlnqa"),
}
PLACEHOLDER_OPTIONS = {"", "nan", "null"}


def question_parts(text):
    parenthesized = list(re.finditer(r"(?m)^\s*\(([A-Z])\)\s*", text))
    if len(parenthesized) >= 2:
        prefix = text[:parenthesized[0].start()].strip()
        if "\nQuestion:" in prefix:
            prefix = prefix.rsplit("\nQuestion:", 1)[1].strip()
        prefix = re.sub(r"\n(?:Choices|Options):\s*$", "", prefix).strip()
        matches = []
        for index, match in enumerate(parenthesized):
            end = parenthesized[index + 1].start() if index + 1 < len(parenthesized) else len(text)
            matches.append((match.group(1), text[match.end():end].strip()))
        instructions = prefix
    else:
        try:
            instructions, options = text.rsplit("\nOptions:", 1)
        except ValueError as error:
            raise ValueError("MMStar question has no recognized choices block") from error
        matches = re.findall(
            r"(?:^|,\s*)([A-Z]):\s*(.*?)(?=,\s*[A-Z]:|$)",
            options.strip(), flags=re.DOTALL,
        )
    letters = [name for name, _ in matches]
    if not 2 <= len(matches) <= 26 or letters != list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(matches)]):
        raise ValueError(f"cannot parse MMStar options: {text!r}")
    return instructions.strip(), {name.lower(): value.strip() for name, value in matches}


def clean_options(criteria, label):
    """Drop missing-value placeholders and reject ambiguous questions."""
    cleaned = {
        key: value for key, value in criteria.items()
        if str(value).strip().casefold() not in PLACEHOLDER_OPTIONS
    }
    if label not in cleaned:
        return None, "gold_option_is_placeholder"
    if len(cleaned) < 2:
        return None, "fewer_than_two_options"
    normalized = [" ".join(str(value).casefold().split()) for value in cleaned.values()]
    if len(normalized) != len(set(normalized)):
        return None, "duplicate_option_descriptions"
    return cleaned, None


def safe_extract(archive, directory):
    directory = Path(directory).resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (directory / member.filename).resolve()
            try:
                target.relative_to(directory)
            except ValueError as error:
                raise ValueError(f"archive member escapes output directory: {member.filename}") from error
        source.extractall(directory)


def ensure_mvbench_assets(directory):
    directory = Path(directory)
    extracted = directory / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    for archive_name, _ in MVBENCH_TASKS.values():
        archive = Path(hf_hub_download(
            MVBENCH_REPO, archive_name, repo_type="dataset",
            revision=MVBENCH_REVISION, local_dir=directory,
        ))
        marker = extracted / f".{archive.stem}.complete"
        if not marker.exists():
            safe_extract(archive, extracted)
            marker.write_text(digest(archive) + "\n", encoding="utf-8")
    return extracted


def media_hashes(directory, workers=16):
    paths = sorted(path for path in Path(directory).rglob("*") if path.is_file())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        values = list(pool.map(digest, paths))
    return dict(zip(paths, values))


def metadata(record, source, identifier, media_sha256, **extra):
    return {
        "source": source,
        "variant": "clean",
        "id": identifier,
        "group_id": f"{source}/media-{media_sha256[:20]}",
        "split": "development",
        "media_sha256": [media_sha256],
        "text_sha256": semantic_hash(record),
        **extra,
    }


def mmstar_rows(output):
    dataset = load_dataset(MMSTAR_REPO, "val", split="val", revision=MMSTAR_REVISION)
    dataset = dataset.cast_column("image", Image(decode=False))
    media_dir = output / "media" / "mmstar"
    media_dir.mkdir(parents=True)
    rows, exclusions = [], Counter()
    for row in dataset:
        instructions, criteria = question_parts(row["question"])
        label = row["answer"].lower()
        criteria, reason = clean_options(criteria, label)
        if reason:
            exclusions[reason] += 1
            continue
        suffix = Path(row["meta_info"]["image_path"]).suffix.lower() or ".jpg"
        image_path = media_dir / f"{row['index']:04d}{suffix}"
        image_path.write_bytes(row["image"]["bytes"])
        identifier = f"mmstar/{row['index']}"
        record = {
            "state": {
                "benchmark": "MMStar",
                "category": row["category"],
                "subcategory": row["l2_category"],
            },
            "media": [{"type": "image", "uri": str(image_path.relative_to(output))}],
            "questions": {"answer": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
                "label": label,
                "src": f"mmstar_{row['category'].replace(' ', '_')}",
            }},
        }
        if record["questions"]["answer"]["label"] not in criteria:
            raise ValueError(f"invalid MMStar answer in row {row['index']}")
        record["_meta"] = metadata(
            record, "mmstar", identifier, digest(image_path),
            source_index=row["index"], source_dataset=row["meta_info"]["source"],
        )
        rows.append(record)
    return rows, dict(sorted(exclusions.items()))


def mvbench_rows(output, assets):
    media_dir = output / "media" / "mvbench"
    media_dir.mkdir(parents=True)
    rows = []
    for task, (_, prefix) in MVBENCH_TASKS.items():
        dataset = load_dataset(MVBENCH_REPO, task, split="train", revision=MVBENCH_REVISION)
        for index, row in enumerate(dataset):
            source = assets / prefix / row["video"]
            if not source.is_file():
                raise FileNotFoundError(f"missing MVBench video: {source}")
            suffix = source.suffix.lower()
            target = media_dir / task / f"{index:04d}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            keys = list("abcdefghijklmnopqrstuvwxyz"[:len(row["candidates"])])
            criteria = dict(zip(keys, row["candidates"]))
            try:
                label = keys[row["candidates"].index(row["answer"])]
            except ValueError as error:
                raise ValueError(f"MVBench answer is not a candidate: {task}/{index}") from error
            identifier = f"mvbench_{task}/{index}"
            record = {
                "state": {"benchmark": "MVBench", "task": task.replace("_", " ")},
                "media": [{"type": "video", "uri": str(target.relative_to(output))}],
                "questions": {"answer": {
                    "type": "choice",
                    "instructions": row["question"],
                    "criteria": criteria,
                    "label": label,
                    "src": f"mvbench_{task}",
                }},
            }
            record["_meta"] = metadata(
                record, f"mvbench_{task}", identifier, digest(target),
                source_index=index,
            )
            rows.append(record)
    return rows


def distribution(rows):
    by_task = {}
    for row in rows:
        question = row["questions"]["answer"]
        keys = list(question["criteria"])
        group = by_task.setdefault(question["src"], {
            "records": 0, "options": Counter(), "label_position": Counter(),
        })
        group["records"] += 1
        group["options"][len(keys)] += 1
        group["label_position"][keys.index(question["label"])] += 1
    result = {}
    for task, counts in by_task.items():
        total = counts["records"]
        probabilities = [value / total for value in counts["label_position"].values()]
        random_baseline = sum(count / option_count for option_count, count in counts["options"].items()) / total
        majority_baseline = max(probabilities)
        if len(counts["label_position"]) < 2:
            raise ValueError(f"degenerate label positions for {task}")
        result[task] = {
            "records": total,
            "option_counts": dict(sorted(counts["options"].items())),
            "label_position": dict(sorted(counts["label_position"].items())),
            "random_baseline": random_baseline,
            "majority_position_baseline": majority_baseline,
            "position_balance_warning": majority_baseline > random_baseline + 0.10,
            "label_position_entropy_nats": -sum(p * math.log(p) for p in probabilities),
        }
    return result


def freeze(output, rows, sources, licenses, reference_media=None, exclusions=None):
    path = output / "development.jsonl"
    write_jsonl(path, rows)
    new_hashes = {value for row in rows for value in row["_meta"]["media_sha256"]}
    reference_files = media_hashes(reference_media) if reference_media else {}
    reference_hashes = set(reference_files.values())
    overlap = sorted(new_hashes & reference_hashes)
    if overlap:
        raise ValueError(f"evaluation media overlaps reference media in {len(overlap)} exact hashes")
    group_hash = hashlib.sha256("\n".join(sorted(new_hashes)).encode()).hexdigest()
    write_json(output / "manifest.json", {
        "name": output.name,
        "description": "Native multimodal decision evaluation with media-level provenance",
        "files": {"development.jsonl": {
            "records": len(rows), "questions": len(rows), "sha256": digest(path),
        }},
        "base_revisions": sources,
        "licenses": licenses,
        "trainable_sources": [],
        "eval_only_sources": sorted({row["_meta"]["source"] for row in rows}),
        "holdout_sources": [],
        "media_audit": {
            "files": len(rows),
            "unique_sha256": len(new_hashes),
            "group_sha256": group_hash,
            "reference_files_compared": len(reference_files),
            "reference_unique_sha256": len(reference_hashes),
            "exact_reference_overlap": 0,
        },
        "label_distribution": distribution(rows),
        "data_quality": {
            "excluded_records": exclusions or {},
            "placeholder_options": 0,
            "duplicate_option_descriptions": 0,
            "gold_labels_missing_from_options": 0,
        },
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--kind", choices=("image", "video"), required=True)
    parser.add_argument("--assets", default="runs/mvbench-assets")
    parser.add_argument("--reference-media", help="media directory for an initial exact-hash overlap screen")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    if args.kind == "image":
        rows, exclusions = mmstar_rows(output)
        sources = {MMSTAR_REPO: MMSTAR_REVISION}
        licenses = {MMSTAR_REPO: "not specified by the dataset card; source images retain upstream terms"}
    else:
        assets = ensure_mvbench_assets(args.assets)
        rows = mvbench_rows(output, assets)
        sources = {MVBENCH_REPO: MVBENCH_REVISION}
        licenses = {MVBENCH_REPO: "MIT metadata; source videos are academic-research-only under upstream terms"}
        exclusions = {}
    freeze(output, rows, sources, licenses, args.reference_media, exclusions)
    print(f"wrote {len(rows)} records to {output}")


if __name__ == "__main__":
    main()
