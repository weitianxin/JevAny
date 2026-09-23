#!/usr/bin/env python3
"""Create a clean suite by removing records flagged by a media-overlap audit."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from jevany.suite import digest, read_json, read_jsonl, write_json, write_jsonl
from scripts.build_multimodal_eval import distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, output = Path(args.suite), Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    audit = read_json(args.audit)
    excluded_paths = {
        str(Path(item["evaluation"]).resolve())
        for item in audit["near_duplicate_matches"]
    }
    excluded_ids = {
        item["evaluation"]["id"]
        for item in audit.get("question_option_matches", [])
    } | set(audit.get("source_id_overlap", []))
    rows, kept, excluded = read_jsonl(source / "development.jsonl"), [], []
    flagged_groups = set()
    for row in rows:
        paths = [str((source / item["uri"]).resolve()) for item in row.get("media", [])]
        if row["_meta"]["id"] in excluded_ids or any(path in excluded_paths for path in paths):
            flagged_groups.add(row["_meta"]["group_id"])
    for row in rows:
        (excluded if row["_meta"]["group_id"] in flagged_groups else kept).append(row)
    if not excluded:
        raise ValueError("audit did not exclude any records")
    output.mkdir(parents=True)
    copied = set()
    for row in kept:
        for item in row.get("media", []):
            relative = Path(item["uri"])
            if relative in copied:
                continue
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source / relative, target)
            except OSError:
                shutil.copy2(source / relative, target)
            copied.add(relative)
    split = output / "development.jsonl"
    write_jsonl(split, kept)
    parent = read_json(source / "manifest.json")
    unique_media = {value for row in kept for value in row["_meta"]["media_sha256"]}
    group_hash = hashlib.sha256("\n".join(sorted(unique_media)).encode()).hexdigest()
    excluded_path = output / "excluded.json"
    write_json(excluded_path, {
        "records": len(excluded),
        "groups": len(flagged_groups),
        "ids": [row["_meta"]["id"] for row in excluded],
        "source_suite": str(source),
        "source_audit": str(Path(args.audit)),
        "source_audit_sha256": digest(Path(args.audit)),
    })
    manifest = {
        **parent,
        "name": output.name,
        "description": parent["description"] + "; prior-exposure media and normalized question-option overlaps removed",
        "files": {"development.jsonl": {
            "records": len(kept), "questions": len(kept), "sha256": digest(split),
        }},
        "media_audit": {
            "parent_manifest_sha256": digest(source / "manifest.json"),
            "reference": audit.get("reference"),
            "reference_files_compared": audit.get("reference_files", audit.get("training_files")),
            "perceptual_hash": audit["perceptual_hash"],
            "near_duplicate_threshold": audit["near_duplicate_threshold"],
            "prior_exposure_records_removed": len(excluded),
            "flagged_media_paths": len(excluded_paths),
            "flagged_question_or_source_ids": len(excluded_ids),
            "exact_reference_overlap_after_filter": 0,
            "files": len(kept),
            "unique_sha256": len(unique_media),
            "group_sha256": group_hash,
        },
        "label_distribution": distribution(kept),
        "artifacts": {"excluded.json": {"sha256": digest(excluded_path)}},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"kept": len(kept), "excluded": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
