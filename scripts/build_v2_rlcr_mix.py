#!/usr/bin/env python3
"""Build a compact, stratified RLCR continuation suite from an admitted SFT suite."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from jevany.suite import digest, load_split, write_json, write_jsonl


QUOTAS = {
    "preference": 5000,
    "agent": 5000,
    "hard_reasoning": 4000,
    "image": 3000,
    "video": 2000,
    "core": 3000,
}


def bucket(record):
    source = record["_meta"]["source"]
    if source == "helpsteer3":
        return "preference"
    if source == "agent_tool":
        return "agent"
    if source in ("arc", "qasc", "commonsense"):
        return "hard_reasoning"
    if source in ("scienceqa", "aokvqa"):
        return "image"
    if source == "video_feedback":
        return "video"
    return "core"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--quotas", help="JSON object overriding the default bucket quotas")
    args = parser.parse_args()
    quotas = {**QUOTAS, **(json.loads(args.quotas) if args.quotas else {})}
    if set(quotas) != set(QUOTAS) or any(not isinstance(value, int) or value < 0 for value in quotas.values()):
        raise ValueError(f"quotas must be nonnegative integers for {sorted(QUOTAS)}")
    root = Path(args.out)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)

    pools = {name: [] for name in QUOTAS}
    for record in load_split(args.suite, "train"):
        pools[bucket(record)].append(record)
    selected = []
    for index, name in enumerate(QUOTAS):
        rows = list(pools[name])
        random.Random(f"{args.seed}:{index}:{name}").shuffle(rows)
        selected.extend(rows[:quotas[name]])
    random.Random(args.seed).shuffle(selected)

    path = root / "train.jsonl"
    write_jsonl(path, selected)
    write_json(root / "manifest.json", {
        "name": "jevany-v2-rlcr",
        "seed": args.seed,
        "files": {"train.jsonl": {
            "records": len(selected),
            "questions": sum(len(row["questions"]) for row in selected),
            "sha256": digest(path),
        }},
        "base_revisions": {},
        "trainable_sources": sorted({row["_meta"]["source"] for row in selected}),
        "eval_only_sources": [],
        "holdout_sources": [],
        "parent_suite": {"path": str(Path(args.suite).resolve()),
                         "manifest_sha256": digest(Path(args.suite) / "manifest.json")},
        "quotas": quotas,
        "counts": {
            "bucket": dict(Counter(bucket(row) for row in selected)),
            "source": dict(Counter(row["_meta"]["source"] for row in selected)),
        },
    })
    print(json.dumps({"records": len(selected), "buckets": Counter(bucket(row) for row in selected)}, indent=2))


if __name__ == "__main__":
    main()
