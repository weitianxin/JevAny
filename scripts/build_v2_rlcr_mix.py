#!/usr/bin/env python3
"""Build a compact, stratified RLCR continuation suite from an admitted SFT suite."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from jevany.suite import digest, load_split, read_jsonl, semantic_hash, write_json, write_jsonl


REASONING_SOURCES = {
    "aqua_rat": ("deepmind/aqua_rat", "33301c6a050c96af81f63cad5562cb5363e88971", "Apache-2.0"),
    "medmcqa": ("openlifescienceai/medmcqa", "91c6572c454088bf71b679ad90aa8dffcd0d5868", "Apache-2.0"),
}


QUOTAS = {
    "preference": 5000,
    "agent": 5000,
    "many_choice": 6000,
    "hard_reasoning": 6000,
    "math_reasoning": 4000,
    "medical_reasoning": 4000,
    "image": 4000,
    "video": 2000,
    "core": 4000,
}


def bucket(record):
    source = record["_meta"]["source"]
    if source == "helpsteer3":
        return "preference"
    if source == "agent_tool":
        return "agent"
    if source == "qasc":
        return "many_choice"
    if source in ("arc", "commonsense"):
        return "hard_reasoning"
    if source == "aqua_rat":
        return "math_reasoning"
    if source == "medmcqa":
        return "medical_reasoning"
    if source in ("scienceqa", "aokvqa"):
        return "image"
    if source == "video_feedback":
        return "video"
    return "core"


def reasoning_record(source, identifier, state, options, answer, instruction, revision):
    if not 2 <= len(options) <= 26 or answer not in range(len(options)):
        raise ValueError("reasoning record must have 2-26 options and a valid answer index")
    keys = "abcdefghijklmnopqrstuvwxyz"[:len(options)]
    question = {"type": "choice", "instructions": instruction,
                "criteria": dict(zip(keys, options)), "label": keys[answer], "src": source}
    record = {"state": state, "questions": {"answer": question}}
    record["_meta"] = {
        "source": source,
        "variant": "clean",
        "id": f"{source}/{identifier}",
        "group_id": f"{source}/{identifier}",
        "row": str(identifier),
        "split": "train",
        "repo": REASONING_SOURCES[source][0],
        "revision": revision,
        "text_sha256": semantic_hash(record),
    }
    return record


def reasoning_rows(quotas, seed):
    from datasets import load_dataset

    output = []
    if quotas["math_reasoning"]:
        repo, revision, _ = REASONING_SOURCES["aqua_rat"]
        rows = load_dataset(repo, split="train", revision=revision)
        indices = list(range(len(rows)))
        random.Random(f"{seed}:aqua_rat").shuffle(indices)
        math_count = 0
        for index in indices:
            row = rows[index]
            options = [option.split(")", 1)[-1].strip() for option in row["options"]]
            answer = ord(row["correct"].lower()) - ord("a")
            if answer not in range(len(options)):
                continue
            output.append(reasoning_record(
                "aqua_rat", index, {"domain": "mathematical reasoning", "problem": row["question"]},
                options, answer, "Which option correctly solves the problem?", revision,
            ))
            math_count += 1
            if math_count == quotas["math_reasoning"]:
                break

    if quotas["medical_reasoning"]:
        repo, revision, _ = REASONING_SOURCES["medmcqa"]
        rows = load_dataset(repo, split="train", revision=revision)
        indices = list(range(len(rows)))
        random.Random(f"{seed}:medmcqa").shuffle(indices)
        medical_count = 0
        for index in indices:
            row = rows[index]
            try:
                answer = int(row["cop"])
            except (TypeError, ValueError):
                continue
            options = [row[key] for key in ("opa", "opb", "opc", "opd")]
            if answer not in range(4) or not all(isinstance(option, str) and option.strip() for option in options):
                continue
            output.append(reasoning_record(
                "medmcqa", row["id"],
                {"subject": row["subject_name"], "topic": row["topic_name"], "question": row["question"]},
                options, answer, "Which option is medically correct?", revision,
            ))
            medical_count += 1
            if medical_count == quotas["medical_reasoning"]:
                break
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--quotas", help="JSON object overriding the default bucket quotas")
    parser.add_argument("--exclude", action="append", default=[], help="JSONL whose content must not enter training")
    args = parser.parse_args()
    quotas = {**QUOTAS, **(json.loads(args.quotas) if args.quotas else {})}
    if set(quotas) != set(QUOTAS) or any(not isinstance(value, int) or value < 0 for value in quotas.values()):
        raise ValueError(f"quotas must be nonnegative integers for {sorted(QUOTAS)}")
    root = Path(args.out)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)

    pools = {name: [] for name in QUOTAS}
    excluded = {semantic_hash(row) for path in args.exclude for row in read_jsonl(path)}
    candidates = load_split(args.suite, "train") + reasoning_rows(quotas, args.seed)
    seen = set(excluded)
    for record in candidates:
        fingerprint = semantic_hash(record)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        pools[bucket(record)].append(record)
    selected = []
    for index, name in enumerate(QUOTAS):
        rows = list(pools[name])
        if len(rows) < quotas[name]:
            raise ValueError(f"{name}: requested {quotas[name]} records but only {len(rows)} are available")
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
        "reasoning_sources": {
            name: {"repo": repo, "revision": revision, "license": license_name}
            for name, (repo, revision, license_name) in REASONING_SOURCES.items()
        },
        "trainable_sources": sorted({row["_meta"]["source"] for row in selected}),
        "eval_only_sources": [],
        "holdout_sources": [],
        "parent_suite": {"path": str(Path(args.suite).resolve()),
                         "manifest_sha256": digest(Path(args.suite) / "manifest.json")},
        "excluded_files": {str(Path(path).resolve()): digest(path) for path in args.exclude},
        "quotas": quotas,
        "counts": {
            "bucket": dict(Counter(bucket(row) for row in selected)),
            "source": dict(Counter(row["_meta"]["source"] for row in selected)),
            "choice_options": dict(Counter(
                len(question["criteria"])
                for row in selected
                for question in row["questions"].values()
                if question["type"] == "choice"
            )),
        },
    })
    print(json.dumps({"records": len(selected), "buckets": Counter(bucket(row) for row in selected)}, indent=2))


if __name__ == "__main__":
    main()
