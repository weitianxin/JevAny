#!/usr/bin/env python3
"""Freeze the MuSR complex-reasoning benchmark as a JevAny evaluation suite."""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from jevany.data import materialize
from jevany.model import MAX_PACKED, encode, load_tokenizer
from jevany.suite import digest, semantic_hash, write_json, write_jsonl


MUSR_REPO = "TAUR-Lab/MuSR"
MUSR_REVISION = "7c365b439a222150f317764d4f16ae6c96d7d94a"
MUSR_LICENSE = "CC-BY-4.0"
MUSR_TASKS = ("murder_mysteries", "object_placements", "team_allocation")


def musr_record(row, task, index):
    choices = ast.literal_eval(row["choices"])
    answer = int(row["answer_index"])
    if not 2 <= len(choices) <= 26 or answer not in range(len(choices)):
        raise ValueError(f"invalid MuSR row {task}/{index}")
    keys = list("abcdefghijklmnopqrstuvwxyz"[:len(choices)])
    questions = {"answer": {
        "type": "choice",
        "instructions": row["question"],
        "criteria": dict(zip(keys, choices)),
        "label": keys[answer],
        "src": f"musr_{task}",
    }}
    record = {
        "state": {"task": task.replace("_", " "), "narrative": row["narrative"]},
        "questions": questions,
    }
    identifier = f"musr_{task}/{index}"
    record["_meta"] = {
        "source": f"musr_{task}",
        "variant": "clean",
        "id": identifier,
        "group_id": identifier,
        "row": str(index),
        "split": "development",
        "repo": MUSR_REPO,
        "revision": MUSR_REVISION,
        "text_sha256": semantic_hash(record),
    }
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.8-27B")
    parser.add_argument("--tokenizer-revision", default="")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    dataset = load_dataset(MUSR_REPO, revision=MUSR_REVISION)
    rows = [musr_record(row, task, index)
            for task in MUSR_TASKS for index, row in enumerate(dataset[task])]
    tokenizer = load_tokenizer(args.tokenizer, revision=args.tokenizer_revision or None)
    token_lengths = []
    for row in rows:
        encoded = encode(tokenizer, materialize(row), max_state=MAX_PACKED, max_branch=MAX_PACKED, strict=True)
        if len(encoded["ids"]) > MAX_PACKED:
            raise ValueError(f"MuSR row exceeds the {MAX_PACKED}-token packed inference window: {row['_meta']['id']}")
        token_lengths.append(len(encoded["ids"]))
    path = output / "development.jsonl"
    write_jsonl(path, rows)
    write_json(output / "manifest.json", {
        "name": "jevany-musr-v1",
        "description": "MuSR complex reasoning evaluation and transductive test-time adaptation inputs",
        "files": {"development.jsonl": {
            "records": len(rows),
            "questions": len(rows),
            "sha256": digest(path),
        }},
        "base_revisions": {MUSR_REPO: MUSR_REVISION},
        "tokenizer": {"name": args.tokenizer, "revision": args.tokenizer_revision or None},
        "context": {"max_state": MAX_PACKED, "max_branch": MAX_PACKED, "max_packed": MAX_PACKED,
                    "maximum_observed": max(token_lengths)},
        "licenses": {MUSR_REPO: MUSR_LICENSE},
        "trainable_sources": [],
        "eval_only_sources": sorted({row["_meta"]["source"] for row in rows}),
        "holdout_sources": [],
        "counts": {"source": dict(Counter(row["_meta"]["source"] for row in rows))},
    })
    print(json.dumps({"records": len(rows), "sources": Counter(row["_meta"]["source"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
