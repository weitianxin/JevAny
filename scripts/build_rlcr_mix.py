# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Build the fixed 8,192-record RLCR mixture without changing a frozen suite.

The SFT recipe is close to uniform over public sources. RLCR instead emphasizes the four weakest development sources,
keeps broad replay, and assigns the remaining examples to compositional, policy and explicit knowable/unknowable pairs.
"""
import argparse
import random
from collections import Counter
from pathlib import Path

from jevany.suite import digest, read_jsonl, write_json, write_jsonl


HARD = ("amazon", "sst5", "yelp", "mnli")
BROAD = ("agnews", "banking77", "boolq", "dbpedia14", "imdb", "trec")


def sample_source(rows, source, count, rng):
    pool = [row for row in rows if row["_meta"]["source"] == source]
    if count > len(pool):
        raise ValueError(f"requested {count} {source} rows from a pool of {len(pool)}")
    return rng.sample(pool, count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True, help="decision-v7 directory")
    parser.add_argument("--uncertainty", required=True, help="525-row knowable/unknowable JSONL")
    parser.add_argument("--transfer", required=True, help="held-out transfer development JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    suite_rows = read_jsonl(Path(args.suite) / "train.jsonl")
    uncertainty = read_jsonl(args.uncertainty)

    selected = []
    for source in HARD:
        selected.extend(sample_source(suite_rows, source, 720, rng))
    for source in BROAD:
        selected.extend(sample_source(suite_rows, source, 340, rng))
    selected.extend(sample_source(suite_rows, "compositional", 1024, rng))
    selected.extend(sample_source(suite_rows, "legacy_policy", 673, rng))
    selected.extend(uncertainty * 3)
    if len(selected) != 8192:
        raise AssertionError(f"RLCR mixture has {len(selected)} records, expected 8192")

    transfer_hashes = {row["_meta"].get("text_sha256") for row in read_jsonl(args.transfer)}
    overlap = {row["_meta"].get("text_sha256") for row in selected} & transfer_hashes
    overlap.discard(None)
    if overlap:
        raise ValueError(f"RLCR mixture overlaps transfer-v9 in {len(overlap)} text hashes")
    rng.shuffle(selected)
    write_jsonl(output, selected)
    counts = Counter(row["_meta"]["source"] for row in selected)
    write_json(output.with_suffix(".meta.json"), {
        "records": len(selected),
        "seed": args.seed,
        "sha256": digest(output),
        "sources": dict(sorted(counts.items())),
        "mixture": {
            "hard_public": 2880,
            "broad_public_replay": 2040,
            "compositional": 1024,
            "legacy_policy": 673,
            "uncertainty_pairs": 1575,
        },
        "inputs": {"suite": args.suite, "uncertainty": args.uncertainty, "transfer_overlap_check": args.transfer},
    })
    print(f"wrote {len(selected)} records to {output}")


if __name__ == "__main__":
    main()
