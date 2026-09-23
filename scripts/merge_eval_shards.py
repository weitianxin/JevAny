#!/usr/bin/env python3
"""Merge independent benchmark shards and recompute all metrics."""
import argparse
import json
from pathlib import Path

import numpy as np

from jevany.benchmark import summarize
from jevany.suite import load_split, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="directory containing shard-000, shard-001, ...")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, output = Path(args.root), Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    shards = sorted(path for path in root.glob("shard-*") if (path / "report.json").is_file())
    if not shards:
        raise ValueError(f"no completed shards under {root}")
    reports = [read_json(path / "report.json") for path in shards]
    expected = reports[0]["num_shards"]
    if len(shards) != expected or {report["shard_index"] for report in reports} != set(range(expected)):
        raise ValueError(f"incomplete shard set: found {len(shards)}, expected {expected}")
    provenance_fields = (
        "suite", "mode", "suite_manifest_sha256", "split_sha256",
        "checkpoint", "base_loading", "load_options", "ablation", "calibration",
        "suite_artifacts_verified",
    )
    for field in provenance_fields:
        if any(report.get(field) != reports[0].get(field) for report in reports[1:]):
            raise ValueError(f"shard provenance differs for {field}")
    rows, latencies, predictions = [], [], []
    for shard in shards:
        rows.extend(read_json(shard / "rows.json"))
        with (shard / "predictions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                predictions.append(item)
                latencies.append(item["prediction"]["latency_ms"])
    keys = [(row["id"], row["question"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate prediction rows across shards")
    expected_keys = {
        (record["_meta"]["id"], question)
        for record in load_split(reports[0]["suite"], "development")
        for question in record["questions"]
    }
    if set(keys) != expected_keys:
        missing, extra = expected_keys - set(keys), set(keys) - expected_keys
        raise ValueError(f"shards do not exactly cover suite: missing={len(missing)}, extra={len(extra)}")
    rows.sort(key=lambda row: (row["id"], row["question"]))
    predictions.sort(key=lambda item: item["id"])
    report = summarize(rows)
    report.update(
        coverage={
            "requested_records": sum(item["coverage"]["requested_records"] for item in reports),
            "requested_questions": sum(item["coverage"]["requested_questions"] for item in reports),
            "evaluated_records": sum(item["coverage"]["evaluated_records"] for item in reports),
            "evaluated_questions": sum(item["coverage"]["evaluated_questions"] for item in reports),
            "rejected_records": sum(item["coverage"]["rejected_records"] for item in reports),
            "truncated_records": sum(item["coverage"]["truncated_records"] for item in reports),
        },
        latency_ms={"median": float(np.median(latencies)), "p95": float(np.quantile(latencies, 0.95))},
        suite=reports[0]["suite"], mode=reports[0]["mode"], shards=expected,
        suite_manifest_sha256=reports[0].get("suite_manifest_sha256"),
        split_sha256=reports[0].get("split_sha256"),
        checkpoint=reports[0].get("checkpoint"),
        base_loading=reports[0].get("base_loading"),
        load_options=reports[0].get("load_options"),
        ablation=reports[0].get("ablation"),
        calibration=reports[0].get("calibration"),
        media_files_verified=reports[0].get("media_files_verified"),
        suite_artifacts_verified=reports[0].get("suite_artifacts_verified"),
    )
    output.mkdir(parents=True)
    write_json(output / "rows.json", rows)
    write_json(output / "report.json", report)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for item in predictions:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"records": len(predictions), "questions": len(rows), "clean": report["clean"]}, indent=2))


if __name__ == "__main__":
    main()
