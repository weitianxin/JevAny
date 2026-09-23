#!/usr/bin/env python3
"""Compare full-media evaluation with blank and shuffled-media controls."""
import argparse
from pathlib import Path

import numpy as np

from jevany.metrics import paired_bootstrap
from jevany.suite import digest, read_json, write_json


def load_result(path):
    path = Path(path)
    report = read_json(path / "report.json")
    coverage = report["coverage"]
    if coverage["requested_questions"] != coverage["evaluated_questions"]:
        raise ValueError(f"incomplete coverage in {path}")
    if coverage["rejected_records"] or coverage["truncated_records"]:
        raise ValueError(f"rejected or truncated records in {path}")
    return report, read_json(path / "rows.json")


def accuracy(rows):
    return float(np.mean([np.argmax(row["p"]) == row["label"] for row in rows]))


def scope(rows, task):
    return rows if task is None else [row for row in rows if row["task"] == task]


def compare(rows_by_mode, task, samples):
    scoped = {mode: scope(rows, task) for mode, rows in rows_by_mode.items()}
    scores = {mode: accuracy(rows) for mode, rows in scoped.items()}
    strongest = max(("blank", "shuffled"), key=scores.get)
    bootstrap = paired_bootstrap(
        scoped["full"], scoped[strongest], samples=samples, seed=20260923,
        metric="acc", aggregation="micro",
    )
    delta = scores["full"] - scores[strongest]
    return {
        "questions": len(scoped["full"]),
        "accuracy": scores,
        "strongest_ablation": strongest,
        "full_minus_strongest": delta,
        "paired_cluster_bootstrap": bootstrap,
        "media_sensitivity_gate": delta >= 0.05 and bootstrap["ci95"][0] > 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--full", required=True)
    parser.add_argument("--blank", required=True)
    parser.add_argument("--shuffled", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--release-equivalence")
    args = parser.parse_args()
    suite = Path(args.suite)
    loaded = {mode: load_result(getattr(args, mode)) for mode in ("full", "blank", "shuffled")}
    reports = {mode: value[0] for mode, value in loaded.items()}
    rows = {mode: value[1] for mode, value in loaded.items()}
    for field in (
        "suite_manifest_sha256", "split_sha256", "checkpoint", "base_loading",
        "load_options", "calibration", "media_files_verified", "suite_artifacts_verified",
    ):
        if any(reports[mode].get(field) != reports["full"].get(field) for mode in ("blank", "shuffled")):
            raise ValueError(f"full and ablation provenance differs for {field}")
    identifiers = {
        mode: {(row["id"], row["question"]) for row in values}
        for mode, values in rows.items()
    }
    if len({frozenset(value) for value in identifiers.values()}) != 1:
        raise ValueError("full and ablation results do not contain the same questions")
    tasks = sorted({row["task"] for row in rows["full"]})
    manifest = read_json(suite / "manifest.json")
    audit_path = suite / "media-audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else {}
    if reports["full"]["suite_manifest_sha256"] != digest(suite / "manifest.json"):
        raise ValueError("evaluation manifest digest does not match the current suite")
    release_equivalence = read_json(args.release_equivalence) if args.release_equivalence else None
    if release_equivalence:
        if release_equivalence.get("equivalent") is not True:
            raise ValueError("release equivalence report did not pass")
        artifacts = reports["full"]["checkpoint"]["artifacts"]
        source = release_equivalence["source"]
        expected = {
            "head.pt": source["head_sha256"],
            "adapter_model.safetensors": source["adapter_sha256"],
            "adapter_config.json": source["adapter_config_sha256"],
        }
        if any(artifacts.get(name) != sha256 for name, sha256 in expected.items()):
            raise ValueError("release equivalence source does not match the evaluated checkpoint")
        evaluation_temperature = reports["full"]["load_options"]["temperature"]
        if evaluation_temperature != release_equivalence["evaluation_temperature"]:
            raise ValueError("release equivalence temperature does not match the evaluation")
    label_distribution = manifest.get("label_distribution") or {}
    result = {
        "protocol": {
            "full": "native media",
            "blank": "same questions and options with a neutral blank asset",
            "shuffled": "same questions and options with media rotated within each source",
            "media_sensitivity_gate": "full accuracy exceeds the strongest ablation by at least 5 points and the paired media-group bootstrap 95% lower bound is above zero",
            "bootstrap_samples": args.samples,
        },
        "suite": {
            "name": manifest["name"],
            "manifest_sha256": digest(suite / "manifest.json"),
            "media_audit_sha256": digest(audit_path) if audit_path.is_file() else None,
            "records": manifest["files"]["development.jsonl"]["records"],
            "unique_media": manifest.get("media_audit", {}).get("unique_sha256"),
            "base_revisions": manifest.get("base_revisions"),
            "licenses": manifest.get("licenses"),
            "media_audit": manifest.get("media_audit"),
            "overlap_counts": {
                "exact_media_overlap": audit.get("exact_sha256_overlap"),
                "perceptual_media_overlap": len(audit.get("near_duplicate_matches", [])),
                "question_option_overlap": len(audit.get("question_option_matches", [])),
                "source_id_overlap": len(audit.get("source_id_overlap", [])),
            },
            "data_quality": manifest.get("data_quality"),
            "artifacts": manifest.get("artifacts"),
            "label_distribution": label_distribution,
            "task_macro_random_baseline": float(np.mean([
                value["random_baseline"] for value in label_distribution.values()
            ])) if label_distribution else None,
            "task_macro_majority_position_baseline": float(np.mean([
                value["majority_position_baseline"] for value in label_distribution.values()
            ])) if label_distribution else None,
        },
        "coverage": {mode: report["coverage"] for mode, report in reports.items()},
        "evaluation": {
            "checkpoint_artifacts": reports["full"]["checkpoint"]["artifacts"],
            "base_loading": reports["full"]["base_loading"],
            "load_options": reports["full"]["load_options"],
            "calibration": reports["full"]["calibration"],
            "controls": {mode: reports[mode]["ablation"] for mode in ("full", "blank", "shuffled")},
            "release_equivalence": {
                "sha256": digest(Path(args.release_equivalence)),
                "report": release_equivalence,
            } if args.release_equivalence else None,
        },
        "overall": compare(rows, None, args.samples),
        "tasks": {task: compare(rows, task, args.samples) for task in tasks},
        "full_metrics": reports["full"]["clean"],
    }
    write_json(args.out, result)
    print(f"{manifest['name']}: full={result['overall']['accuracy']['full']:.3f}, "
          f"strongest={result['overall']['accuracy'][result['overall']['strongest_ablation']]:.3f}, "
          f"media_sensitivity_gate={result['overall']['media_sensitivity_gate']}")


if __name__ == "__main__":
    main()
