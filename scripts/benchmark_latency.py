"""Measure warmed local decision latency on a fixed, labelled suite.

Run with ``python -m scripts.benchmark_latency --run CHECKPOINT --suite SUITE
--out latency.json``. Pair with jevany.benchmark for accuracy and calibration.
"""
import argparse
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from jevany.checkpoint import LoadOptions
from jevany.device import sync
from jevany.predictors import LocalPredictor
from jevany.suite import digest, load_split, record_digest


def latency_summary(values):
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("latencies must be nonempty, finite and positive")
    return {
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "mean": float(np.mean(values)),
    }


def measure(records, predictor, device, warmup=16, repeats=3, seed=0):
    """Time serial requests, excluding warmup; fail if any record is rejected."""
    if not records:
        raise ValueError("latency panel is empty")
    if warmup < 1 or repeats < 1:
        raise ValueError("warmup and repeats must be positive")
    for index in range(warmup):
        predictor(records[index % len(records)])
    sync(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    samples = []
    rng = random.Random(seed)
    for repeat in range(repeats):
        order = list(range(len(records)))
        rng.shuffle(order)
        for index in order:
            record = records[index]
            sync(device)
            start = time.perf_counter()
            prediction = predictor(record)
            sync(device)
            elapsed = 1000 * (time.perf_counter() - start)
            samples.append({
                "id": record["_meta"]["id"], "repeat": repeat,
                "questions": len(record["questions"]),
                "input_tokens": prediction["input_tokens"],
                "forward_ms": float(prediction["latency_ms"]),
                "end_to_end_ms": elapsed,
            })
    forward = latency_summary([sample["forward_ms"] for sample in samples])
    end_to_end = latency_summary([sample["end_to_end_ms"] for sample in samples])
    return {
        "records": len(records), "measured_requests": len(samples),
        "warmup_requests": warmup, "repeats": repeats, "seed": seed,
        "concurrency": 1,
        "forward_ms": forward, "end_to_end_ms": end_to_end,
        "serial_requests_per_second": 1000 / end_to_end["mean"],
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None
        ),
        "samples": samples,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    parser.add_argument("--modality", choices=("all", "text", "image", "video"), default="text")
    parser.add_argument("--records", type=int, default=256, help="first N eligible records; 0 = all")
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.records < 0 or min(args.warmup, args.repeats) < 1:
        parser.error("records must be nonnegative; warmup and repeats must be positive")
    target = Path(args.out)
    if target.exists():
        parser.error("refusing to overwrite an existing latency report")
    records = load_split(args.suite, "development")
    if args.modality != "all":
        records = [
            row for row in records
            if {item["type"] for item in row.get("media", [])} == (
                set() if args.modality == "text" else {args.modality}
            )
        ]
    if args.records:
        records = records[:args.records]
    if not records:
        parser.error(f"suite has no {args.modality} records")
    identities = [row["_meta"]["id"] for row in records]
    if len(set(identities)) != len(identities):
        parser.error("latency panel contains duplicate record IDs")
    options = LoadOptions.from_env()
    predictor = LocalPredictor(args.run, args.device, options)
    report = measure(records, predictor, args.device, args.warmup, args.repeats, args.seed)
    report.update(
        checkpoint=args.run, base_loading=predictor.base_loading,
        suite_manifest_sha256=digest(Path(args.suite) / "manifest.json"),
        split_sha256=digest(Path(args.suite) / "development.jsonl"),
        panel_id_sha256=record_digest(identities), panel_ids=identities,
        modality=args.modality,
        timing={
            "forward": "ModelPredictor forward, probability/logit CPU transfer and output construction; excludes encoding",
            "end_to_end": "local preprocessing, media decode, tokenization, forward and output construction",
            "excluded": "model loading, warmup, network transport and report writing",
            "throughput": "serial reciprocal mean latency; not saturated server throughput",
        },
        runtime={
            "device": args.device, "python": platform.python_version(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf32": torch.backends.cuda.matmul.allow_tf32,
            "flash_sdp": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp": torch.backends.cuda.mem_efficient_sdp_enabled(),
        },
        load_options={
            "dtype": str(options.dtype) if options.dtype else "checkpoint",
            "merge": options.merge, "attn": options.attn,
            "lora_scale": options.lora_scale, "temperature": options.temperature,
        },
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({key: report[key] for key in ("measured_requests", "forward_ms", "end_to_end_ms")}))


if __name__ == "__main__":
    main()
