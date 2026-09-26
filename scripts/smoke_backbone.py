"""Short real-backbone train/save/reload check, also usable under torchrun.

Run from the repository root:
    python -m scripts.smoke_backbone --base Qwen/Qwen3-0.6B --out runs/smoke-qwen

The evaluation probe intentionally reuses the tiny training fixture: this checks
optimization and serialization, not generalization or task quality.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import time

import torch

from jevany.backbones import decision_tokens
from jevany.checkpoint import Checkpoint
from jevany.data import load_records, materialize
from jevany.predictors import ModelPredictor
from jevany.model import tokenizer_of
from jevany.suite import digest, load_split, read_json, write_json, write_jsonl
from jevany.train import main as train


def prepare_fixture(directory: Path, media: str | None = None, mixed_text: bool = False) -> None:
    directory.mkdir(parents=True)
    rows = []
    for index in range(8):
        options = {"billing": "Payment problems", "shipping": "Delivery problems"}
        if index % 2:
            options = dict(reversed(list(options.items())))
        rows.append({
            "state": f"Case {index}: The customer was charged twice and needs an urgent refund today.",
            "questions": {
                "team": {"type": "choice", "instructions": "Choose the responsible team.",
                         "criteria": options, "label": "billing"},
                "urgent": {"type": "noul", "instructions": "Does this need urgent attention?", "label": True},
                "priority": {"type": "score", "instructions": "How urgent is this?",
                             "criteria": ["low", "high"], "label": 1},
            },
        })
    if media:
        from PIL import Image
        rows = []
        for color in ("red", "blue"):
            image = Image.new("RGB", (112, 112), color)
            if media == "image":
                image.save(directory / f"{color}.png")
            else:
                import av
                with av.open(str(directory / f"{color}.mp4"), "w") as output:
                    stream = output.add_stream("libx264", rate=4)
                    stream.width = stream.height = 112
                    stream.pix_fmt = "yuv420p"
                    for _ in range(8):
                        for packet in stream.encode(av.VideoFrame.from_image(image)):
                            output.mux(packet)
                    for packet in stream.encode():
                        output.mux(packet)
        for index in range(8):
            color = "red" if index % 2 == 0 else "blue"
            row = {"state": "Inspect the sample.", "questions": {
                "color": {"type": "choice", "instructions": "What is the sample's color?",
                          "criteria": {"red": "Red", "blue": "Blue"}, "label": color}},
                   "media": [{"type": media, "uri": f"{color}.{'png' if media == 'image' else 'mp4'}"}]}
            if mixed_text and index >= 6:
                row.pop("media")
                row["state"] = f"The sample is {color}."
            rows.append(row)
    write_jsonl(directory / "train.jsonl", rows)
    probe = load_records(directory / "train.jsonl")[:4]
    files = {}
    for split in ("calibration", "development"):
        path = directory / f"{split}.jsonl"
        write_jsonl(path, probe)
        files[path.name] = {"records": len(probe), "sha256": digest(path)}
    write_json(directory / "manifest.json", {
        "files": files, "holdout_sources": [],
        "purpose": "Repeated training fixture for optimization smoke testing; no generalization claim.",
    })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--head-lr", type=float, default=0.0001)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--branch-mode", choices=["auto", "packed", "rows"], default="auto")
    parser.add_argument("--media", choices=["image", "video"], help="exercise native media training")
    parser.add_argument("--mixed-text", action="store_true", help="include text-only rows in a media fixture")
    args = parser.parse_args(argv)
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    fixture = args.out / "fixture"
    if rank == 0:
        prepare_fixture(fixture, args.media, args.mixed_text)
    else:
        deadline = time.monotonic() + 60
        while not (fixture / "manifest.json").is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("rank 0 did not prepare the smoke fixture")
            time.sleep(0.1)
    topology = {"rank": rank, "local_rank": local_rank, "world_size": world_size,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if args.device == "cuda":
        torch.cuda.set_device(local_rank)
        properties = torch.cuda.get_device_properties(local_rank)
        topology.update(cuda_device=torch.cuda.current_device(), gpu=properties.name,
                        gpu_uuid=str(properties.uuid), gpu_memory=properties.total_memory)
    write_json(args.out / f"rank-{rank}.json", topology)
    print(json.dumps({"topology": topology}), flush=True)
    checkpoint_path = args.out / "checkpoint"
    command = [
        "--base", args.base, "--base-revision", args.revision,
        "--data", str(fixture / "train.jsonl"), "--out", str(checkpoint_path),
        "--device", args.device, "--weights-dtype", "bf16" if args.device == "cuda" else "fp32",
        "--dtype", "bf16" if args.device == "cuda" else "fp32",
        "--epochs", str(args.steps), "--max-steps", str(args.steps), "--batch", "2", "--accum", "1",
        "--lora", "4", "--head-dim", "32", "--lr", str(args.lr), "--head-lr", str(args.head_lr),
        "--checkpointing", "1", "--weight-decay", "0", "--seed", "17",
        "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0",
        "--branch-mode", args.branch_mode, "--eval-suite", str(fixture),
        "--eval-before-start", "--eval-every-steps", str(max(1, args.steps // 3)),
    ]
    if args.media:
        command.append("--multimodal")
    train(command)
    if rank:
        return
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
        # Match the evaluator used inside the trainer.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    history = [json.loads(line) for line in (checkpoint_path / "training_eval/history.jsonl").read_text().splitlines()]
    losses = [{"step": point["optimizer_step"], "nll": point["clean"]["nll"]} for point in history]
    checkpoint = Checkpoint(checkpoint_path)
    tokenizer, model = checkpoint.load(args.device)
    predictor = ModelPredictor(model, tokenizer, args.device)
    records = load_split(fixture, "development")
    saved_rows = read_json(checkpoint_path / f"training_eval/step-{args.steps:06d}/development/rows.json")
    expected = {(row["id"], row["question"]): row for row in saved_rows}
    maximum_delta = 0.0
    for record in records:
        result = predictor(record)
        for question_id, probabilities in result["probabilities"].items():
            row = expected[record["_meta"]["id"], question_id]
            maximum_delta = max(maximum_delta, max(abs(probabilities[key] - value)
                                                   for key, value in zip(row["keys"], row["p"])))
    encoded = model.encode(tokenizer, materialize(records[0]))
    cache_delta = None
    media_delta = None
    if args.media:
        second = model.encode(tokenizer, materialize(records[1]))
        media_delta = max(float((a - b).abs().max()) for a, b in zip(model.probs(encoded), model.probs(second)))
    else:
        direct = model.probs(encoded)
        first, prefix = model.probs_and_prefix(encoded)
        cached = model.probs_with_prefix(encoded, prefix)
        cache_delta = max(float((left - right).abs().max())
                          for left, right in zip(direct + direct, first + cached))
    tok = tokenizer_of(tokenizer)
    from safetensors.torch import load_file
    weights = load_file(checkpoint_path / "adapter_model.safetensors")
    finite = all(bool(torch.isfinite(value).all()) for value in weights.values())
    lora_updated = any("lora_B" in name and bool(value.abs().sum() > 0) for name, value in weights.items())
    report = {
        "base": args.base, "base_revision": args.revision, "model_type": model.lm.config.model_type,
        "branch_mode": model.branch_mode, "token_schema": tok.init_kwargs["jevany_token_schema"],
        "decision_token_ids": [tok.convert_tokens_to_ids(token) for token in decision_tokens(tok)],
        "backbone_adapter": model.backbone_adapter, "media": args.media, "mixed_text": args.mixed_text,
        "trained_delimiter_embeddings": model.special_embeddings, "steps": args.steps,
        "learning_rate": args.lr, "head_learning_rate": args.head_lr,
        "world_size": world_size, "losses": losses, "adapter_tensors_finite": finite,
        "lora_updated": lora_updated, "checkpoint_probability_max_delta": maximum_delta,
        "prefix_cache_probability_max_delta": cache_delta,
        "media_probability_max_delta": media_delta,
        "training": read_json(checkpoint_path / "training_metrics.json"),
        "scope": "Real pretrained weights; repeated tiny training fixture; no generalization claim.",
    }
    report["passed"] = (
        finite and lora_updated and all(math.isfinite(point["nll"]) for point in losses)
        and losses[-1]["nll"] < losses[0]["nll"] and maximum_delta < 1e-5
        and (cache_delta is None or cache_delta < 0.01)
        and (media_delta is None or media_delta > 0)
    )
    write_json(args.out / "smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("backbone smoke check failed; see smoke.json")


if __name__ == "__main__":
    main()
