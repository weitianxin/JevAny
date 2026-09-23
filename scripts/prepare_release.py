#!/usr/bin/env python3
"""Create a public checkpoint directory with portable metadata and a file manifest."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


BASE = "Qwen/Qwen3.8-27B"
COPY = ("adapter_model.safetensors", "adapter_config.json")
ARGUMENTS = (
    "epochs", "lr", "head_lr", "weight_decay", "lora", "rlcr", "rlcr_group_size",
    "rlcr_sigma_start", "rlcr_sigma_end", "rlcr_policy_w", "rlcr_ce_w", "accum", "batch", "dtype",
    "weights_dtype", "checkpointing", "option_isolation", "special_embeddings", "multimodal", "head_dim",
    "max_state", "max_branch", "max_packed",
    "lora_targets", "base_revision", "p_none", "p_none_distract", "p_distract", "p_none_pair",
    "seed", "max_steps", "eval_before_start", "eval_every_steps", "eval_records",
    "checkpoint_every_steps",
)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="source checkpoint directory")
    parser.add_argument("--out", required=True, help="new public checkpoint directory")
    parser.add_argument("--card", required=True, help="model card copied to README.md")
    parser.add_argument("--kind", required=True, choices=("sft", "rlcr"))
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--suite-name", default="jevany-v2")
    parser.add_argument("--data-name", default="jevany-rlcr-v2 (40,000 records)")
    parser.add_argument("--init-name", default="JevAny-27B-SFT")
    parser.add_argument("--calibration-name", default="jevany-v2/calibration (1,100 questions)")
    args = parser.parse_args()

    source, output = Path(args.run), Path(args.out)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for name in (*COPY, "head.pt"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    output.mkdir(parents=True)

    for name in COPY:
        shutil.copy2(source / name, output / name)
    config_path = output / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = args.base
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metadata = torch.load(source / "head.pt", map_location="cpu", weights_only=False)
    old_arguments = metadata.get("args", {})
    clean_arguments = {key: old_arguments[key] for key in ARGUMENTS if key in old_arguments}
    clean_arguments.update(
        base=args.base,
        suite=args.suite_name if args.kind == "sft" else None,
        data=None if args.kind == "sft" else args.data_name,
        init_from=None if args.kind == "sft" else args.init_name,
        eval_suite=args.suite_name,
        eval_transfer_suite="transfer-v9",
    )
    metadata["args"] = clean_arguments
    metadata["base"] = args.base
    metadata.pop("full_finetune", None)
    if metadata.get("init_source"):
        init_source = metadata["init_source"]
        metadata["init_source"] = {
            "init_from": args.init_name,
            "adapter_sha256": init_source["adapter_sha256"],
            "head_sha256": init_source["head_sha256"],
            "adapter_tensors": init_source["adapter_tensors"],
        }
    if metadata.get("temperature_fit"):
        metadata["temperature_fit"]["rows"] = args.calibration_name
    torch.save(metadata, output / "head.pt")

    shutil.copy2(args.card, output / "README.md")
    root = Path(__file__).resolve().parents[1]
    for name in ("LICENSE", "NOTICE", "ACKNOWLEDGEMENTS.md"):
        shutil.copy2(root / name, output / name)

    serialized = json.dumps({key: value for key, value in metadata.items() if key != "head"}, default=str)
    forbidden_metadata = ("/lustre", "/home/", "jaredpalmer", "kev-qwen", "qwen38-")
    found = [value for value in forbidden_metadata if value.lower() in serialized.lower()]
    if found:
        raise ValueError(f"private or legacy metadata remains: {found}")
    public_text = "\n".join((output / name).read_text(encoding="utf-8")
                            for name in ("README.md", "adapter_config.json"))
    forbidden_text = ("/lustre", "/home/", "decision-v7", "8,192", "v0.1")
    found = [value for value in forbidden_text if value.lower() in public_text.lower()]
    if found:
        raise ValueError(f"private or stale release text remains: {found}")

    manifest = {
        "format": "JevAny checkpoint v1",
        "base": args.base,
        "base_revision": metadata.get("base_revision"),
        "kind": args.kind,
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.name != "release-manifest.json":
            manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"prepared {output}")


if __name__ == "__main__":
    main()
