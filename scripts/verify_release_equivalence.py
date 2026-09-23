#!/usr/bin/env python3
"""Verify that an evaluation checkpoint is inference-equivalent to a release."""
import argparse
import json
from pathlib import Path

import torch

from jevany.suite import digest, read_json, write_json


INFERENCE_HEAD_FIELDS = (
    "base_revision", "lora", "head_dim", "option_isolation",
    "special_embeddings", "multimodal", "weights_dtype",
)


def adapter_config_without_base(config):
    return {key: value for key, value in config.items() if key != "base_model_name_or_path"}


def model_name(value):
    return Path(str(value)).name.lower()


def head_inference_metadata(head):
    value = {field: head.get(field) for field in INFERENCE_HEAD_FIELDS}
    value["lora_targets"] = head.get("args", {}).get("lora_targets", "all")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source, release = Path(args.source), Path(args.release)
    source_head = torch.load(source / "head.pt", map_location="cpu", weights_only=False)
    release_head = torch.load(release / "head.pt", map_location="cpu", weights_only=False)
    evaluation_report_path = Path(args.evaluation_report)
    evaluation_report = read_json(evaluation_report_path)
    evaluation_temperature = evaluation_report["load_options"]["temperature"]
    if evaluation_temperature != evaluation_report["calibration"]["inference_temperature"]:
        raise ValueError("evaluation report has inconsistent inference temperatures")
    source_tensors, release_tensors = source_head["head"], release_head["head"]
    keys_match = source_tensors.keys() == release_tensors.keys()
    tensor_equality = {
        key: bool(torch.equal(source_tensors[key], release_tensors[key]))
        for key in sorted(source_tensors.keys() & release_tensors.keys())
    }
    source_adapter = digest(source / "adapter_model.safetensors")
    release_adapter = digest(release / "adapter_model.safetensors")
    source_adapter_config = json.loads((source / "adapter_config.json").read_text(encoding="utf-8"))
    release_adapter_config = json.loads((release / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config_match = adapter_config_without_base(source_adapter_config) == adapter_config_without_base(release_adapter_config)
    source_metadata = head_inference_metadata(source_head)
    release_metadata = head_inference_metadata(release_head)
    metadata_equality = {
        field: source_metadata[field] == release_metadata[field]
        for field in source_metadata
    }
    base_identity_match = (
        model_name(source_head["base"]) == model_name(release_head["base"])
        and release_adapter_config.get("base_model_name_or_path") == release_head["base"]
    )
    source_artifacts = evaluation_report["checkpoint"]["artifacts"]
    evaluation_checkpoint_match = (
        source_artifacts.get("head.pt") == digest(source / "head.pt")
        and source_artifacts.get("adapter_model.safetensors") == source_adapter
        and source_artifacts.get("adapter_config.json") == digest(source / "adapter_config.json")
    )
    equivalent = (
        keys_match and all(tensor_equality.values()) and source_adapter == release_adapter
        and adapter_config_match and all(metadata_equality.values()) and base_identity_match
        and evaluation_checkpoint_match
        and float(release_head["temperature"]) == evaluation_temperature
    )
    if not equivalent:
        raise ValueError("source checkpoint is not inference-equivalent to the release")
    write_json(args.out, {
        "equivalent": True,
        "source": {
            "head_sha256": digest(source / "head.pt"),
            "adapter_sha256": source_adapter,
            "adapter_config_sha256": digest(source / "adapter_config.json"),
            "embedded_temperature": float(source_head["temperature"]),
        },
        "release": {
            "repository": "tianxinwei/JevAny-27B-SFT",
            "head_sha256": digest(release / "head.pt"),
            "adapter_sha256": release_adapter,
            "adapter_config_sha256": digest(release / "adapter_config.json"),
            "embedded_temperature": float(release_head["temperature"]),
        },
        "evaluation_report": {
            "sha256": digest(evaluation_report_path),
            "checkpoint_artifacts_match_source": evaluation_checkpoint_match,
        },
        "evaluation_temperature": evaluation_temperature,
        "head_tensor_equality": tensor_equality,
        "head_metadata_equality": metadata_equality,
        "adapter_config_equality_except_base_path": adapter_config_match,
        "allowed_adapter_config_difference": ["base_model_name_or_path"],
        "base_identity_match": base_identity_match,
        "interpretation": "The release uses the evaluated LoRA and pointer head, the same inference settings, and the evaluation temperature.",
    })
    print(f"verified release equivalence: {args.out}")


if __name__ == "__main__":
    main()
