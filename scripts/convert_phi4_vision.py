"""Convert Microsoft's original Phi-4 Multimodal weights to a native vision base.

Run once, then pass the output directory to ``jevany train --multimodal``.
The original, pretrained vision LoRA is merged before JevAny adds its own LoRA.
The output is for image/text decisions; it does not retain the speech LoRA.

Configuration/key conversion follows Hugging Face's Apache-2.0 converter:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/phi4_multimodal/convert_phi4_multimodal_weights_to_hf.py
"""
# Configuration and key mapping adapted from Hugging Face Transformers.
# Copyright 2025 The HuggingFace Inc. team. Licensed under Apache-2.0.
import argparse
import json
from pathlib import Path
import re

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from tokenizers import Tokenizer
from transformers import (
    Phi4MultimodalAudioConfig, Phi4MultimodalConfig, Phi4MultimodalFeatureExtractor,
    Phi4MultimodalForCausalLM, Phi4MultimodalImageProcessor, Phi4MultimodalProcessor,
    Phi4MultimodalVisionConfig, PreTrainedTokenizerFast,
)


KEY_MAPPING = {
    r"\.feed_forward_(in|out)\.net\.0\.linear": r".feed_forward_\1.gate_up_proj",
    r"\.feed_forward_(in|out)\.net\.2": r".feed_forward_\1.down_proj",
    r"\.self_attn\.linear_(q|k|v)": r".self_attn.\1_proj",
    r"\.self_attn\.linear_out": ".self_attn.o_proj",
    r"\.image_embed\.img_projection\.0": ".image_embed.img_projection_up",
    r"\.image_embed\.img_projection\.2": ".image_embed.img_projection_down",
    r"\.image_embed\.glb_GN": ".image_embed.global_img_feature_extensor",
    r"\.image_embed\.sub_GN": ".image_embed.sub_img_feature_extensor",
    r"\.audio_projection\.speech\.0": ".up_proj_for_speech",
    r"\.audio_projection\.speech\.2": ".down_proj_for_speech",
    r"\.audio_projection\.vision\.0": ".up_proj_for_vision_speech",
    r"\.audio_projection\.vision\.2": ".down_proj_for_vision_speech",
}


def native_config(original: dict) -> Phi4MultimodalConfig:
    original = json.loads(json.dumps(original))
    for key in ("_name_or_path", "architectures", "auto_map", "model_type", "vision_lora", "speech_lora",
                "transformers_version", "_attn_implementation"):
        original.pop(key, None)
    embeddings = original.pop("embd_layer")
    audio = original.pop("audio_processor")["config"]
    for key in ("activation_checkpointing", "cnn_layer_norm", "input_layer", "batch_norm",
                "encoder_embedding_config", "ext_pw_kernel_size", "bias_in_glu", "causal"):
        audio.pop(key, None)
    for old, new in (("attention_dim", "hidden_size"), ("attention_heads", "num_attention_heads"),
                     ("linear_units", "intermediate_size")):
        audio[new] = audio.pop(old)
    audio["nemo_conv_channels"] = audio.pop("nemo_conv_settings")["conv_channels"]
    audio["bias_max_distance"] = audio.pop("relative_attention_bias_args")["t5_bias_max_distance"]
    audio["downsample_rate"] = embeddings["audio_embd_layer"]["downsample_rate"]
    return Phi4MultimodalConfig(
        **original, vision_config=Phi4MultimodalVisionConfig(
            crop_size=embeddings["image_embd_layer"]["crop_size"]),
        audio_config=Phi4MultimodalAudioConfig(**audio),
    )


def convert_weights(weights: dict, scale: float) -> tuple[dict, int]:
    merged = 0
    converted = {}
    for key, value in weights.items():
        if ".lora_" in key:
            continue
        if key.endswith(".base_layer.weight"):
            prefix = key.removesuffix("base_layer.weight")
            a, b = prefix + "lora_A.vision.weight", prefix + "lora_B.vision.weight"
            if a not in weights or b not in weights:
                raise ValueError(f"missing pretrained vision LoRA for {key}")
            value = (value.float() + (weights[b].float() @ weights[a].float()) * scale).to(value.dtype)
            merged += 1
        key = key.replace(".base_layer.", ".")
        for pattern, replacement in KEY_MAPPING.items():
            key = re.sub(pattern, replacement, key)
        converted[key] = value
    if not merged:
        raise ValueError("no pretrained vision LoRA found; expected the original Phi-4 Multimodal checkpoint")
    return converted, merged


def convert_processor(source: Path, output: Path) -> None:
    backend = json.loads((source / "tokenizer.json").read_text())
    renames = {"<|endoftext10|>": "<|image|>", "<|endoftext11|>": "<|audio|>"}
    for old, new in renames.items():
        backend["model"]["vocab"][new] = backend["model"]["vocab"].pop(old)
    for token in backend["added_tokens"]:
        token["content"] = renames.get(token["content"], token["content"])
    settings = json.loads((source / "tokenizer_config.json").read_text())
    special = {key: settings[key] for key in ("bos_token", "eos_token", "pad_token", "unk_token")
               if settings.get(key) is not None}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(backend)), **special,
        extra_special_tokens={"image_token": "<|image|>", "audio_token": "<|audio|>"},
    )
    if (tokenizer.image_token_id, tokenizer.audio_token_id) != (200010, 200011):
        raise ValueError("unexpected Phi-4 media token IDs")
    processor = Phi4MultimodalProcessor(
        image_processor=Phi4MultimodalImageProcessor(), audio_processor=Phi4MultimodalFeatureExtractor(),
        tokenizer=tokenizer,
    )
    processor.save_pretrained(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument("--revision", help="pin a Hub commit; required for Hub sources")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("--out must not already exist")
    source = Path(args.source)
    if not source.is_dir():
        if not args.revision:
            parser.error("--revision is required for a Hub source")
        source = Path(snapshot_download(args.source, revision=args.revision,
                                       allow_patterns=["*.json", "*.safetensors"]))
    original = json.loads((source / "config.json").read_text())
    if original.get("model_type") != "phi4mm":
        parser.error("expected the original model_type='phi4mm'")
    weights = {}
    for shard in sorted(source.glob("*.safetensors")):
        weights.update(load_file(shard))
    vision = original["vision_lora"]
    converted, merged = convert_weights(weights, vision["lora_alpha"] / vision["r"])
    del weights
    with torch.device("meta"):
        model = Phi4MultimodalForCausalLM(native_config(original))
    missing, unexpected = model.load_state_dict(converted, strict=False, assign=True)
    if unexpected or (missing and not (missing == ["lm_head.weight"] and model.config.tie_word_embeddings)):
        raise ValueError(f"incomplete native conversion: missing={missing}, unexpected={unexpected}")
    model.tie_weights()
    model.save_pretrained(args.out)
    convert_processor(source, args.out)
    (args.out / "conversion.json").write_text(json.dumps({
        "source": args.source, "revision": args.revision, "native_model_type": model.config.model_type,
        "merged_vision_lora_layers": merged, "supported_media": ["image"],
        "speech_lora_retained": False,
    }, indent=2))
    print(f"Converted vision base: {args.out}; merged {merged} pretrained LoRA layers")


if __name__ == "__main__":
    main()
