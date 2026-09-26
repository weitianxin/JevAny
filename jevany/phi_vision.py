"""Native Transformers loading for Microsoft's Phi-4 Reasoning Vision weights.

The released model combines Phi3, SigLIP2's penultimate features and a two-layer
GELU projector. This adapter retains that layout without executing Hub Python.
Reference: microsoft/Phi-4-reasoning-vision-15B, modeling_phi4_visionr.py.
"""
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoTokenizer, BatchEncoding, LlavaConfig, LlavaModel, Phi3Config, PretrainedConfig,
    PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, Siglip2VisionConfig,
)
from transformers.models.siglip2.image_processing_pil_siglip2 import Siglip2ImageProcessorPil

from .backbones import VisionAdapter, prepare_tokenizer


KEY_MAPPING = {
    r"^model\.vision_tower\.vision_tower\.": "vision_tower.",
    r"^model\.mm_projector\.0\.": "multi_modal_projector.linear_1.",
    r"^model\.mm_projector\.2\.": "multi_modal_projector.linear_2.",
    r"^model\.(embed_tokens|layers|norm)\.": r"language_model.\1.",
}


def native_config(original: dict) -> LlavaConfig:
    if original.get("model_type") != "phi4-siglip":
        raise ValueError("phi_reasoning_vision requires a Phi-4 Reasoning Vision base")
    text = {key: value for key, value in original.items()
            if key not in ("model_type", "architectures", "auto_map", "vision_config")}
    if original.get("mm_projector_type", "mlp2x_gelu") != "mlp2x_gelu":
        raise ValueError("phi_reasoning_vision requires the released mlp2x_gelu projector")
    vision = {**original["vision_config"]}
    vision.setdefault("patch_size", 16)
    return LlavaConfig(text_config=Phi3Config(**text).to_dict(),
                       vision_config=Siglip2VisionConfig(**vision).to_dict(),
                       vision_feature_layer=-2, vision_feature_select_strategy="full",
                       projector_hidden_act="gelu", multimodal_projector_bias=True)


class PhiReasoningProcessor(ProcessorMixin):
    """Expand image placeholders before the shared decision-token layout."""

    def __init__(self, image_processor: Siglip2ImageProcessorPil, tokenizer: PreTrainedTokenizerBase,
                 image_token: str = "<image>", min_num_patches: int = 256, max_num_patches: int = 3600):
        if not 1 <= min_num_patches <= max_num_patches:
            raise ValueError("Phi patch limits must satisfy 1 <= min_num_patches <= max_num_patches")
        self.image_token = image_token
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        tokenizer.add_special_tokens({"extra_special_tokens": [image_token]},
                                     replace_extra_special_tokens=False)
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)

    def __call__(self, text: list[str], images: list[Image.Image], return_tensors: str = "pt") -> BatchEncoding:
        if not images or len(text) != 1 or text[0].count(self.image_token) != len(images):
            raise ValueError("Phi vision expects one text with one placeholder per image")
        if return_tensors != "pt":
            raise ValueError("Phi vision training requires return_tensors='pt'")
        patches = []
        for image in images:
            size = self.image_processor.patch_size
            budget = max(self.min_num_patches, min(self.max_num_patches,
                         max(1, (image.width // size) * (image.height // size))))
            patches.append(self.image_processor(images=[image], max_num_patches=budget,
                                                return_tensors="pt"))
        lengths = [int(item["pixel_attention_mask"].sum()) for item in patches]
        parts = text[0].split(self.image_token)
        expanded = "".join(part + self.image_token * length for part, length in zip(parts, lengths)) + parts[-1]
        batch = self.tokenizer([expanded], return_tensors="pt", return_token_type_ids=False)
        width = max(item["pixel_values"].shape[1] for item in patches)
        batch["pixel_values"] = torch.cat([
            torch.nn.functional.pad(item["pixel_values"], (0, 0, 0, width - item["pixel_values"].shape[1]))
            for item in patches
        ])
        batch["pixel_attention_mask"] = torch.cat([
            torch.nn.functional.pad(item["pixel_attention_mask"], (0, width - item["pixel_attention_mask"].shape[1]))
            for item in patches
        ])
        batch["spatial_shapes"] = torch.cat([item["spatial_shapes"] for item in patches])
        batch["image_token_mask"] = batch["input_ids"] == self.tokenizer.convert_tokens_to_ids(self.image_token)
        return batch


class PhiReasoningVisionAdapter(VisionAdapter):
    name = "phi_reasoning_vision"

    def load_preprocessor(self, name: str, revision: str | None = None) -> PhiReasoningProcessor:
        if Path(name).is_dir() and (Path(name) / "processor_config.json").exists():
            processor = PhiReasoningProcessor.from_pretrained(name)
        else:
            raw, _ = PretrainedConfig.get_config_dict(name, revision=revision)
            processor = PhiReasoningProcessor(
                Siglip2ImageProcessorPil(patch_size=raw["vision_config"].get("patch_size", 16)),
                AutoTokenizer.from_pretrained(name, revision=revision),
                min_num_patches=raw.get("min_num_patches", 256),
                max_num_patches=raw.get("max_num_patches", 3600),
            )
        prepare_tokenizer(processor.tokenizer)
        return processor

    def load_model(self, name: str, *, revision: str | None, dtype: torch.dtype,
                   attn: str) -> tuple[PreTrainedModel, PreTrainedModel]:
        raw, _ = PretrainedConfig.get_config_dict(name, revision=revision)
        model, info = LlavaModel.from_pretrained(
            name, revision=revision, config=native_config(raw), dtype=dtype,
            attn_implementation=attn, key_mapping=KEY_MAPPING, output_loading_info=True,
        )
        unexpected = set(info["unexpected_keys"]) - {"lm_head.weight"}
        if info["missing_keys"] or unexpected or info.get("mismatched_keys"):
            raise ValueError(f"incomplete Phi vision load: {info}")
        model.requires_grad_(False)
        return model.language_model, model

    def forward_media(self, language_model, multimodal_model, inputs: dict) -> torch.Tensor:
        ids = inputs["input_ids"]
        pixels = inputs["pixel_values"].to(dtype=multimodal_model.vision_tower.dtype)
        mask = inputs["pixel_attention_mask"]
        vision = multimodal_model.vision_tower(
            pixel_values=pixels, pixel_attention_mask=mask, spatial_shapes=inputs["spatial_shapes"],
            output_hidden_states=True,
        ).hidden_states[-2]
        features = multimodal_model.multi_modal_projector(vision)[mask.bool()]
        embeddings = language_model.get_input_embeddings()(ids)
        image_mask = inputs["image_token_mask"]
        if int(image_mask.sum()) != features.shape[0]:
            raise ValueError("Phi image placeholders do not match the native visual features")
        embeddings = embeddings.masked_scatter(image_mask.unsqueeze(-1), features.to(embeddings.dtype))
        return language_model(inputs_embeds=embeddings, attention_mask=inputs["attention_mask"],
                              use_cache=False).last_hidden_state
