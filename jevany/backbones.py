"""Backbone and tokenizer adapters for decision training.

The default adapter uses Transformers' base-model interface. Custom adapters can
subclass ``BackboneAdapter`` and be selected by an import path (``module:Class``)
in both the training recipe and the saved checkpoint.
"""
import importlib

import torch
from torch import nn
from transformers import (
    AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
    DynamicCache, PreTrainedModel, PreTrainedTokenizerBase,
)
from transformers.pytorch_utils import Conv1D


LEGACY_TOKENS = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
DECISION_TOKENS = ["<|jev_state|>", "<|jev_question|>", "<|jev_option|>", "<|jev_end_option|>", "<|jev_decide|>"]


def decision_tokens(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Return this tokenizer's persisted decision alphabet."""
    schema = getattr(tokenizer, "init_kwargs", {}).get("jevany_token_schema", "legacy")
    if schema not in ("legacy", "v1"):
        raise ValueError(f"unsupported JevAny tokenizer schema: {schema!r}")
    return LEGACY_TOKENS if schema == "legacy" else DECISION_TOKENS


def prepare_tokenizer(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Reuse legacy delimiters or add five model-independent special tokens."""
    vocabulary = tokenizer.get_vocab()
    schema = tokenizer.init_kwargs.get("jevany_token_schema")
    if schema is None:
        schema = "legacy" if all(token in vocabulary for token in LEGACY_TOKENS) else "v1"
        tokenizer.init_kwargs["jevany_token_schema"] = schema
        tokens = decision_tokens(tokenizer)
        added = [token for token in tokens if token not in vocabulary]
        tokenizer.add_special_tokens({"extra_special_tokens": tokens}, replace_extra_special_tokens=False)
        tokenizer.init_kwargs["jevany_added_tokens"] = added
    tokens = decision_tokens(tokenizer)
    vocabulary = tokenizer.get_vocab()
    if any(token not in vocabulary for token in tokens):
        raise ValueError("saved tokenizer is missing its JevAny decision tokens")
    if len({tokenizer.convert_tokens_to_ids(token) for token in tokens}) != len(tokens):
        raise ValueError("JevAny decision tokens must have distinct token IDs")
    return tokenizer


def prepare_embeddings(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Resize the base and initialize newly added delimiters deterministically.

    Only these rows need training; PEFT saves their learned values in the adapter.
    Existing vocabulary rows, including a padded vocabulary tail, are retained.
    """
    embedding = model.get_input_embeddings()
    if not isinstance(embedding, nn.Embedding):
        raise ValueError("backbone must expose an nn.Embedding via get_input_embeddings()")
    added = tokenizer.init_kwargs.get("jevany_added_tokens", [])
    ids = [tokenizer.convert_tokens_to_ids(token) for token in added]
    size = max(tokenizer.get_vocab().values()) + 1
    if size > embedding.num_embeddings and not ids:
        raise ValueError("tokenizer vocabulary exceeds base embeddings without added-token metadata")
    if ids:
        with torch.no_grad():
            initial = embedding.weight.mean(dim=0)
            if size > embedding.num_embeddings:
                model.resize_token_embeddings(size, mean_resizing=False)
            model.get_input_embeddings().weight[ids] = initial
    return ids


class BackboneAdapter:
    """Transformers text backbone contract; override methods for other layouts.

    Models expose ``get_input_embeddings`` and return ``last_hidden_state`` from
    token IDs, position IDs and an attention mask. The default execution uses
    independent causal rows unless packed-mask support is known.
    """

    def load_preprocessor(self, name: str, revision: str | None = None):
        return prepare_tokenizer(AutoTokenizer.from_pretrained(name, revision=revision))

    def load_model(self, name: str, *, revision: str | None, dtype: torch.dtype,
                   attn: str) -> tuple[PreTrainedModel, PreTrainedModel | None]:
        config = AutoConfig.from_pretrained(name, revision=revision)
        if config.is_encoder_decoder:
            raise ValueError("the text adapter requires a decoder-only text base; select a suitable backbone_adapter")
        causal = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn)
        if causal.base_model is causal:
            raise ValueError("causal model does not expose its base_model; provide a custom backbone_adapter")
        return causal.base_model, None

    def supports_packed(self, config) -> bool:
        layer_types = set(getattr(config, "layer_types", None) or [])
        return (
            config.model_type in {"qwen2", "qwen3", "llama", "mistral", "phi3", "gemma", "gemma2", "gemma3_text"}
            and not layer_types.difference({"full_attention"})
            and not getattr(config, "sliding_window", None)
        )

    def lora_modules(self, model: nn.Module, preset: str, explicit: str = "") -> str | list[str]:
        """Resolve a preset against actual modules, including fused projections."""
        if preset not in ("all", "dense", "attn", "qv"):
            raise ValueError("lora_targets must be all, dense, attn, or qv")
        linear = {name for name, module in model.named_modules() if isinstance(module, (nn.Linear, Conv1D))}
        if explicit:
            targets = [name.strip() for name in explicit.split(",")]
            missing = [target for target in targets
                       if not target or not any(name == target or name.endswith("." + target) for name in linear)]
            if missing:
                raise ValueError(f"lora_target_modules do not name linear layers in this backbone: {missing}")
            return targets
        if preset in ("all", "dense"):
            # Retain the released Qwen adapter layout, including its dense ablation.
            if model.config.model_type.startswith("qwen"):
                names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
                if preset == "all":
                    names.update({"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"})
                targets = sorted(name for name in linear if name.rsplit(".", 1)[-1] in names)
                if targets:
                    return targets
            return "all-linear"
        names = {"q_proj", "v_proj"} if preset == "qv" else {
            "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "out_proj",
            "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
        }
        targets = sorted(name for name in linear if name.rsplit(".", 1)[-1] in names)
        if not targets:
            raise ValueError(f"lora_targets={preset!r} has no matching layers; use 'all' or explicit lora_target_modules")
        return targets

    def attach_language_model(self, multimodal_model, language_model) -> None:
        """Attach a PEFT-wrapped language model to a multimodal parent."""
        if multimodal_model is not None:
            raise ValueError("multimodal adapter must implement attach_language_model()")

    def frozen_modules(self, multimodal_model) -> tuple[nn.Module, ...]:
        return ()

    def new_cache(self, model: PreTrainedModel) -> DynamicCache:
        """Create a cache supporting deepcopy/reorder and, for packed mode, crop."""
        return DynamicCache(config=model.config)

    def encode_media(self, processor, record, **kwargs) -> dict:
        raise ValueError("this backbone adapter does not support media")


class QwenVisionAdapter(BackboneAdapter):
    """Existing native Qwen image/video path, separate from text adaptation."""

    def load_preprocessor(self, name: str, revision: str | None = None):
        processor = AutoProcessor.from_pretrained(name, revision=revision)
        prepare_tokenizer(processor.tokenizer)
        return processor

    def load_model(self, name: str, *, revision: str | None, dtype: torch.dtype,
                   attn: str) -> tuple[PreTrainedModel, PreTrainedModel | None]:
        model = AutoModel.from_pretrained(name, revision=revision, dtype=dtype, attn_implementation=attn)
        if not hasattr(model, "language_model") or not hasattr(model, "visual"):
            raise ValueError("qwen_vl requires a Qwen vision base with language_model and visual modules")
        return model.language_model, model

    def attach_language_model(self, multimodal_model, language_model) -> None:
        multimodal_model.language_model = language_model

    def frozen_modules(self, multimodal_model) -> tuple[nn.Module, ...]:
        return (multimodal_model.visual,)

    def encode_media(self, processor, record, **kwargs) -> dict:
        from .model import encode_multimodal
        return encode_multimodal(processor, record, **kwargs)


def get_backbone_adapter(name: str = "auto", *, multimodal: bool = False) -> BackboneAdapter:
    """Load a built-in adapter or an explicitly requested ``module:Class``."""
    if name == "auto":
        name = "qwen_vl" if multimodal else "text"
    builtins = {"text": BackboneAdapter, "qwen_vl": QwenVisionAdapter}
    if name in builtins:
        if multimodal and name == "text":
            raise ValueError("backbone_adapter='text' cannot be used with multimodal=true")
        return builtins[name]()
    module, separator, attribute = name.partition(":")
    if not separator or not module or not attribute:
        raise ValueError("backbone_adapter must be auto, text, qwen_vl, or module:Class")
    adapter_class = getattr(importlib.import_module(module), attribute)
    if not isinstance(adapter_class, type) or not issubclass(adapter_class, BackboneAdapter):
        raise ValueError(f"{name} must subclass jevany.backbones.BackboneAdapter")
    return adapter_class()
