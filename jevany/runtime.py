# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""One inference runtime shared by Python applications and the HTTP server."""
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .api import SystemOneRequest, output_tokens, to_answers, to_record, validate_response
from .checkpoint import Checkpoint, LoadOptions
from .client import DecisionClient
from .device import default_device, sync
from .inference import InferenceOptions
from .model import DecisionModel

DEFAULT_CHECKPOINT = "tianxinwei/JevAny-27B-SFT"


@dataclass
class DecisionRuntime:
    checkpoint: Checkpoint
    tok: object
    model: DecisionModel
    device: str
    model_id: str = "jevany"
    inference_options: InferenceOptions = field(default_factory=InferenceOptions)
    lock: Any = field(default_factory=threading.RLock, repr=False)
    prefix_cache: dict = field(default_factory=dict)
    prefix_hits: int = 0
    prefix_misses: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_name must be a nonempty string")

    @property
    def limits(self) -> dict[str, int]:
        options = self.inference_options
        window = self.model.inference_capabilities.context_window
        return {
            "state_tokens": min(options.max_state_tokens, window) if window else options.max_state_tokens,
            "branch_tokens": min(options.max_branch_tokens, window) if window else options.max_branch_tokens,
            "packed_tokens": options.max_packed_tokens,
        }

    @property
    def aliases(self) -> list[str]:
        return [] if self.model_id == "jevany-latest" else ["jevany-latest"]

    def describe(self) -> dict[str, Any]:
        """Describe the loaded model and effective settings without loading weights again."""
        with self.lock:
            capabilities = self.model.inference_capabilities
            return {
                "id": self.model_id, "aliases": self.aliases,
                "run": self.checkpoint.requested, "base": self.checkpoint.meta.base,
                "lora": self.checkpoint.meta.lora, "device": self.device,
                "temperature": self.model.head.temperature,
                "backbone_adapter": self.model.backbone_adapter,
                "branch_mode": self.model.branch_mode,
                "capabilities": asdict(capabilities), "limits": self.limits,
                "prefix_cache": {
                    "enabled": self.inference_options.prefix_cache_size > 0 and capabilities.prefix_cache,
                    "size": self.inference_options.prefix_cache_size,
                    "min_state_tokens": self.inference_options.prefix_min_tokens,
                    "hits": self.prefix_hits, "misses": self.prefix_misses,
                    "cached_states": len(self.prefix_cache),
                },
            }

    def clear_cache(self) -> None:
        """Release cached prefixes under the same lock as inference."""
        with self.lock:
            self.prefix_cache.clear()
            self.prefix_hits = self.prefix_misses = 0

    def probs(self, record: dict) -> tuple[list[list[float]], dict]:
        """Encode and score under one lock, including the shared media processor."""
        with self.lock:
            limits, options = self.limits, self.inference_options
            capabilities = self.model.inference_capabilities
            if record.get("media"):
                if any(item["type"] not in capabilities.media_types for item in record["media"]):
                    raise ValueError("checkpoint does not support the requested media type")
                maximum = capabilities.max_media_questions
                if maximum is not None and len(record["questions"]) > maximum:
                    raise ValueError(f"media requests support at most {maximum} question(s)")
            encoding = self.model.encode(
                self.tok, record, max_state=limits["state_tokens"], max_branch=limits["branch_tokens"], strict=True,
            )
            if len(encoding["ids"]) > limits["packed_tokens"]:
                raise ValueError(f"request exceeds {limits['packed_tokens']} packed tokens: {len(encoding['ids'])}")
            if capabilities.context_window is not None and max(encoding["pos"]) >= capabilities.context_window:
                raise ValueError(f"request exceeds backbone context window of {capabilities.context_window} tokens")
            state_tokens = encoding["seg"].count(0)
            key = (tuple(encoding["ids"][:state_tokens]), bool(encoding.get("option_isolation")))
            cache, hit = self.prefix_cache, False
            sync(self.device)
            start = time.perf_counter()
            eligible = (options.prefix_cache_size > 0 and capabilities.prefix_cache
                        and state_tokens >= options.prefix_min_tokens
                        and not record.get("media") and not encoding.get("multimodal"))
            if eligible and key in cache:
                prefix = cache.pop(key)
                probabilities = self.model.probs_with_prefix(encoding, prefix)
                cache[key] = prefix
                self.prefix_hits += 1
                hit = True
            elif eligible:
                probabilities, prefix = self.model.probs_and_prefix(encoding)
                cache[key] = prefix
                while len(cache) > options.prefix_cache_size:
                    cache.pop(next(iter(cache)))
                self.prefix_misses += 1
            else:
                probabilities = self.model.probs(encoding)
            sync(self.device)
            elapsed = time.perf_counter() - start
        return [p.tolist() for p in probabilities], {
            "tokens": len(encoding["ids"]), "state_tokens": state_tokens,
            "latency_ms": round(elapsed * 1000, 1), "prefix_cache_hit": hit,
        }

    def answer(self, request: SystemOneRequest) -> dict[str, Any]:
        if request.model not in (self.model_id, *self.aliases):
            raise ValueError(f"unknown model {request.model!r}; this deployment serves {self.model_id!r}")
        record, metadata = to_record(request)
        # Output accounting uses the same tokenizer as encoding; media processors
        # can mutate its settings, so serialize both operations together.
        with self.lock:
            probabilities, measurements = self.probs(record)
            answers = to_answers(probabilities, metadata)
            return validate_response(request, {
                "model": self.model_id, "answers": answers,
                "usage": {"input_tokens": measurements["tokens"], "output_tokens": output_tokens(self.tok, answers)},
                "latency_ms": measurements["latency_ms"],
            })


class JevModel(DecisionClient):
    """Load a checkpoint once and make decisions without starting an HTTP server."""

    def __init__(self, runtime: DecisionRuntime) -> None:
        self.runtime = runtime
        self.model_id = runtime.model_id

    @classmethod
    def from_pretrained(
        cls, checkpoint: str | Path = DEFAULT_CHECKPOINT, *,
        device: str | None = None, dtype: str | None = None,
        model_name: str | None = None, options: LoadOptions | None = None,
        inference_options: InferenceOptions | None = None,
    ) -> "JevModel":
        """Load a local run or Hugging Face adapter ID (optionally ``owner/repo@revision``).

        The full backbone must fit on the selected device. ``dtype`` accepts
        fp32, fp16 or bf16; omission uses the checkpoint/environment settings.
        Files used by native media requests are trusted local paths.
        """
        import torch

        device = default_device() if device is None else device
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps or cuda")
        options = options or LoadOptions.from_env()
        inference_options = inference_options or InferenceOptions.from_env()
        if model_name is not None and (not isinstance(model_name, str) or not model_name.strip()):
            raise ValueError("model_name must be a nonempty string")
        if dtype is not None:
            dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
            if dtype not in dtypes:
                raise ValueError("dtype must be fp32, fp16 or bf16")
            options = replace(options, dtype=dtypes[dtype])
        if device == "mps" and options.attn is None:
            options = replace(options, attn="sdpa")
        loaded = Checkpoint(checkpoint)
        if model_name is None:
            source = loaded.requested.partition("@")[0]
            model_name = ("jevany-27b" if source in (
                DEFAULT_CHECKPOINT, "tianxinwei/JevAny-27B-RLCR",
            ) else Path(source).name or Path(loaded.path).resolve().name)
        tokenizer, model = loaded.load(device, options)
        return cls(DecisionRuntime(loaded, tokenizer, model, device, model_name, inference_options))

    def describe(self) -> dict[str, Any]:
        """Return identity, backbone capabilities, effective limits and cache statistics."""
        return self.runtime.describe()

    def clear_cache(self) -> None:
        """Release this model's cached state prefixes."""
        self.runtime.clear_cache()

    def __call__(self, request: SystemOneRequest | dict) -> dict[str, Any]:
        validated = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        return self.runtime.answer(validated)
