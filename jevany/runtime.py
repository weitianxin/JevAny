# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""One inference runtime shared by Python applications and the HTTP server."""
import os
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .api import SystemOneRequest, output_tokens, to_answers, to_record, validate_response
from .checkpoint import Checkpoint, LoadOptions
from .client import DecisionClient
from .device import default_device, sync

DEFAULT_CHECKPOINT = "tianxinwei/JevAny-27B-SFT"
INFER_MAX_STATE, INFER_MAX_BRANCH, INFER_MAX_PACKED = 8192, 8192, 8192
PREFIX_CACHE_SIZE = int(os.environ.get("JEVANY_PREFIX_CACHE", "4"))
PREFIX_MIN_TOKENS = int(os.environ.get("JEVANY_PREFIX_MIN_TOKENS", "384"))


@dataclass
class DecisionRuntime:
    checkpoint: Checkpoint
    tok: object
    model: object
    device: str
    model_id: str = "jevany-27b"
    lock: threading.Lock = field(default_factory=threading.Lock)
    prefix_cache: dict = field(default_factory=dict)
    prefix_hits: int = 0
    prefix_misses: int = 0

    def probs(self, record: dict) -> tuple[list[list[float]], dict]:
        """Encode and score under one lock, including the shared media processor."""
        with self.lock:
            encoding = self.model.encode(
                self.tok, record, max_state=INFER_MAX_STATE, max_branch=INFER_MAX_BRANCH, strict=True,
            )
            if len(encoding["ids"]) > INFER_MAX_PACKED:
                raise ValueError(f"request exceeds {INFER_MAX_PACKED} packed tokens: {len(encoding['ids'])}")
            state_tokens = encoding["seg"].count(0)
            key = (tuple(encoding["ids"][:state_tokens]), bool(encoding.get("option_isolation")))
            cache, hit = self.prefix_cache, False
            sync(self.device)
            start = time.perf_counter()
            eligible = PREFIX_CACHE_SIZE > 0 and state_tokens >= PREFIX_MIN_TOKENS and not encoding.get("multimodal")
            if eligible and key in cache:
                prefix = cache.pop(key)
                probabilities = self.model.probs_with_prefix(encoding, prefix)
                cache[key] = prefix
                self.prefix_hits += 1
                hit = True
            elif eligible:
                probabilities, prefix = self.model.probs_and_prefix(encoding)
                cache[key] = prefix
                while len(cache) > PREFIX_CACHE_SIZE:
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
        record, metadata = to_record(request)
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
    ) -> "JevModel":
        """Load a local run or Hugging Face adapter ID (optionally ``owner/repo@revision``).

        The full backbone must fit on the selected device. ``dtype`` accepts
        fp32, fp16 or bf16; omission uses the checkpoint/environment settings.
        Files used by native media requests are trusted local paths.
        """
        import torch

        device = device or default_device()
        if device not in ("cpu", "mps", "cuda"):
            raise ValueError("device must be cpu, mps or cuda")
        options = options or LoadOptions.from_env()
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
            ) else Path(source).name)
        tokenizer, model = loaded.load(device, options)
        return cls(DecisionRuntime(loaded, tokenizer, model, device, model_name))

    def __call__(self, request: SystemOneRequest | dict) -> dict[str, Any]:
        validated = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        return self.runtime.answer(validated)
