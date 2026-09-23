# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Trained checkpoints: a run directory or a Hub repo holding a LoRA adapter, `head.pt` and the tokenizer.

This is the one place that knows the layout of `head.pt` and how a checkpoint becomes a `DecisionModel`:
`jevany.serve`, `jevany.benchmark` and `jevany.train --init_from` all go through it.

    ck = Checkpoint("tianxinwei/JevAny-27B-RLCR")    # or a local run directory; `@tag` pins a Hub revision
    tok, model = ck.load("mps", LoadOptions.from_env())
    ck.meta.temperature                             # the calibration the checkpoint carries
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .model import DecisionModel, load_preprocessor

HUB_ID = re.compile(r"[\w.-]+/[\w.-]+(@[\w.-]+)?")


def is_hub_id(run):
    return not os.path.isdir(run) and HUB_ID.fullmatch(str(run)) is not None


def resolve_run(run):
    """Local run directory as given, or a Hub repo id such as tianxinwei/JevAny-27B-RLCR, optionally pinned to a revision."""
    if os.path.isdir(run):
        return str(run)
    from huggingface_hub import snapshot_download
    repo, _, revision = str(run).partition("@")
    return snapshot_download(repo, revision=revision or None, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"])


@dataclass
class Meta:
    """Contents of `head.pt`. Every reader gets the same defaults for fields older checkpoints did not write.
    `extra` keeps the rest of the file (training args, suite hash, init provenance, temperature fit) so a
    read-modify-write round trip loses nothing."""
    base: str
    head: dict | None = None
    base_revision: str | None = None
    lora: int = 0
    head_dim: int = 256
    option_isolation: bool = False
    special_embeddings: bool = False
    multimodal: bool = False
    weights_dtype: str = "fp32"
    temperature: float = 1.0
    holdout: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    KNOWN = ("base", "head", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "multimodal", "weights_dtype", "temperature", "holdout")

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in cls.KNOWN if k in d}, extra={k: v for k, v in d.items() if k not in cls.KNOWN})

    def to_dict(self):
        return {**self.extra, **{k: getattr(self, k) for k in self.KNOWN}}   # known fields win over a stray key in extra


def read_meta(run):
    return Meta.from_dict(torch.load(f"{run}/head.pt", map_location="cpu"))


def write_meta(run, meta):
    torch.save(meta.to_dict(), f"{run}/head.pt")


@dataclass(frozen=True)
class LoadOptions:
    """How a checkpoint is turned into a model. Defaults are the exact path every reported number uses; the fields
    are the same knobs the JEVANY_* environment variables expose to the command-line tools (see from_env).

    dtype        None = fp32 (bf16 when the checkpoint was trained with a bf16 backbone). bf16 halves memory for
                 serving large backbones; probabilities then differ from fp32 in the third decimal.
    merge        fold the LoRA into the base weights in fp32 before any cast. Exact in fp32; in bf16 it is faster (~15%)
                 and closer to fp32 than merging directly into bf16. Ignored for adapters that carry trained token embeddings.
    attn         attention backend; None = the model default (SDPA on CUDA, eager elsewhere). "sdpa" on MPS measured
                 parity with eager and is a few percent faster.
    lora_scale   WiSE-FT-style interpolation between base (0) and fine-tuned weights (1), at inference.
    temperature  None = the temperature the checkpoint carries (fitted by scripts/calibrate_checkpoint.py); 1.0 = raw logits.
    base_load_path
                 optional node-local mirror for base-model I/O. Checkpoint metadata still names the canonical base.
    """
    dtype: torch.dtype | None = None
    merge: bool = True
    attn: str | None = None
    lora_scale: float = 1.0
    temperature: float | None = None
    base_load_path: str | None = None

    @classmethod
    def from_env(cls, env=os.environ):
        """Read JEVANY_DTYPE, JEVANY_MERGE, JEVANY_ATTN, JEVANY_LORA_SCALE,
        JEVANY_TEMPERATURE, and JEVANY_BASE_LOAD_PATH at command-line entry points."""
        return cls(dtype={"bf16": torch.bfloat16, "fp16": torch.float16}.get(env.get("JEVANY_DTYPE", "")),
                   merge=env.get("JEVANY_MERGE", "1") != "0", attn=env.get("JEVANY_ATTN") or None,
                   lora_scale=float(env.get("JEVANY_LORA_SCALE", "1")),
                   temperature=float(env["JEVANY_TEMPERATURE"]) if env.get("JEVANY_TEMPERATURE") else None,
                   base_load_path=env.get("JEVANY_BASE_LOAD_PATH") or None)


class Checkpoint:
    def __init__(self, run):
        self.requested = str(run)                    # what the caller asked for (a Hub id stays a Hub id in labels)
        self.path = resolve_run(run)
        self.meta = read_meta(self.path)

    def file(self, name):
        return Path(self.path) / name

    def adapter_config(self):
        return json.loads(self.file("adapter_config.json").read_text(encoding="utf-8"))

    def load(self, device, opts=LoadOptions()):
        """-> (tokenizer, DecisionModel) in eval mode with the LoRA applied and the pointer head loaded."""
        meta = self.meta
        dtype, merge = opts.dtype or torch.float32, opts.merge
        if meta.weights_dtype == "bf16":
            # Keep a fp32 adapter unmerged when the checkpoint was trained over a bf16 backbone.
            dtype, merge = torch.bfloat16, False
        source, revision = meta.base, meta.base_revision
        if opts.base_load_path:
            if not Path(opts.base_load_path).is_dir():
                raise ValueError(f"base load path does not exist: {opts.base_load_path}")
            source, revision = opts.base_load_path, None
        tok = load_preprocessor(source, revision=revision, multimodal=meta.multimodal)
        from peft import PeftModel
        merge = merge and not self.adapter_config().get("trainable_token_indices")   # token-trained adapters stay unmerged
        m = DecisionModel(source, tok, device, lora=None, revision=revision, head_dim=meta.head_dim,
                          option_isolation=meta.option_isolation, dtype=torch.float32 if merge else dtype,
                          attn=opts.attn, multimodal=meta.multimodal)
        m.set_language_model(PeftModel.from_pretrained(m.lm, self.path, torch_device=str(device)).to(device))
        if opts.lora_scale != 1:
            for module in m.lm.modules():
                if isinstance(getattr(module, "scaling", None), dict):
                    for k in module.scaling: module.scaling[k] *= opts.lora_scale
            m.lora_scale = opts.lora_scale
        if merge: m.set_language_model(m.lm.merge_and_unload())
        if dtype != torch.float32: m.set_language_model(m.lm.to(dtype))
        m.head.load_state_dict(meta.head); m.eval()
        m.head.temperature = meta.temperature if opts.temperature is None else opts.temperature
        return tok, m

    COMPAT_FIELDS = ("base", "base_revision", "lora", "head_dim", "option_isolation", "special_embeddings", "multimodal")

    def warm_start(self, model, ours):
        """Delta training: load this checkpoint's adapter and pointer head into `model` (a fresh DecisionModel built with
        LoRA). `ours` is the Meta the new run will save; every architecture field is compared BEFORE loading, because peft
        loads matching keys silently and a half-loaded adapter still trains and still reports a loss. Returns provenance."""
        from peft import get_peft_model_state_dict, load_peft_weights, set_peft_model_state_dict
        from .suite import digest
        for name in self.COMPAT_FIELDS:
            theirs, mine = getattr(self.meta, name), getattr(ours, name)
            if theirs != mine and not (name == "base_revision" and None in (theirs, mine)):
                raise ValueError(f"--init_from {self.path}: {name} is {theirs!r} there and {mine!r} here")
        weights = load_peft_weights(self.path, device="cpu")
        have = set(get_peft_model_state_dict(model.lm))
        unexpected, missing = sorted(set(weights) - have), sorted(have - set(weights))
        if unexpected:
            raise ValueError(f"--init_from {self.path} carries {len(unexpected)} adapter tensors this model does not have (e.g. {unexpected[:2]}); check --lora_targets / --lora against its adapter_config.json")
        if missing:
            raise ValueError(f"--init_from {self.path} does not cover {len(missing)} of this model's adapter tensors (e.g. {missing[:2]}); check --lora_targets")
        set_peft_model_state_dict(model.lm, weights)
        model.head.load_state_dict(self.meta.head)
        return {"init_from": self.requested, "resolved": self.path, "adapter_sha256": digest(self.file("adapter_model.safetensors")),
                "head_sha256": digest(self.file("head.pt")), "adapter_tensors": len(weights)}


def load(run, device, opts=LoadOptions()):
    """Convenience: Checkpoint(run).load(device, opts)."""
    return Checkpoint(run).load(device, opts)
