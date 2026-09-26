"""Offline architecture coverage using real Transformers models with tiny weights."""
import json

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM, Gemma3TextConfig, GPT2Config, LlamaConfig, MistralConfig,
    Phi3Config, PreTrainedTokenizerFast, Qwen3Config, Qwen3_5TextConfig,
)

from jevany.backbones import DECISION_TOKENS, LEGACY_TOKENS, decision_tokens, prepare_tokenizer
from jevany.checkpoint import Checkpoint, Meta, write_meta
from jevany.model import DecisionModel, encode, load_tokenizer, user_tokens


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_tokenizer(legacy=False):
    words = ["[UNK]", "[PAD]", "[EOS]", "state", "choose", "yes", "no", "a", "b", "other"]
    if legacy:
        words += LEGACY_TOKENS
    raw = Tokenizer(WordLevel({token: i for i, token in enumerate(words)}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
        additional_special_tokens=LEGACY_TOKENS if legacy else [],
    )


def make_base(path, family, *, legacy=False):
    tokenizer = make_tokenizer(legacy)
    tokenizer.save_pretrained(path)
    common = dict(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=512, pad_token_id=1, eos_token_id=2, bos_token_id=2)
    configurations = {
        "qwen": lambda: Qwen3Config(**common, head_dim=8),
        "qwen35": lambda: Qwen3_5TextConfig(
            **common, head_dim=8, layer_types=["linear_attention", "full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=8, linear_value_head_dim=8,
        ),
        "llama": lambda: LlamaConfig(**common),
        "gemma": lambda: Gemma3TextConfig(**common, head_dim=8, query_pre_attn_scalar=8,
                                         sliding_window=16, layer_types=["sliding_attention", "full_attention"]),
        "mistral": lambda: MistralConfig(**common, sliding_window=16),
        # Padded vocab exercises newly added tokens that do not require a resize.
        "phi": lambda: Phi3Config(**{**common, "vocab_size": len(tokenizer) + 8},
                                  original_max_position_embeddings=512),
        "gpt2": lambda: GPT2Config(vocab_size=len(tokenizer), n_embd=32, n_layer=2, n_head=4,
                                   n_positions=512, pad_token_id=1, eos_token_id=2, bos_token_id=2),
    }
    model = AutoModelForCausalLM.from_config(configurations[family]())
    model.save_pretrained(path)
    return model.get_input_embeddings().weight.detach().clone()


RECORD = {"state": "state " * 20, "questions": [
    {"instr": "choose", "options": ["yes", "no"], "label": 0},
    {"instr": "choose other", "options": ["a", "b"], "label": 1},
]}


@pytest.mark.parametrize("family", ["qwen", "llama", "gemma", "mistral", "phi", "gpt2"])
def test_train_tokens_lora_isolation_cache_and_reload(tmp_path, family):
    torch.manual_seed(17)
    base = tmp_path / "base"
    original = make_base(base, family)
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8)
    encoded = model.encode(tokenizer, RECORD)
    token_ids = [tokenizer.convert_tokens_to_ids(token) for token in DECISION_TOKENS]
    assert model.special_embeddings
    assert set(token_ids).issubset(tokenizer.all_special_ids)
    assert not set(token_ids).intersection(user_tokens(tokenizer, " ".join(DECISION_TOKENS)))
    before = model.lm.get_input_embeddings().weight.detach().clone()
    initial_head = model.head.q.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.01)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(encoded)
        loss = sum(torch.nn.functional.cross_entropy(z[None], torch.tensor([q["label"]]))
                   for z, q in zip(logits, RECORD["questions"]))
        loss.backward()
        grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(gradient).all() for _, gradient in grads)
        assert any("lora_" in name and gradient.abs().sum() > 0 for name, gradient in grads)
        optimizer.step()
    after = model.lm.get_input_embeddings().weight.detach()
    assert not torch.equal(initial_head, model.head.q.weight)
    assert not torch.equal(before[token_ids], after[token_ids])
    ordinary = [i for i in range(len(original)) if i not in token_ids]
    assert torch.equal(original[ordinary], after[ordinary])
    model.eval()
    with torch.no_grad():
        expected = model.probs(encoded)
        rows = [z.softmax(-1) for z in model.forward_rows_batch([encoded])[0]]
        for left, right in zip(expected, rows):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)
        cached, prefix = model.probs_and_prefix(encoded)
        for _ in range(2):
            reused = model.probs_with_prefix(encoded, prefix)
            for left, first, right in zip(expected, cached, reused):
                torch.testing.assert_close(left, first, atol=2e-6, rtol=1e-5)
                torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)
        separate = {**RECORD, "questions": RECORD["questions"][1:]}
        torch.testing.assert_close(expected[1], model.probs(model.encode(tokenizer, separate))[0],
                                   atol=2e-6, rtol=1e-5)
    checkpoint = tmp_path / "checkpoint"
    model.lm.save_pretrained(checkpoint, save_embedding_layers=False)
    tokenizer.save_pretrained(checkpoint)
    write_meta(checkpoint, Meta(base=str(base), head=model.head.state_dict(), lora=2, head_dim=8,
                               special_embeddings=True, tokenizer_saved=True, branch_mode=model.branch_mode))
    from safetensors.torch import load_file
    saved = load_file(checkpoint / "adapter_model.safetensors")
    assert any("trainable_tokens" in name for name in saved)
    assert not any(tensor.shape == original.shape for tensor in saved.values())
    restored_tokenizer, restored = Checkpoint(checkpoint).load("cpu")
    assert restored_tokenizer.get_vocab() == tokenizer.get_vocab()
    for left, right in zip(expected, restored.probs(restored.encode(restored_tokenizer, RECORD))):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)


def test_legacy_tokens_and_checkpoint_stay_compatible(tmp_path):
    base = tmp_path / "base"
    make_base(base, "qwen", legacy=True)
    tokenizer = load_tokenizer(base)
    assert decision_tokens(tokenizer) == LEGACY_TOKENS
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8)
    assert not model.special_embeddings
    model.eval()
    expected = model.probs(encode(tokenizer, RECORD))
    checkpoint = tmp_path / "legacy"
    model.lm.save_pretrained(checkpoint)
    # A pre-adapter checkpoint does not have tokenizer_saved or adapter metadata.
    torch.save({"base": str(base), "head": model.head.state_dict(), "lora": 2, "head_dim": 8},
               checkpoint / "head.pt")
    tok, restored = Checkpoint(checkpoint).load("cpu")
    for left, right in zip(expected, restored.probs(encode(tok, RECORD))):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("family,dtype", [
    ("qwen35", torch.float32), ("llama", torch.float16), ("llama", torch.bfloat16),
])
def test_unsupported_prefix_cache_uses_full_forward_in_serving(tmp_path, family, dtype):
    from jevany.inference import InferenceOptions
    from jevany.checkpoint import LoadOptions
    from jevany.runtime import JevModel
    base = tmp_path / "base"
    make_base(base, family)
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8, dtype=dtype)
    assert model.lm.dtype == dtype
    encoded = model.encode(tokenizer, RECORD)
    assert not model.inference_capabilities.prefix_cache
    assert all(torch.isfinite(value).all() for value in model.probs(encoded))
    for call in (lambda: model.prefix(encoded), lambda: model.probs_and_prefix(encoded),
                 lambda: model.probs_with_prefix(encoded, None)):
        with pytest.raises(ValueError, match="prefix caching is not supported"):
            call()
    checkpoint = tmp_path / "checkpoint"
    model.lm.save_pretrained(checkpoint, save_embedding_layers=False)
    tokenizer.save_pretrained(checkpoint)
    write_meta(checkpoint, Meta(base=str(base), head=model.head.state_dict(), lora=2, head_dim=8,
                               special_embeddings=True, tokenizer_saved=True, branch_mode=model.branch_mode))
    runtime = JevModel.from_pretrained(
        checkpoint, device="cpu", options=LoadOptions(dtype=dtype, merge=False),
        inference_options=InferenceOptions(prefix_cache_size=1, prefix_min_tokens=0),
    ).runtime
    assert not runtime.describe()["prefix_cache"]["enabled"]
    first, first_info = runtime.probs(RECORD)
    second, second_info = runtime.probs(RECORD)
    assert first == second
    assert not first_info["prefix_cache_hit"] and not second_info["prefix_cache_hit"]
    assert not runtime.prefix_cache


def test_invalid_tokenizer_and_adapter_configuration(tmp_path):
    tokenizer = make_tokenizer()
    tokenizer.init_kwargs["jevany_token_schema"] = "v1"
    with pytest.raises(ValueError, match="missing"):
        prepare_tokenizer(tokenizer)
    tokenizer.init_kwargs["jevany_token_schema"] = "unknown"
    with pytest.raises(ValueError, match="schema"):
        prepare_tokenizer(tokenizer)
    base = tmp_path / "base"
    make_base(base, "phi")
    tok = load_tokenizer(base)
    with pytest.raises(ValueError, match="no matching layers"):
        DecisionModel(base, tok, "cpu", lora=2, lora_targets="qv")
    with pytest.raises(ValueError, match="do not name linear layers"):
        DecisionModel(base, tok, "cpu", lora=2, lora_target_modules="qkv_proj,missing")
    with pytest.raises(ValueError, match="backbone_adapter"):
        DecisionModel(base, tok, "cpu", lora=2, backbone_adapter="missing")
    with pytest.raises(ValueError, match="subclass"):
        DecisionModel(base, tok, "cpu", lora=2, backbone_adapter="builtins:object")
    with pytest.raises(ValueError, match="subclass"):
        DecisionModel(base, tok, "cpu", lora=2, backbone_adapter="builtins:breakpoint")
    model = DecisionModel(base, tok, "cpu", lora=2, lora_targets="attn")
    assert any("qkv_proj.lora_A" in name for name, _ in model.named_parameters())


def test_saved_token_ids_are_checked_before_loading_adapter_weights(tmp_path):
    base = tmp_path / "base"
    make_base(base, "llama")
    tokenizer = load_tokenizer(base)
    model = DecisionModel(base, tokenizer, "cpu", lora=2, head_dim=8)
    path = tmp_path / "checkpoint"
    model.lm.save_pretrained(path, save_embedding_layers=False)
    tokenizer.save_pretrained(path)
    meta = Meta(base=str(base), head=model.head.state_dict(), lora=2, head_dim=8,
                special_embeddings=True, tokenizer_saved=True, branch_mode=model.branch_mode)
    write_meta(path, meta)
    config_path = path / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["trainable_token_indices"]["embed_tokens"].reverse()
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="token IDs differ"):
        Checkpoint(path).load("cpu")
    (path / "tokenizer_config.json").unlink()
    with pytest.raises(ValueError, match="missing saved tokenizer"):
        Checkpoint(path).load("cpu")


def test_custom_adapter_is_persisted_and_used_by_training(tmp_path, monkeypatch):
    from jevany.train import main
    module = tmp_path / "custom_adapter.py"
    module.write_text(
        "from jevany.backbones import BackboneAdapter\n"
        "class CausalRows(BackboneAdapter):\n"
        "    def supports_packed(self, config):\n"
        "        return False\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    base = tmp_path / "base"
    make_base(base, "llama")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"state": "state", "questions": {
        "q": {"type": "choice", "instructions": "choose", "criteria": {"a": None, "b": None}, "label": "a"},
    }}) + "\n")
    args = ["--base", str(base), "--data", str(data), "--device", "cpu", "--lora", "2",
            "--head-dim", "8", "--backbone-adapter", "custom_adapter:CausalRows",
            "--lora-target-modules", "q_proj,v_proj", "--epochs", "2", "--accum", "1", "--max-steps", "2",
            "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0"]
    first = main(args + ["--out", str(tmp_path / "sft")])
    final = main(args + ["--out", str(tmp_path / "rlcr"), "--init-from", str(first), "--rlcr"])
    checkpoint = Checkpoint(final)
    assert checkpoint.meta.backbone_adapter == "custom_adapter:CausalRows"
    assert checkpoint.meta.branch_mode == "rows"
    tokenizer, model = checkpoint.load("cpu")
    assert type(model.adapter).__name__ == "CausalRows"
    assert torch.isfinite(model.probs(model.encode(tokenizer, RECORD))[0]).all()


class _LinearDecision(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(float(rank + 1)))

    def forward_batch(self, value):
        return self.weight * value


def _ddp_worker(rank, rendezvous):
    import torch.distributed as dist
    from jevany.train import distributed_model

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        model = _LinearDecision(rank)
        wrapped = distributed_model(model, None)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        wrapped(torch.tensor(float(1 + 2 * rank))).backward()
        optimizer.step()
        # Rank-zero initialization (1), then the average gradient (1 + 3) / 2.
        torch.testing.assert_close(model.weight, torch.tensor(0.8))
    finally:
        dist.destroy_process_group()


def test_ddp_initialization_and_gradient_average(tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


@pytest.mark.parametrize("override", [None, "1"])
def test_launcher_sets_communicator_default_and_keeps_explicit_override(tmp_path, override):
    import os
    from pathlib import Path
    import subprocess
    import sys

    python = tmp_path / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    print(2)\n"
        "else:\n"
        "    print(json.dumps({'nonblocking': os.environ.get('TORCH_NCCL_USE_COMM_NONBLOCKING'), 'args': sys.argv[1:]}))\n"
    )
    python.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(python), NNODES="1", NODE_RANK="0", PROCESSES_PER_HOST="2")
    env.pop("TORCH_NCCL_USE_COMM_NONBLOCKING", None)
    if override is not None:
        env["TORCH_NCCL_USE_COMM_NONBLOCKING"] = override
    launcher = Path(__file__).resolve().parents[1] / "infra/train.sh"
    result = subprocess.run(["bash", str(launcher), "--base", "example/base"], env=env,
                            capture_output=True, text=True, check=True)
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["nonblocking"] == (override or "0")
    assert payload["args"][-4:] == ["jevany", "train", "--base", "example/base"]
