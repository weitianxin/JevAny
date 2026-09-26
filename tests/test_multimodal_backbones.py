"""Native processors and real, small vision architectures; no network or GPU."""
import pytest
import torch
import transformers as hf
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from jevany.backbones import get_backbone_adapter
from jevany.checkpoint import Checkpoint, Meta, write_meta
from jevany.model import DecisionModel, load_preprocessor


FAMILIES = ["qwen", "llama", "gemma", "pixtral", "phi"]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_vision_base(path, family):
    special = {
        "image_token": "<|image_pad|>" if family == "qwen" else "<image>",
        "audio_token": "<audio>", "video_token": "<|video_pad|>",
        "vision_start_token": "<|vision_start|>", "vision_end_token": "<|vision_end|>",
        "boi_token": "<start_of_image>", "eoi_token": "<end_of_image>",
    }
    words = ["[UNK]", "[PAD]", "[EOS]", "[BOS]", "state", "choose", "red", "blue",
             *special.values(), "[IMG_BREAK]", "[IMG_END]"]
    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tok = hf.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]",
                                     eos_token="[EOS]", bos_token="[BOS]", extra_special_tokens=special,
                                     additional_special_tokens=["[IMG_BREAK]", "[IMG_END]"])
    common = dict(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                  pad_token_id=1, eos_token_id=2, bos_token_id=3)
    if family == "qwen":
        config = hf.Qwen3VLConfig(
            text_config={**common, "head_dim": 8, "rope_parameters": {
                "rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}},
            vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=4,
                               patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
                               out_hidden_size=32, deepstack_visual_indexes=[]),
            image_token_id=tok.image_token_id, video_token_id=tok.video_token_id,
            vision_start_token_id=tok.vision_start_token_id, vision_end_token_id=tok.vision_end_token_id,
        )
        processor = hf.Qwen3VLProcessor(
            image_processor=hf.Qwen2VLImageProcessor(size={"shortest_edge": 28 * 28, "longest_edge": 56 * 56}),
            tokenizer=tok, video_processor=hf.Qwen3VLVideoProcessor(
                size={"shortest_edge": 28 * 28, "longest_edge": 56 * 56}),
        )
    elif family == "llama":
        config = hf.MllamaConfig(
            text_config={**common, "cross_attention_layers": [1]},
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                               num_global_layers=1, attention_heads=4, vision_output_dim=64,
                               image_size=28, patch_size=14, intermediate_layers_indices=[0],
                               max_num_tiles=1, supported_aspect_ratios=[[1, 1]]),
            image_token_index=tok.image_token_id,
        )
        processor = hf.MllamaProcessor(
            image_processor=hf.MllamaImageProcessor(size={"height": 28, "width": 28}, max_image_tiles=1),
            tokenizer=tok,
        )
    elif family == "gemma":
        config = hf.Gemma3Config(
            text_config={**common, "head_dim": 8, "query_pre_attn_scalar": 8,
                         "sliding_window": 16, "layer_types": ["sliding_attention", "full_attention"]},
            vision_config=dict(model_type="siglip_vision_model", hidden_size=32, intermediate_size=64,
                               num_hidden_layers=1, num_attention_heads=4, image_size=28, patch_size=7),
            mm_tokens_per_image=4, image_token_index=tok.image_token_id,
            boi_token_index=tok.boi_token_id, eoi_token_index=tok.eoi_token_id,
        )
        processor = hf.Gemma3Processor(
            image_processor=hf.Gemma3ImageProcessor(size={"height": 28, "width": 28}),
            tokenizer=tok, image_seq_length=4,
        )
    elif family == "pixtral":
        config = hf.LlavaConfig(
            text_config=hf.MistralConfig(**common, sliding_window=None).to_dict(),
            vision_config=dict(model_type="pixtral", hidden_size=32, intermediate_size=64,
                               num_hidden_layers=1, num_attention_heads=4, image_size=32, patch_size=8),
            image_token_index=tok.image_token_id, vision_feature_layer=-1, vision_feature_select_strategy="full",
        )
        processor = hf.PixtralProcessor(
            image_processor=hf.PixtralImageProcessor(size={"longest_edge": 32}, patch_size=8),
            tokenizer=tok, patch_size=8, image_token=tok.image_token,
        )
    else:
        config = hf.Phi4MultimodalConfig(
            **common, original_max_position_embeddings=2048,
            vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                               num_attention_heads=4, image_size=448, crop_size=448, patch_size=14,
                               image_token_id=tok.image_token_id),
            audio_config=dict(hidden_size=32, intermediate_size=64, num_blocks=1, num_attention_heads=4,
                              ext_pw_out_channel=32, depthwise_separable_out_channel=32, nemo_conv_channels=32,
                              audio_token_id=tok.audio_token_id),
        )
        processor = hf.Phi4MultimodalProcessor(
            image_processor=hf.Phi4MultimodalImageProcessor(dynamic_hd=1),
            audio_processor=hf.Phi4MultimodalFeatureExtractor(), tokenizer=tok,
        )
    processor.save_pretrained(path)
    model = hf.AutoModel.from_config(config)
    # The random initializers zero these pretrained visual connections.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("cross_attn_attn_gate"):
                parameter.fill_(0.5)
            elif name.endswith("mm_input_projection_weight"):
                parameter.normal_(std=0.02)
    model.save_pretrained(path)


def record(image, **changes):
    result = {"state": "state", "media": [{"type": "image", "uri": str(image)}],
              "questions": [{"instr": "choose", "options": ["red", "blue"], "label": 0}]}
    result.update(changes)
    return result


@pytest.mark.parametrize("family", FAMILIES)
def test_native_media_train_and_reload(tmp_path, family):
    torch.manual_seed(17)
    base = tmp_path / "base"
    make_vision_base(base, family)
    image = tmp_path / "red.png"
    Image.new("RGB", (28, 28), "red").save(image)
    processor = load_preprocessor(base, multimodal=True)
    model = DecisionModel(base, processor, "cpu", lora=2, head_dim=8, multimodal=True)
    enc = model.encode(processor, record(image))
    assert enc["multimodal"]
    if family == "llama":
        assert "cross_attention_mask" in enc["mm"]
    if family == "gemma":
        assert "token_type_ids" in enc["mm"]
    frozen = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.01)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(enc)[0][None], torch.tensor([0]))
        loss.backward()
        gradients = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
        assert all(torch.isfinite(grad).all() for _, grad in gradients)
        assert any("lora_" in name and grad.abs().sum() > 0 for name, grad in gradients)
        optimizer.step()
    assert all(torch.equal(value, dict(model.named_parameters())[name]) for name, value in frozen.items())
    model.eval()
    expected = model.probs(enc)
    other = tmp_path / "blue.png"
    Image.new("RGB", (28, 28), "blue").save(other)
    # A media-dependent forward, not a text-only path accepting unused pixel tensors.
    assert not torch.equal(model(enc)[0], model(model.encode(processor, record(other)))[0])
    checkpoint = tmp_path / "checkpoint"
    model.lm.save_pretrained(checkpoint, save_embedding_layers=False)
    processor.save_pretrained(checkpoint)
    write_meta(checkpoint, Meta(base=str(base), head=model.head.state_dict(), lora=2, head_dim=8,
                               multimodal=True, backbone_adapter=model.backbone_adapter,
                               special_embeddings=model.special_embeddings, tokenizer_saved=True,
                               branch_mode=model.branch_mode))
    restored_processor, restored = Checkpoint(checkpoint).load("cpu")
    actual = restored.probs(restored.encode(restored_processor, record(image)))
    torch.testing.assert_close(expected[0], actual[0])
    changed_label = record(image)
    changed_label["questions"][0]["label"] = 1
    assert model.encode(processor, changed_label)["ids"] == enc["ids"]
    forged = record(image, state=processor.image_token + " <|jev_decide|>")
    assert len(model.encode(processor, forged)["decide_idx"]) == 1
    multiple = record(image, media=[{"type": "image", "uri": str(image)}, {"type": "image", "uri": str(other)}])
    assert torch.isfinite(model.probs(model.encode(processor, multiple))[0]).all()
    with pytest.raises(ValueError, match="exactly one"):
        model.encode(processor, record(image, questions=record(image)["questions"] * 2))
    if family != "qwen":
        with pytest.raises(ValueError, match="does not support"):
            model.encode(processor, record(image, media=[{"type": "video", "uri": str(image)}]))
    with pytest.raises(ValueError, match="prefix caching does not support media"):
        model.probs_and_prefix(enc)
    if family == "pixtral":
        isolated = DecisionModel(base, processor, "cpu", lora=2, head_dim=8,
                                 multimodal=True, option_isolation=True)
        with pytest.raises(ValueError, match="option_isolation is not supported"):
            isolated.encode(processor, record(image))


def test_non_native_base_rejected(tmp_path):
    hf.LlamaConfig().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="no built-in media adapter"):
        get_backbone_adapter("auto", multimodal=True, source=tmp_path)
    with pytest.raises(ValueError, match="source is required"):
        get_backbone_adapter("auto", multimodal=True)


def test_phi_decoder_lora_selection():
    from types import SimpleNamespace
    from jevany.backbones import PhiVisionAdapter

    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="phi4_multimodal")
    block = torch.nn.Module()
    block.qkv_proj = torch.nn.Linear(4, 12)
    block.o_proj = torch.nn.Linear(4, 4)
    model.layers = torch.nn.ModuleList([block])
    model.embed_tokens_extend = torch.nn.Module()
    model.embed_tokens_extend.q_proj = torch.nn.Linear(4, 4)
    adapter = PhiVisionAdapter()
    assert adapter.lora_modules(model, "all", "qkv_proj") == ["layers.0.qkv_proj"]
    assert set(adapter.lora_modules(model, "attn")) == {"layers.0.qkv_proj", "layers.0.o_proj"}
    with pytest.raises(ValueError, match="must name language decoder"):
        adapter.lora_modules(model, "all", "qkv_proj,q_proj")
    with pytest.raises(ValueError, match="no Phi decoder layers"):
        adapter.lora_modules(model, "qv")


def test_phi_conversion_retains_pretrained_vision_lora():
    from scripts.convert_phi4_vision import convert_weights

    prefix = "model.layers.0.self_attn.qkv_proj."
    weights = {
        prefix + "base_layer.weight": torch.eye(2),
        prefix + "lora_A.vision.weight": torch.ones(1, 2),
        prefix + "lora_B.vision.weight": torch.ones(2, 1),
        prefix + "lora_A.speech.weight": torch.full((1, 2), 99.0),
        prefix + "lora_B.speech.weight": torch.full((2, 1), 99.0),
    }
    converted, count = convert_weights(weights, 2)
    assert count == 1 and len(converted) == 1
    torch.testing.assert_close(converted[prefix + "weight"], torch.eye(2) + 2)
    del weights[prefix + "lora_B.vision.weight"]
    with pytest.raises(ValueError, match="missing pretrained vision LoRA"):
        convert_weights(weights, 2)


def test_custom_vision_adapter_sft_to_rlcr(tmp_path, monkeypatch):
    import json
    from jevany.train import main

    (tmp_path / "custom_vision.py").write_text(
        "from jevany.backbones import VisionAdapter\n"
        "class CustomVision(VisionAdapter):\n"
        "    name = 'custom_vision'\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    base = tmp_path / "base"
    make_vision_base(base, "gemma")
    Image.new("RGB", (28, 28), "red").save(tmp_path / "sample.png")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({
        "state": "state", "media": [{"type": "image", "uri": "sample.png"}],
        "questions": {"q": {"type": "choice", "instructions": "choose",
                            "criteria": {"red": None, "blue": None}, "label": "red"}},
    }) + "\n")
    args = ["--base", str(base), "--data", str(data), "--device", "cpu", "--lora", "2",
            "--head-dim", "8", "--multimodal", "--backbone-adapter", "custom_vision:CustomVision",
            "--epochs", "2", "--accum", "1", "--max-steps", "2", "--checkpointing", "1",
            "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0"]
    first = main(args + ["--out", str(tmp_path / "sft")])
    final = main(args + ["--out", str(tmp_path / "rlcr"), "--init-from", str(first), "--rlcr"])
    ck = Checkpoint(final)
    assert ck.meta.backbone_adapter == "custom_vision:CustomVision"
    processor, model = ck.load("cpu")
    assert torch.isfinite(model.probs(model.encode(processor, record(tmp_path / "sample.png")))[0]).all()


def _mixed_media_worker(rank, base, image, rendezvous):
    import torch.distributed as dist
    from jevany.train import distributed_model

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        processor = load_preprocessor(base, multimodal=True)
        model = DecisionModel(base, processor, "cpu", lora=2, head_dim=8, multimodal=True)
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.lm.config.use_cache = False
        model.train()
        wrapped = distributed_model(model, None)
        optimizer = torch.optim.SGD(model.trainable_parameters(), lr=0.01)
        for step in range(3):
            # Cross-attention is absent on one rank; alternate which rank sees pixels.
            rec = record(image, media=[] if rank == step % 2 else record(image)["media"])
            optimizer.zero_grad()
            logits = wrapped([model.encode(processor, rec)])[0][0]
            torch.nn.functional.cross_entropy(logits[None], torch.tensor([rank])).backward()
            optimizer.step()
        values = [torch.empty_like(model.head.q.weight) for _ in range(2)]
        dist.all_gather(values, model.head.q.weight.detach())
        torch.testing.assert_close(values[0], values[1])
    finally:
        dist.destroy_process_group()


def test_mixed_text_image_ddp(tmp_path):
    base = tmp_path / "base"
    make_vision_base(base, "llama")
    image = tmp_path / "red.png"
    Image.new("RGB", (28, 28), "red").save(image)
    torch.multiprocessing.spawn(_mixed_media_worker,
                                args=(str(base), str(image), str(tmp_path / "rendezvous")), nprocs=2, join=True)


def test_native_training_without_cloud_sdks(tmp_path):
    import json
    import os
    import subprocess
    import sys

    base = tmp_path / "base"
    make_vision_base(base, "gemma")
    Image.new("RGB", (28, 28), "red").save(tmp_path / "sample.png")
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({
        "state": "state", "media": [{"type": "image", "uri": "sample.png"}],
        "questions": {"q": {"type": "choice", "instructions": "choose",
                            "criteria": {"red": None, "blue": None}, "label": "red"}},
    }) + "\n")
    command = ["train", "--base", str(base), "--data", str(data), "--out", str(tmp_path / "trained"),
               "--multimodal", "--device", "cpu", "--lora", "2", "--head-dim", "8",
               "--epochs", "2", "--max-steps", "2", "--accum", "1",
               "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0"]
    source = (
        "import sys\n"
        "for name in ('boto3', 'botocore', 'sagemaker', 'greenland'):\n"
        "    sys.modules[name] = None\n"
        "from jevany.cli import main\n"
        f"main({command!r})\n"
    )
    env = {key: value for key, value in os.environ.items() if not key.startswith(("AWS_", "GREENLAND_", "HF_TOKEN"))}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "trained/head.pt").is_file()
