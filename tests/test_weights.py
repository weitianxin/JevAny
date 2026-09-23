# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Optional checkpoint smoke test. Set JEVANY_TEST_CHECKPOINT to a released adapter directory."""
import os
import gc

import pytest


def test_checkpoint_loads_and_scores_one_request():
    run = os.environ.get("JEVANY_TEST_CHECKPOINT")
    if not run:
        pytest.skip("set JEVANY_TEST_CHECKPOINT to run the weight-backed smoke test")

    from jevany.api import SystemOneRequest, to_record
    from jevany.checkpoint import LoadOptions, load

    tok, model = load(run, "cuda", LoadOptions(dtype=None, merge=False))
    record, _ = to_record(SystemOneRequest.model_validate({
        "state": "The order was charged twice.",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {"billing": "Payment problems", "shipping": "Delivery problems"},
            }
        },
    }))
    probabilities = model.probs(model.encode(tok, record))
    assert len(probabilities) == 1
    assert probabilities[0].shape == (2,)
    assert abs(float(probabilities[0].sum()) - 1.0) < 1e-5


def test_checkpoint_load_matches_training_reconstruction():
    run = os.environ.get("JEVANY_TEST_CHECKPOINT")
    if not run:
        pytest.skip("set JEVANY_TEST_CHECKPOINT to run the weight-backed parity test")

    import torch
    from jevany.api import SystemOneRequest, to_record
    from jevany.checkpoint import Checkpoint, LoadOptions
    from jevany.model import DecisionModel, load_preprocessor

    checkpoint = Checkpoint(run)
    base_load_path = os.environ.get("JEVANY_BASE_LOAD_PATH")
    options = LoadOptions(dtype=None, merge=False, temperature=1.0, base_load_path=base_load_path)
    tokenizer, loaded = checkpoint.load("cuda", options)
    adapter_dtypes = {parameter.dtype for name, parameter in loaded.lm.named_parameters()
                      if "lora_" in name}
    assert adapter_dtypes == {torch.float32}
    record, _ = to_record(SystemOneRequest.model_validate({
        "state": "The order was charged twice.",
        "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                                 "criteria": {"billing": "Payment problems", "shipping": "Delivery problems"}}},
    }))
    loaded_logits = loaded.forward(loaded.encode(tokenizer, record))[0].float().cpu()
    del loaded
    gc.collect()
    torch.cuda.empty_cache()

    source = base_load_path or checkpoint.meta.base
    revision = None if base_load_path else checkpoint.meta.base_revision
    tokenizer = load_preprocessor(source, revision=revision, multimodal=checkpoint.meta.multimodal)
    trained = DecisionModel(
        source, tokenizer, "cuda", lora=checkpoint.meta.lora, revision=revision,
        head_dim=checkpoint.meta.head_dim, option_isolation=checkpoint.meta.option_isolation,
        special_embeddings=checkpoint.meta.special_embeddings,
        lora_targets=checkpoint.meta.extra.get("args", {}).get("lora_targets", "all"),
        dtype=torch.bfloat16 if checkpoint.meta.weights_dtype == "bf16" else torch.float32,
        multimodal=checkpoint.meta.multimodal,
    )
    checkpoint.warm_start(trained, checkpoint.meta)
    trained.eval()
    trained.head.temperature = 1.0
    trained_logits = trained.forward(trained.encode(tokenizer, record))[0].float().cpu()
    assert torch.equal(loaded_logits, trained_logits)
