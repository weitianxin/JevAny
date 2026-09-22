# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Optional checkpoint smoke test. Set JEVANY_TEST_CHECKPOINT to a released adapter directory."""
import os

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
