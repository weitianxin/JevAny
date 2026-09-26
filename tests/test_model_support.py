"""Keep maintained model claims tied to training and serving evidence."""
import json
import math
from pathlib import Path

import pytest

from jevany.backbones import get_backbone_adapter


ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "docs/supported-models.json").read_text())["models"]
VALIDATION = json.loads((ROOT / "results/model-support-v1.json").read_text())
EVIDENCE = {model["id"]: model for model in VALIDATION["models"]}


@pytest.mark.parametrize("model", CATALOG, ids=lambda model: model["id"])
def test_catalog_training_reload_and_serving_evidence(model, tmp_path):
    evidence = EVIDENCE[model["id"]]
    for field in ("revision", "model_type", "media"):
        assert evidence[field] == model[field]
    inputs = model["media"] or ["text"]
    required = {(kind, objective) for kind in inputs for objective in ("sft", "rlcr")}
    checks = {(check["input"], check["objective"]): check for check in evidence["checks"]}
    assert required <= checks.keys()
    for key in required:
        check = checks[key]
        assert check["passed"]
        assert check["lora_updated"] and check["adapter_tensors_finite"]
        assert check["checkpoint_probability_max_delta"] <= VALIDATION["protocol"]["reload_max_probability_delta_limit"]
        assert all(math.isfinite(loss["nll"]) for loss in check["losses"])
        if check["objective"] == "sft":
            assert check["losses"][-1]["nll"] < check["losses"][0]["nll"]
        if check["input"] != "text":
            assert check["mixed_text"]
            assert check["media_probability_max_delta"] > 0
        serving = check["serving"]
        for field in ("passed", "python_http_answers_equal", "repeated_requests",
                      "invalid_requests_rejected", "health", "model_description"):
            assert serving[field], (model["id"], key, field)

    # Selection depends on config, including snapshots with arbitrary directory names.
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model["model_type"]}))
    adapter = get_backbone_adapter(multimodal=bool(model["media"]), source=str(tmp_path))
    assert adapter.media_types == frozenset(model["media"])


def test_catalog_has_unique_ids_and_only_current_qwen_models():
    assert len({model["id"] for model in CATALOG}) == len(CATALOG)
    assert {model["id"] for model in CATALOG if model["family"] == "Qwen"} == {
        "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", "Qwen/Qwen3.5-35B-A3B",
    }
