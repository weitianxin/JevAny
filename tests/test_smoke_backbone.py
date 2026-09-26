"""The public compatibility check includes training, reload and HTTP serving."""
import json

import pytest
import torch

from scripts.smoke_backbone import main, prepare_fixture
from test_backbones import make_base
from test_multimodal_backbones import make_vision_base


def test_mixed_fixture_evaluates_media_and_text_rows(tmp_path):
    fixture = tmp_path / "fixture"
    prepare_fixture(fixture, "image", mixed_text=True)
    training = (fixture / "train.jsonl").read_text().splitlines()
    for split in ("calibration", "development"):
        rows = [json.loads(line) for line in (fixture / f"{split}.jsonl").read_text().splitlines()]
        assert len(rows) == len(training) == 8
        assert sum(bool(row.get("media")) for row in rows) == 6
        assert sum(not row.get("media") for row in rows) == 2


@pytest.mark.parametrize("family", ["llama", "gemma4_unified"])
def test_sft_rlcr_and_serving_smoke(tmp_path, family):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(17)
        base = tmp_path / "base"
        if family == "llama":
            make_base(base, family)
        else:
            make_vision_base(base, family)
        args = ["--base", str(base), "--device", "cpu", "--lr", "0.001", "--head-lr", "0.001"]
        if family != "llama":
            args += ["--media", "image"]
        main(args + ["--steps", "12", "--out", str(tmp_path / "sft")])
        main(args + ["--steps", "2", "--out", str(tmp_path / "rlcr"), "--rlcr",
                     "--init-from", str(tmp_path / "sft/checkpoint")])
        for stage in ("sft", "rlcr"):
            report = json.loads((tmp_path / stage / "smoke.json").read_text())
            assert report["passed"]
            assert report["objective"] == stage
            assert report["lora_updated"]
            assert report["serving"]["passed"]
            assert report["serving"]["invalid_requests_rejected"]
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("extra", [["--steps", "0"], ["--rlcr"]])
def test_invalid_smoke_arguments(tmp_path, extra):
    with pytest.raises(SystemExit):
        main(["--base", "unused", "--out", str(tmp_path), *extra])
