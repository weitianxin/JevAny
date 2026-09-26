import math

import pytest

from scripts import benchmark_latency as latency


def record(identifier):
    return {"_meta": {"id": identifier}, "questions": {"answer": {}}}


def test_measure_excludes_warmup_and_times_full_predictor(monkeypatch):
    clock = [0.0]
    calls = []
    syncs = []
    monkeypatch.setattr(latency.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(latency, "sync", lambda device: syncs.append(device))

    def predictor(row):
        calls.append(row["_meta"]["id"])
        clock[0] += 0.007  # encoding plus forward; only 2 ms is the forward
        return {"latency_ms": 2.0, "input_tokens": 42}

    result = latency.measure([record("a"), record("b")], predictor, "cpu", warmup=3, repeats=2)
    assert len(calls) == 7
    assert len(syncs) == 9
    assert result["measured_requests"] == 4
    assert result["forward_ms"]["median"] == 2.0
    assert result["end_to_end_ms"]["median"] == pytest.approx(7.0)
    assert result["serial_requests_per_second"] == pytest.approx(1000 / 7)
    assert sorted((row["id"], row["repeat"]) for row in result["samples"]) == [
        ("a", 0), ("a", 1), ("b", 0), ("b", 1),
    ]
    assert result["peak_allocated_bytes"] is None


@pytest.mark.parametrize("records,warmup,repeats", [([], 1, 1), ([record("a")], 0, 1), ([record("a")], 1, 0)])
def test_invalid_panel_fails_before_inference(records, warmup, repeats):
    def forbidden(_):
        pytest.fail("invalid panel reached model")
    with pytest.raises(ValueError):
        latency.measure(records, forbidden, "cpu", warmup, repeats)


@pytest.mark.parametrize("values", [[], [0.0], [-1.0], [math.inf], [math.nan]])
def test_invalid_latency_is_not_reported(values):
    with pytest.raises(ValueError):
        latency.latency_summary(values)


def test_rejected_record_aborts_instead_of_changing_panel():
    def reject(_):
        raise ValueError("packed request exceeds frozen token limit")
    with pytest.raises(ValueError, match="exceeds"):
        latency.measure([record("too-long")], reject, "cpu", warmup=1)


def test_cli_rejects_empty_modality_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr(latency, "load_split", lambda *_: [record("text")])
    monkeypatch.setattr(latency, "LocalPredictor", lambda *_: pytest.fail("loaded model"))
    with pytest.raises(SystemExit):
        latency.main([
            "--run", "unused", "--suite", "unused", "--modality", "video",
            "--out", str(tmp_path / "result.json"),
        ])
    assert not (tmp_path / "result.json").exists()


def test_cli_preserves_existing_report(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("existing measurements")
    with pytest.raises(SystemExit):
        latency.main(["--run", "unused", "--suite", "unused", "--out", str(report)])
    assert report.read_text() == "existing measurements"
