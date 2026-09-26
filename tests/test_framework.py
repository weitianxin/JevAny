"""Framework contracts and a real, offline tiny-model train/save/serve cycle."""
import json
import pytest

from jevany import Choice, JevClient, Noul, Score, SystemOneRequest
from jevany.api import to_answers, to_record, validate_response
from jevany.data import load_records, validate_dataset
from jevany.datasets import init_starter


def test_starter_is_valid_and_keeps_development_separate(tmp_path):
    root = init_starter(tmp_path / "starter")
    assert validate_dataset(root / "train.jsonl")["questions"] == 72
    assert validate_dataset(root / "development.jsonl")["records"] == 8
    from jevany.suite import semantic_hash
    train = {semantic_hash(r) for r in load_records(root / "train.jsonl")}
    development = {semantic_hash(r) for r in load_records(root / "development.jsonl")}
    assert train.isdisjoint(development)
    with pytest.raises(FileExistsError):
        init_starter(root)


@pytest.mark.parametrize("question", [
    {"type": "score", "criteria": ["low", "high"], "label": 99},
    {"type": "score", "criteria": ["low", "high"], "label": True},
    {"type": "noul", "label": "false"},
    {"type": "noul", "label": 2},
    {"type": "choice", "criteria": {"a": None}, "label": "b"},
    {"type": "noul", "label": True, "target": {"true": -1, "false": 2}},
    {"type": "noul", "label": True, "target": {"true": float("nan")}},
    {"type": "noul", "label": True, "target": {"false": 0, "true": 0}},
    {"type": "noul", "label": True, "target": {"unknown": 1}},
    {"type": "noul", "label": True, "target": [0, 1]},
])
def test_bad_labels_fail_at_the_data_boundary(tmp_path, question):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"state": "x", "questions": {"q": question}}) + "\n")
    with pytest.raises(ValueError, match=r"bad.jsonl:1:"):
        load_records(source)


def test_client_wire_payload_and_validation(monkeypatch):
    import io
    import urllib.request

    calls = []

    def urlopen(request, timeout):
        body = json.loads(request.data)
        calls.append(body)
        validated = SystemOneRequest.model_validate(body)
        _, metadata = to_record(validated)
        return io.BytesIO(json.dumps({
            "model": "test-model", "answers": to_answers([[0.2, 0.8], [0.1, 0.9], [0.2, 0.8]], metadata),
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = JevClient(model="test-model")
    result = client.system_one(
        "state", {"team": Choice(criteria={"a": None, "b": None}), "urgent": Noul(),
                  "priority": Score(criteria=["low", "high"])},
    )
    assert calls[0]["questions"]["team"]["type"] == "choice"
    assert result["answers"]["team"]["choice"] == "b"
    assert result["answers"]["urgent"]["noul"] == 0.9
    assert result["answers"]["priority"]["score"] == 0.8
    with pytest.raises(ValueError):
        client({"state": "x", "questions": {}})
    assert len(calls) == 1


def test_score_legend_accepts_hosted_rendering_but_requires_all_levels():
    request = SystemOneRequest(state="state", questions={"q": Score(criteria=[{"level": "low"}, {"level": "high"}])})
    response = {"answers": {"q": {"type": "score", "score": 0.8, "confidence": 0.8,
                                "probabilities": {"0": 0.2, "1": 0.8},
                                "legend": {"0": "level: low", "1": "level: high"}}}}
    validate_response(request, response)
    del response["answers"]["q"]["legend"]["1"]
    with pytest.raises(ValueError, match="every level"):
        validate_response(request, response)


def test_training_config_overrides_and_rejects_mistakes(tmp_path, capsys):
    from jevany.train import parse_args, main
    data = init_starter(tmp_path / "data") / "train.jsonl"
    config = tmp_path / "sft.toml"
    config.write_text(f'data = "{data}"\nout = "{tmp_path / "run"}"\nlr = 0.01\n')
    parsed = parse_args(["--config", str(config), "--lr", "0.02", "--head-dim", "8"])
    assert parsed.lr == 0.02 and parsed.head_dim == 8
    main(["--config", str(config), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["data"]["records"] == 24
    assert not (tmp_path / "run").exists()
    for invalid in ('unknown = 1', 'batch = true', 'device = "potato"', 'rlcr = "yes"'):
        config.write_text(invalid)
        with pytest.raises(SystemExit):
            parse_args(["--config", str(config)])


def test_small_training_runs_have_nonempty_ranks_and_finite_schedules():
    import math
    import torch
    from jevany.train import distributed_slice, learning_rate_schedule

    shards = [distributed_slice([1, 2, 3], rank, 8)[0] for rank in range(8)]
    assert shards == [[1], [2], [3], [1], [2], [3], [1], [2]]
    for steps in (1, 2, 3, 10, 20):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter])
        scheduler = learning_rate_schedule(optimizer, [0.001], steps)
        for _ in range(steps):
            optimizer.step()
            scheduler.step()
            assert math.isfinite(scheduler.get_last_lr()[0])


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    from jevany.model import SPECIAL
    from jevany.train import main

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("framework")
    base = root / "base"
    vocabulary = {token: index for index, token in enumerate(
        ["[UNK]", "[PAD]", "[EOS]"] + SPECIAL + ["yes", "no", "choose", "a", "b", "state"])}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                  pad_token="[PAD]", eos_token="[EOS]", additional_special_tokens=SPECIAL)
    fast.save_pretrained(base)
    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, pad_token_id=1, eos_token_id=2,
    ))
    model.save_pretrained(base)
    data = init_starter(root / "data") / "train.jsonl"
    args = ["--base", str(base), "--data", str(data), "--device", "cpu", "--lora", "2",
            "--head-dim", "8", "--lora-targets", "qv", "--max-steps", "2", "--accum", "1",
            "--p-none", "0", "--p-none-distract", "0", "--p-distract", "0"]
    sft = main(args + ["--out", str(root / "sft")])
    rlcr = main(args + ["--out", str(root / "rlcr"), "--init-from", str(sft), "--rlcr"])
    yield root, sft, rlcr
    torch.set_num_threads(old_threads)


def test_training_and_local_http_parity(tiny_run, monkeypatch):
    from fastapi.testclient import TestClient
    from jevany import JevModel
    from jevany import serve

    root, sft, rlcr = tiny_run
    for checkpoint in (sft, rlcr):
        metrics = json.loads((checkpoint / "training_metrics.json").read_text())
        assert metrics["optimizer_steps"] == 2
        assert metrics["records_seen"] == 2
    inferred = JevModel.from_pretrained(sft, device="cpu")
    assert inferred.model_id == "sft"
    local = JevModel.from_pretrained(rlcr, device="cpu", model_name="my-jev")
    request = SystemOneRequest(
        state="state", model="jevany-latest",
        questions={"choice": Choice(criteria={"a": None, "b": None}),
                   "noul": Noul(), "score": Score(criteria=["low", "high"])},
    )
    expected = local(request)
    runtime = local.runtime
    server = serve.Server(runtime.checkpoint, runtime.tok, runtime.model, runtime.device, runtime.model_id)
    monkeypatch.setattr(serve.app.state, "server", server, raising=False)
    with TestClient(serve.app) as http:
        actual = http.post("/v1/systemone", json=request.model_dump()).json()
        assert actual["model"] == "my-jev"
        assert actual["answers"] == expected["answers"]
        assert actual["usage"] == expected["usage"]
        assert http.get("/v1/models").json()["models"][0]["id"] == "my-jev"
        assert http.get("/health").json() == {"status": "ready"}
        assert http.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 422
        too_long = {**request.model_dump(), "state": "state " * 9000}
        assert http.post("/v1/systemone", json=too_long).status_code == 422
        from typesafe_sdk import TypeSafeClient, Choice as OfficialChoice, Noul as OfficialNoul, Score as OfficialScore
        with TypeSafeClient(api_key="local", model="jevany-latest", base_url="http://testserver",
                            http_client=http) as official:
            result = official.system_one("state", {
                "choice": OfficialChoice(criteria={"a": None, "b": None}),
                "noul": OfficialNoul(),
                "score": OfficialScore(criteria=[{"level": "low"}, {"level": "high"}]),
            })
            assert result.choices["choice"].choice in ("a", "b")
            assert 0 <= result.nouls["noul"].noul <= 1
            assert result.scores["score"].legend == {0: {"level": "low"}, 1: {"level": "high"}}


def test_checkpoint_missing_path_fails_without_download(tmp_path):
    from jevany.checkpoint import Checkpoint
    with pytest.raises(ValueError, match="checkpoint does not exist"):
        Checkpoint(tmp_path / "missing")
    with pytest.raises(ValueError, match="missing head.pt"):
        Checkpoint(tmp_path)


def test_examples_execute_and_report_wrong_choices():
    from examples import sql_repair, service_recovery, inbox

    class FixedClient:
        def system_one(self, state, questions):
            request = SystemOneRequest(state=state, questions=questions)
            _, metadata = to_record(request)
            probabilities = [[float(key == ("exists" if self.correct else "join")) for key in m["keys"]]
                             if m["id"] == "repair" else [1 / len(m["keys"])] * len(m["keys"]) for m in metadata]
            return {"answers": to_answers(probabilities, metadata)}

    fake = FixedClient()
    fake.correct = True
    assert sql_repair.run(fake)["passed"]
    fake.correct = False
    assert not sql_repair.run(fake)["passed"]
    assert len(inbox.run(fake)) == 3
    environment = service_recovery.Recovery()
    environment.reset()
    for action in ("select_a", "warm_a", "canary", "promote", "probe"):
        _, _, done, info = environment.step(action)
    assert done and info["success"]
    environment.reset()
    for action in ("select_a", "canary", "promote", "probe"):
        _, _, done, info = environment.step(action)
    assert done and not info["success"]
    with pytest.raises(ValueError, match="unavailable"):
        environment.step("probe")
