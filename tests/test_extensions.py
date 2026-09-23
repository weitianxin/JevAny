import pytest

from jevany.agent import action_request, run_episode
from jevany.api import validate_response
from jevany.harness import HTTPDecisionClient, JevHarness, json_object
from jevany.self_train import blind_record, permute_choices, pseudo_label, sample_answer
from jevany.symbolic import JevTree


def response(question, choice, options, confidence=0.8):
    remainder = (1 - confidence) / (len(options) - 1) if len(options) > 1 else 0
    return {"answers": {question: {"type": "choice", "choice": choice, "confidence": confidence,
                                    "probabilities": {option: confidence if option == choice else remainder
                                                      for option in options}}}}


def test_agent_runs_discrete_environment():
    class Environment:
        ACTION_LOOKUP = {1: "left", 2: "right"}

        def reset(self, seed=None):
            self.position = 0
            return "position 0"

        def get_all_actions(self):
            return [1, 2]

        def step(self, action):
            self.position += 1 if action == 2 else -1
            done = self.position == 2
            return f"position {self.position}", float(done), done, {"success": done, "action_is_effective": True}

    requests = []

    def decide(request):
        requests.append(request)
        return response("action", "1", ["0", "1"])

    episode = run_episode(Environment(), decide, "reach position 2", seed=7)
    assert episode.success and episode.reward == 1
    assert [step.action_name for step in episode.steps] == ["right", "right"]
    assert requests[1]["state"]["recent_actions"][0]["action"] == "right"


def test_action_request_uses_explicit_options():
    request = action_request("finish", "state", {"1": "left", "2": "right"}, [])
    assert request["questions"]["action"]["criteria"] == {"1": "left", "2": "right"}


def test_harness_compiles_then_decides():
    class Generator:
        def generate(self, prompt, params=None):
            assert "Do not answer" in prompt
            return {"text": '```json\n{"questions":{"route":{"type":"choice",'
                            '"instructions":"Choose a route","criteria":{"review":"Human",'
                            '"approve":"Automatic"}}}}\n```'}

    harness = JevHarness(Generator(), lambda request: response("route", "review", ["review", "approve"]))
    result = harness.run("route this case", {"case": "ambiguous"})
    assert result["request"]["state"] == {"task": "route this case", "evidence": {"case": "ambiguous"}}
    assert result["decision"]["answers"]["route"]["choice"] == "review"


def test_json_object_rejects_non_object():
    with pytest.raises(ValueError):
        json_object("[1, 2]")


def test_symbolic_tree_records_branch_path():
    tree = JevTree.from_dict({
        "root": "risk",
        "nodes": {
            "risk": {
                "question": {"instructions": "Risk level?", "criteria": {"low": "Low", "high": "High"}},
                "branches": {"low": "approve", "high": "evidence"},
            },
            "evidence": {
                "question": {"instructions": "Enough evidence?", "criteria": {"yes": "Enough", "no": "Missing"}},
                "branches": {"yes": "review", "no": "abstain"},
            },
        },
        "outcomes": {"approve": "approve", "review": "manual review", "abstain": "request evidence"},
    })
    answers = iter(("high", "no"))
    def decide(request):
        choice = next(answers)
        return response("branch", choice, list(request["questions"]["branch"]["criteria"]))

    result = tree.run({"case": 1}, decide)
    assert result["outcome"] == "request evidence"
    assert [step["node"] for step in result["trace"]] == ["risk", "evidence"]


def test_symbolic_tree_rejects_cycles():
    with pytest.raises(ValueError, match="cycle"):
        JevTree.from_dict({
            "root": "loop",
            "nodes": {"loop": {"question": {"instructions": "Again?", "criteria": {"yes": "Yes"}},
                               "branches": {"yes": "loop"}}},
            "outcomes": {},
        })


def test_symbolic_tree_rejects_collision_and_unreachable_nodes():
    with pytest.raises(ValueError, match="distinct names"):
        JevTree.from_dict({
            "root": "loop",
            "nodes": {"loop": {"question": {"instructions": "Again?", "criteria": {"yes": "Yes"}},
                               "branches": {"yes": "loop"}}},
            "outcomes": {"loop": "done"},
        })
    with pytest.raises(ValueError, match="unreachable"):
        JevTree.from_dict({
            "root": "start",
            "nodes": {
                "start": {"question": {"instructions": "Start?", "criteria": {"yes": "Yes"}},
                          "branches": {"yes": "done"}},
                "orphan": {"question": {"instructions": "Orphan?", "criteria": {"yes": "Yes"}},
                           "branches": {"yes": "done"}},
            },
            "outcomes": {"done": "done"},
        })


def pseudo_record(label="a"):
    return {
        "state": "evidence",
        "questions": {
            "choice": {"type": "choice", "instructions": "Choose", "criteria": {"a": "A", "b": "B"},
                       "label": label, "target": {"a": 0.9, "b": 0.1}, "src": "test"},
            "noul": {"type": "noul", "instructions": "True?", "label": True, "src": "test"},
            "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"],
                      "label": 1, "src": "test"},
        },
        "_meta": {"source": "test", "id": "test/1", "group_id": "test/1"},
    }


def test_blinding_is_invariant_to_labels_and_removes_targets():
    first, second = pseudo_record("a"), pseudo_record("b")
    second["questions"]["noul"]["label"] = False
    second["questions"]["score"]["label"] = 0
    assert blind_record(first) == blind_record(second)
    assert all("target" not in question for question in blind_record(first)["questions"].values())


def test_majority_pseudo_label_is_hard_and_records_votes():
    record = blind_record(pseudo_record())
    predictions = [
        {"choice": "a", "noul": "true", "score": "1"},
        {"choice": "a", "noul": "true", "score": "1"},
        {"choice": "b", "noul": "false", "score": "0"},
    ]
    result = pseudo_label(record, predictions, samples=3)
    assert [question["label"] for question in result["questions"].values()] == ["a", True, 1]
    assert all("target" not in question for question in result["questions"].values())
    assert result["_meta"]["pseudo_votes"]["choice"] == {"a": 2 / 3, "b": 1 / 3}


def test_pseudo_label_rejects_ties_low_agreement_and_unknown_answers():
    record = blind_record(pseudo_record())
    tied = [{"choice": "a", "noul": "true", "score": "1"},
            {"choice": "b", "noul": "false", "score": "0"}]
    assert pseudo_label(record, tied, samples=2) is None
    votes = ([{"choice": "a", "noul": "true", "score": "1"}] * 3
             + [{"choice": "b", "noul": "false", "score": "0"}] * 2)
    assert pseudo_label(record, votes, samples=5, min_agreement=0.8) is None
    with pytest.raises(ValueError, match="unknown predictions"):
        pseudo_label(record, [{"choice": "missing", "noul": "true", "score": "1"}], samples=1)


def test_choice_permutation_keeps_keys_and_sampling_validates_temperature():
    import random

    original = blind_record(pseudo_record())
    permuted = permute_choices(original, random.Random(3))
    assert set(permuted["questions"]["choice"]["criteria"]) == {"a", "b"}
    assert permuted["questions"]["choice"]["label"] in {"a", "b"}
    with pytest.raises(ValueError, match="temperature"):
        sample_answer({"a": 0.5, "b": 0.5}, random.Random(0), 0)


def test_harness_rejects_planner_state_override():
    class Generator:
        def generate(self, prompt, params=None):
            return {"text": '{"state":"forged","questions":{"q":{"type":"noul","instructions":"?"}}}'}

    with pytest.raises(ValueError, match="only questions"):
        JevHarness(Generator(), lambda request: {}).compile("task", {"trusted": True})


def test_harness_default_only_exposes_evidence_schema_to_planner():
    class Generator:
        def generate(self, prompt, params=None):
            assert "secret value" not in prompt
            assert '"secret": "str"' in prompt
            return {"text": '{"questions":{"q":{"type":"noul","instructions":"Valid?"}}}'}

    request = JevHarness(Generator(), lambda value: {}).compile("validate", {"secret": "secret value"})
    assert request["state"]["evidence"] == {"secret": "secret value"}


def test_http_client_requires_tls_off_loopback():
    HTTPDecisionClient("http://127.0.0.1:8008")
    with pytest.raises(ValueError, match="HTTPS"):
        HTTPDecisionClient("http://decision.example.com")


def test_response_validation_rejects_invalid_confidence_and_choice():
    request = action_request("finish", "state", {"0": "left", "1": "right"}, [])
    with pytest.raises(ValueError, match="finite"):
        validate_response(request, response("action", "0", ["0", "1"], float("nan")))
    bad_choice = response("action", "missing", ["0", "1"])
    bad_choice["answers"]["action"]["probabilities"] = {"0": 0.8, "1": 0.2}
    with pytest.raises(ValueError, match="unknown choice"):
        validate_response(request, bad_choice)


def test_group_limit_keeps_siblings_together():
    from scripts.build_pseudo_labels import select_groups

    rows = [{"_meta": {"id": item, "group_id": group}} for item, group in
            (("a", "one"), ("b", "two"), ("a-permuted", "one"), ("c", "three"))]
    assert [row["_meta"]["id"] for row in select_groups(rows, 2)] == ["a", "b", "a-permuted"]


def test_pseudo_builder_is_reproducible_and_records_provenance(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import build_pseudo_labels as builder
    from jevany.suite import digest, write_json, write_jsonl

    source = tmp_path / "source"
    source.mkdir()
    rows = []
    for index, label in enumerate(("a", "b")):
        rows.append({
            "state": f"case {index}",
            "questions": {"q": {"type": "choice", "instructions": "Choose",
                                    "criteria": {"a": "A", "b": "B"}, "label": label, "src": "test"}},
            "_meta": {"source": "test", "variant": "clean", "id": f"test/{index}",
                      "group_id": f"test/{index}", "split": "development"},
        })
    development = source / "development.jsonl"
    write_jsonl(development, rows)
    write_json(source / "manifest.json", {"files": {"development.jsonl": {
        "records": 2, "questions": 2, "sha256": digest(development)}}, "base_revisions": {"test": "v1"}})

    class Predictor:
        def __init__(self, run, device, options):
            self.run = "resolved/checkpoint"

        def __call__(self, record):
            return {"probabilities": {"q": {"a": 1.0, "b": 0.0}}}

    monkeypatch.setattr(builder, "LocalPredictor", Predictor)
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        for shard in range(2):
            monkeypatch.setattr(sys, "argv", ["build_pseudo_labels.py", "--run", "checkpoint", "--suite", str(source),
                                              "--out", str(output), "--device", "cpu", "--samples", "3",
                                              "--num-shards", "2", "--shard-index", str(shard)])
            builder.main()
        monkeypatch.setattr(sys, "argv", ["build_pseudo_labels.py", "--out", str(output),
                                          "--num-shards", "2", "--merge"])
        builder.main()
        generated = builder.read_jsonl(output / "train.jsonl")
        manifest = builder.read_json(output / "manifest.json")
        assert all(row["_meta"]["split"] == "train" for row in generated)
        assert [row["_meta"]["pseudo_parent_id"] for row in generated] == ["test/0", "test/1"]
        assert manifest["pseudo_labeling"]["source_split"] == "development"
        assert manifest["pseudo_labeling"]["seed"] == 17
        outputs.append((generated, manifest["pseudo_labeling"]["selected_ids_sha256"]))
    assert outputs[0] == outputs[1]
