from scripts.build_v2_data import (
    content_hash,
    eval_split,
    helpsteer_record,
    hermes_records,
    text_choice_record,
)
from scripts.build_multimodal_eval import clean_options, distribution, question_parts
from scripts.eval_multimodal import ablate, verify_media
from scripts.audit_media_overlap import question_fingerprint


def test_helpsteer_preference_direction_and_soft_target():
    row = {
        "overall_preference": 2,
        "response1": "weak",
        "response2": "strong",
        "context": [{"role": "user", "content": "help"}],
        "domain": "general",
        "language": "en",
    }
    converted = helpsteer_record(row, 4, "train")
    question = converted["questions"]["preferred_response"]
    assert question["label"] == "response_b"
    assert question["target"] == {"response_a": 0.1, "response_b": 0.9}


def test_hermes_emits_each_agent_decision_and_keeps_one_group():
    row = {
        "id": "conversation",
        "category": "weather",
        "tools": '[{"type":"function","function":{"name":"forecast","description":"Get forecast"}}]',
        "conversations": [
            {"from": "human", "value": "Will it rain?"},
            {"from": "gpt", "value": '<tool_call>{"name":"forecast","arguments":{}}</tool_call>'},
            {"from": "tool", "value": "sunny"},
            {"from": "gpt", "value": "No."},
        ],
    }
    converted = hermes_records(row, "train", "multi")
    assert [item["questions"]["next_action"]["label"] for item in converted] == [
        "forecast", "respond_without_tool",
    ]
    assert len({item["_meta"]["group_id"] for item in converted}) == 1


def test_choice_converter_accepts_options_and_letter_answer():
    row = {"question": "Pick one", "options": "['zero', 'one']", "answer": "B"}
    converted = text_choice_record(row, "hard", 1, "train")
    assert converted["questions"]["answer"]["label"] == "1"


def test_eval_split_never_separates_groups():
    rows = []
    for group in range(4):
        for sibling in range(3):
            rows.append({"_meta": {"source": "video", "group_id": f"g{group}", "id": f"g{group}/{sibling}"}})
    calibration, development = eval_split(rows, 4, 7)
    calibration_groups = {row["_meta"]["group_id"] for row in calibration}
    development_groups = {row["_meta"]["group_id"] for row in development}
    assert calibration_groups.isdisjoint(development_groups)


def test_content_hash_ignores_labels_and_training_metadata():
    request = {
        "state": "same prompt",
        "questions": {"answer": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}}},
    }
    first = {**request, "questions": {"answer": {**request["questions"]["answer"], "label": "a", "src": "one"}}}
    second = {**request, "questions": {"answer": {**request["questions"]["answer"], "label": "b", "target": {"b": 1.0}}}}
    assert content_hash(first) == content_hash(second)


def test_mmstar_question_parser_separates_options():
    instructions, criteria = question_parts(
        "Which route is safe?\nOptions: A: Go left, then stop., B: Go right., "
        "C: Wait, then turn., D: None of these."
    )
    assert instructions == "Which route is safe?"
    assert criteria == {
        "a": "Go left, then stop.", "b": "Go right.",
        "c": "Wait, then turn.", "d": "None of these.",
    }
    instructions, criteria = question_parts(
        "Hint: return a letter.\nQuestion: Which number?\nChoices:\n"
        "(A) one\n(B) two\n(C) three"
    )
    assert instructions == "Which number?"
    assert criteria == {"a": "one", "b": "two", "c": "three"}


def test_media_ablation_blanks_and_deranges_within_source(tmp_path):
    blank = tmp_path / "blank.png"
    blank.write_bytes(b"x")
    rows = [
        {
            "state": "x",
            "media": [{"type": "image", "uri": f"{index}.png"}],
            "questions": {},
            "_meta": {
                "source": "same", "id": str(index), "group_id": f"group-{index}",
                "media_sha256": [f"hash-{index}"],
            },
        }
        for index in range(4)
    ]
    blanked = ablate(rows, "blank", blank_image=blank)
    assert {item["media"][0]["uri"] for item in blanked} == {str(blank.resolve())}
    shuffled = ablate(rows, "shuffled", seed=3)
    assert all(after["media"] != before["media"] for before, after in zip(rows, shuffled))
    assert sorted(item["media"][0]["uri"] for item in shuffled) == [f"{index}.png" for index in range(4)]


def test_media_ablation_keeps_sibling_questions_on_one_replacement():
    rows = [
        {
            "state": "x", "media": [{"type": "image", "uri": f"{group}.png"}], "questions": {},
            "_meta": {
                "source": "same", "id": f"{group}-{sibling}", "group_id": group,
                "media_sha256": [f"hash-{group}"],
            },
        }
        for group in ("one", "two", "three") for sibling in range(2)
    ]
    shuffled = ablate(rows, "shuffled", seed=3)
    by_group = {}
    for row in shuffled:
        by_group.setdefault(row["_meta"]["group_id"], set()).add(row["media"][0]["uri"])
        assert row["_meta"]["ablation_media_sha256"] != row["_meta"]["media_sha256"]
    assert all(len(replacements) == 1 for replacements in by_group.values())


def test_multimodal_eval_verifies_media_bytes(tmp_path):
    from jevany.suite import digest

    media = tmp_path / "image.png"
    media.write_bytes(b"original")
    rows = [{
        "media": [{"type": "image", "uri": str(media)}],
        "_meta": {"id": "one", "media_sha256": [digest(media)]},
    }]
    assert verify_media(rows) == 1
    media.write_bytes(b"changed")
    import pytest
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_media(rows)


def test_question_fingerprint_ignores_option_keys_and_order():
    first = {
        "type": "choice", "instructions": " Which route is safe? ",
        "criteria": {"a": "North", "b": "South"},
    }
    renamed = {
        "type": "choice", "instructions": "which ROUTE is safe?",
        "criteria": {"1": " south ", "0": "north"},
    }
    changed = {**renamed, "criteria": {"0": "north", "1": "east"}}
    assert question_fingerprint(first) == question_fingerprint(renamed)
    assert question_fingerprint(first) != question_fingerprint(changed)


def test_label_distribution_rejects_degenerate_tasks():
    def row(label):
        return {
            "questions": {"answer": {
                "src": "task", "criteria": {"a": "A", "b": "B"}, "label": label,
            }}
        }

    balanced = distribution([row("a"), row("b")] * 10)
    assert balanced["task"]["majority_position_baseline"] == 0.5
    import pytest
    with pytest.raises(ValueError, match="degenerate"):
        distribution([row("a")] * 20)


def test_multimodal_options_drop_placeholders_and_reject_ambiguity():
    cleaned, reason = clean_options({"a": "north", "b": "nan", "c": "south"}, "c")
    assert cleaned == {"a": "north", "c": "south"} and reason is None
    assert clean_options({"a": "north", "b": "nan"}, "b")[1] == "gold_option_is_placeholder"
    assert clean_options({"a": "North", "b": " north "}, "a")[1] == "duplicate_option_descriptions"
