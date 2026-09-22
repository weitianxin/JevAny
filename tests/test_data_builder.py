from scripts.build_v2_data import (
    content_hash,
    eval_split,
    helpsteer_record,
    hermes_records,
    text_choice_record,
)


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
