# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Load labelled System One JSONL records and prepare training variants."""
import hashlib
import json
from pathlib import Path

from .api import SystemOneRequest, to_record

EVAL_ONLY = ("mmlu", "emotion", "tweet_offensive", "qnli", "paws", "sciq")
NONE_OPTIONS = [
    ("other", "None of the above"),
    ("none", "None of these"),
    ("not_listed", "Not listed here"),
    ("unknown", "Cannot be determined from the options given"),
    ("no_match", "No option matches"),
]
DISTRACTORS = {
    "weather": "Bad weather caused it",
    "purple": "The colour purple",
    "pancakes": "A recipe for pancakes",
    "taxes": "Unrelated: quarterly tax filing",
}


def source_seed(seed, source):
    return int.from_bytes(hashlib.sha256(f"{seed}:{source}".encode()).digest()[:8], "big")


def augment(request, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Permute choice options and optionally add a none option or distractor."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    result = {"state": request["state"], "questions": {}}
    for question_id, question in request["questions"].items():
        if question["type"] != "choice":
            result["questions"][question_id] = question
            continue
        criteria, label = dict(question["criteria"]), question["label"]
        if question.get("target") is not None:
            keys = list(criteria)
            rng.shuffle(keys)
            result["questions"][question_id] = {**question, "criteria": {key: criteria[key] for key in keys}}
            continue
        draw = rng.random()
        none_options = [(key, value) for key, value in NONE_OPTIONS if key not in criteria]
        distractors = [key for key in DISTRACTORS if key not in criteria]
        if len(criteria) > 2 and draw < p_none and none_options:
            key, description = rng.choice(none_options)
            criteria.pop(label)
            criteria[key], label = description, key
        elif draw < p_none + p_none_distract and len(criteria) < 255 and none_options:
            key, description = rng.choice(none_options)
            criteria[key] = description
        elif draw < p_none + p_none_distract + p_distract and len(criteria) < 255 and distractors:
            key = rng.choice(distractors)
            criteria[key] = DISTRACTORS[key]
        keys = list(criteria)
        rng.shuffle(keys)
        result["questions"][question_id] = {
            **question,
            "criteria": {key: criteria[key] for key in keys},
            "label": label,
        }
    return result


def none_pair(request, rng):
    """Return a minimal pair with the correct option present and removed."""
    eligible = [
        (question_id, question)
        for question_id, question in request["questions"].items()
        if question["type"] == "choice" and len(question["criteria"]) >= 3 and question.get("target") is None
    ]
    if not eligible:
        return []
    question_id, question = rng.choice(eligible)
    key, description = rng.choice([item for item in NONE_OPTIONS if item[0] not in question["criteria"]] or [("none_of_these", None)])
    keys = list(question["criteria"]) + [key]
    rng.shuffle(keys)
    present = {**question, "criteria": {item: description if item == key else question["criteria"][item] for item in keys}}
    absent = {**present, "criteria": {item: value for item, value in present["criteria"].items() if item != question["label"]}, "label": key}
    return [
        {"state": request["state"], "questions": {question_id: present}},
        {"state": request["state"], "questions": {question_id: absent}},
    ]


def load_records(path, source="custom"):
    """Load one labelled System One request per JSONL line."""
    records = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        if "state" not in record or not isinstance(record.get("questions"), dict) or not record["questions"]:
            raise ValueError(f"{path}:{index + 1}: a record needs a state and non-empty questions")
        for question_id, question in record["questions"].items():
            if "label" not in question:
                raise ValueError(f"{path}:{index + 1}: question {question_id!r} has no label")
            question.setdefault("src", f"{source}_{question['type']}")
        text = record["state"] if isinstance(record["state"], str) else json.dumps(record["state"], sort_keys=True, ensure_ascii=False)
        defaults = {
            "source": source,
            "variant": "clean",
            "id": f"{source}/{index}",
            "group_id": f"{source}/{index}",
            "row": index,
            "split": "custom",
            "text_sha256": hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest(),
        }
        record["_meta"] = {**defaults, **record.get("_meta", {})}
        records.append(record)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def api_request(record):
    """Remove labels, soft targets and metadata from a labelled request."""
    return {
        "state": record["state"],
        "questions": {
            question_id: {key: value for key, value in question.items() if key in ("type", "instructions", "criteria")}
            for question_id, question in record["questions"].items()
        },
    }


def materialize(request):
    """Convert a labelled request through the serving tokenizer format."""
    record, metadata = to_record(SystemOneRequest.model_validate(api_request(request)))
    for question, info, (question_id, source_question) in zip(record["questions"], metadata, request["questions"].items()):
        label = source_question["label"]
        question["label"] = info["keys"].index(label) if info["type"] == "choice" else int(label)
        question.update(src=source_question["src"], qtype=info["type"], qid=question_id, keys=info["keys"])
        if source_question.get("target") is not None:
            target = [float(source_question["target"].get(key, 0.0)) for key in question["keys"]]
            total = sum(target)
            if total <= 0:
                raise ValueError(f"target for {question_id} puts no mass on any option")
            question["target"] = [value / total for value in target]
    return record
