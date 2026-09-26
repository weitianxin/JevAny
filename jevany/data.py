# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""Load labelled System One JSONL records and prepare training variants."""
import hashlib
import json
import math
from collections import Counter
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


def resolve_media(record, directory):
    """Resolve relative media URIs against the JSONL file that contains the record."""
    for media in record.get("media", []):
        uri = Path(media["uri"])
        if not uri.is_absolute() and "://" not in media["uri"]:
            media["uri"] = str((Path(directory) / uri).resolve())
    return record


def augment(request, rng, p_none=0.1, p_none_distract=0.12, p_distract=0.15):
    """Permute choice options and optionally add a none option or distractor."""
    if min(p_none, p_none_distract, p_distract) < 0 or p_none + p_none_distract + p_distract > 1:
        raise ValueError("augmentation probabilities must be nonnegative and sum to at most one")
    result = {"state": request["state"], "questions": {}}
    if request.get("media"):
        result["media"] = request["media"]
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
    common = {"state": request["state"]}
    if request.get("media"):
        common["media"] = request["media"]
    return [{**common, "questions": {question_id: present}}, {**common, "questions": {question_id: absent}}]


def load_records(path, source="custom"):
    """Load one labelled System One request per JSONL line."""
    records = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict) or "state" not in record or not isinstance(record.get("questions"), dict) or not record["questions"]:
                raise ValueError("a record needs a state and non-empty questions")
            for question_id, question in record["questions"].items():
                if not isinstance(question, dict) or "label" not in question:
                    raise ValueError(f"question {question_id!r} has no label")
                question.setdefault("src", f"{source}_{question.get('type', 'unknown')}")
            materialize(record)
            if not isinstance(record.get("_meta", {}), dict):
                raise ValueError("_meta must be an object")
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f"{path}:{index + 1}: {error}") from error
        resolve_media(record, Path(path).parent)
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
    request = {
        "state": record["state"],
        "questions": {
            question_id: {key: value for key, value in question.items() if key in ("type", "instructions", "criteria")}
            for question_id, question in record["questions"].items()
        },
    }
    if record.get("media"):
        request["media"] = record["media"]
    return request


def materialize(request):
    """Convert a labelled request through the serving tokenizer format."""
    record, metadata = to_record(SystemOneRequest.model_validate(api_request(request)))
    for question, info, (question_id, source_question) in zip(record["questions"], metadata, request["questions"].items()):
        label = source_question["label"]
        if info["type"] == "choice" and (not isinstance(label, str) or label not in info["keys"]):
            raise ValueError(f"label for {question_id!r} must be one of {info['keys']}")
        if info["type"] == "noul" and (type(label) not in (bool, int) or label not in (0, 1)):
            raise ValueError(f"label for {question_id!r} must be true or false (or 0/1)")
        if info["type"] == "score" and (type(label) is not int or not 0 <= label < len(info["keys"])):
            raise ValueError(f"label for {question_id!r} must be an integer in 0..{len(info['keys']) - 1}")
        question["label"] = info["keys"].index(label) if info["type"] == "choice" else int(label)
        question.update(src=source_question.get("src", f"custom_{info['type']}"),
                        qtype=info["type"], qid=question_id, keys=info["keys"])
        if source_question.get("target") is not None:
            raw_target = source_question["target"]
            if not isinstance(raw_target, dict) or set(raw_target) - set(info["keys"]):
                raise ValueError(f"target for {question_id!r} must use the question's option keys")
            try:
                target = [float(raw_target.get(key, 0.0)) for key in question["keys"]]
            except (TypeError, ValueError) as error:
                raise ValueError(f"target for {question_id!r} must contain numeric weights") from error
            if any(not math.isfinite(value) or value < 0 for value in target):
                raise ValueError(f"target for {question_id!r} must contain finite nonnegative weights")
            total = sum(target)
            if not math.isfinite(total) or total <= 0:
                raise ValueError(f"target for {question_id} puts no mass on any option")
            question["target"] = [value / total for value in target]
    return record


def validate_dataset(path: str | Path) -> dict:
    """Validate labelled JSONL without loading weights; return counts by type/source."""
    records = load_records(path)
    missing = [item["uri"] for record in records for item in record.get("media", [])
               if "://" not in item["uri"] and not Path(item["uri"]).is_file()]
    if missing:
        raise ValueError(f"{path}: missing media file: {missing[0]}")
    return {
        "records": len(records),
        "questions": sum(len(record["questions"]) for record in records),
        "types": dict(Counter(q["type"] for record in records for q in record["questions"].values())),
        "sources": dict(Counter(record["_meta"]["source"] for record in records)),
    }
