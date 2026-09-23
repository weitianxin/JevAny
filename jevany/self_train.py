"""Ground-truth-free pseudo labels for test-time SFT and RLCR."""
import copy
import math
import random
from collections import Counter

from .api import question_keys


def blind_record(record):
    """Replace every label before inference so pseudo labelling cannot consume ground truth."""
    blinded = copy.deepcopy(record)
    for question in blinded["questions"].values():
        keys = question_keys(question["type"], question.get("criteria"))
        question["label"] = False if question["type"] == "noul" else 0 if question["type"] == "score" else keys[0]
        question.pop("target", None)
    return blinded


def permute_choices(record, rng):
    permuted = copy.deepcopy(record)
    for question in permuted["questions"].values():
        if question["type"] != "choice":
            continue
        items = list(question["criteria"].items())
        rng.shuffle(items)
        question["criteria"] = dict(items)
        question["label"] = items[0][0]
    return permuted


def sample_answer(probabilities, rng, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    keys = list(probabilities)
    if not keys:
        raise ValueError("probabilities cannot be empty")
    values = [float(probabilities[key]) for key in keys]
    if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("probabilities must be finite, nonnegative, and have positive mass")
    weights = [max(value, 1e-12) ** (1 / temperature) for value in values]
    return rng.choices(keys, weights=weights, k=1)[0]


def pseudo_label(record, predictions, *, samples, min_agreement=0.5):
    """Create hard labels from a strict majority of sampled answers.

    Vote distributions are retained as provenance only. Training must use the
    winning label rather than distilling the model back into its own logits.
    """
    if len(predictions) != samples or samples < 1:
        raise ValueError("predictions must contain exactly samples entries")
    if not 0.5 <= min_agreement <= 1:
        raise ValueError("min_agreement must be in [0.5, 1]")
    output = copy.deepcopy(record)
    agreements, vote_distributions = [], {}
    for question_id, question in output["questions"].items():
        keys = question_keys(question["type"], question.get("criteria"))
        try:
            answers = [prediction[question_id] for prediction in predictions]
        except KeyError as error:
            raise ValueError(f"missing prediction for question {question_id!r}") from error
        unknown = set(answers) - set(keys)
        if unknown:
            raise ValueError(f"unknown predictions for question {question_id!r}: {sorted(unknown)}")
        votes = Counter(answers)
        winner, winning_votes = votes.most_common(1)[0]
        if winning_votes * 2 <= samples:
            return None
        agreement = winning_votes / samples
        if agreement < min_agreement:
            return None
        question["label"] = (winner == "true") if question["type"] == "noul" else int(winner) if question["type"] == "score" else winner
        question.pop("target", None)
        agreements.append(agreement)
        vote_distributions[question_id] = {key: votes[key] / samples for key in keys}
    output["_meta"] = {
        **output.get("_meta", {}),
        "pseudo_label": True,
        "pseudo_samples": samples,
        "pseudo_agreement": sum(agreements) / len(agreements),
        "pseudo_votes": vote_distributions,
    }
    return output
