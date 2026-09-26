# Modified for JevAny by Tianxin Wei, 2026.
# Derived from Kev by Jared Palmer under Apache-2.0. See NOTICE.
"""TypeSafe-compatible request/response shapes (POST /v1/systemone) mapped onto the single pointer primitive.

Noul   -> 2 options [false, true];            answer = p(true)
Choice -> options 'name' or 'name: desc';      answer = argmax, probabilities by name, confidence
Score  -> options = ordered level descriptions; answer = expected level, legend, probabilities by index
"""
import json
import math
import re
from datetime import datetime
from typing import Annotated, Any, Literal, Union
from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255
MAX_QUESTIONS = 64


class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: JSONContent = None
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: JSONContent = None
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS: raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: JSONContent = None
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Annotated[Union[Noul, Choice, Score], Field(discriminator="type")]


class Media(BaseModel):
    type: Literal["image", "video"]
    uri: str = Field(min_length=1)


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str = Field(default="jevany-latest", min_length=1)
    media: list[Media] = Field(default_factory=list, max_length=32)
    questions: dict[str, Question] = Field(min_length=1, max_length=MAX_QUESTIONS)


def validate_distribution(raw, keys):
    """Return a normalized probability list and the reported sum."""
    if not isinstance(raw, dict) or set(raw) != set(keys):
        raise ValueError("probability keys do not match requested options")
    try:
        values = [float(raw[key]) for key in keys]
    except (TypeError, ValueError) as error:
        raise ValueError("probabilities must be numeric") from error
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("probabilities must be finite and in [0, 1]")
    total = sum(values)
    tolerance = max(1e-5, len(keys) * 0.005 + 1e-8)
    if total <= 0 or abs(total - 1) > tolerance:
        raise ValueError(f"invalid probability sum: {total}")
    return [value / total for value in values], total


def validate_response(request: SystemOneRequest | dict, response: dict) -> dict:
    """Validate the answer ids, types, choices, and finite probability fields."""
    request = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
    if not isinstance(response, dict) or not isinstance(response.get("answers"), dict):
        raise ValueError("decision response must contain an answers object")
    if set(response["answers"]) != set(request.questions):
        raise ValueError("decision response question ids do not match the request")

    def probability(value, name):
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be numeric") from error
        if not math.isfinite(number) or not 0 <= number <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return number

    for question_id, question in request.questions.items():
        answer = response["answers"][question_id]
        if not isinstance(answer, dict) or answer.get("type") != question.type:
            raise ValueError(f"invalid answer type for {question_id!r}")
        if question.type == "noul":
            probability(answer.get("noul"), f"noul probability for {question_id!r}")
            continue
        keys = list(question.criteria) if question.type == "choice" else [
            str(index) for index in range(len(question.criteria))]
        probabilities = answer.get("probabilities")
        values, _ = validate_distribution(probabilities, keys)
        probability(answer.get("confidence"), f"confidence for {question_id!r}")
        if question.type == "choice":
            choice = answer.get("choice")
            if choice not in question.criteria:
                raise ValueError(f"unknown choice for {question_id!r}")
            if values[keys.index(choice)] < max(values):
                raise ValueError(f"choice for {question_id!r} is not an argmax")
        if question.type == "score":
            try:
                score = float(answer.get("score"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"score for {question_id!r} must be numeric") from error
            if not math.isfinite(score) or not 0 <= score <= len(question.criteria) - 1:
                raise ValueError(f"invalid score for {question_id!r}")
            legend = answer.get("legend")
            if not isinstance(legend, dict) or set(legend) != set(keys):
                raise ValueError(f"score legend for {question_id!r} must cover every level")
    return response


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None: return ""
    if isinstance(v, (str, int, float, bool)): return str(v)
    if isinstance(v, list): return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}" for k, x in v.items())


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE = re.compile(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b")


def date_facts(text: str) -> str:
    """Deterministic date arithmetic for the model: every pair of absolute dates found in `text`, as one sentence each
    ("August 3, 2026 is 12 days after July 22, 2026."). Returns "" when fewer than two dates are found. Dates are listed
    in order of first appearance."""
    found = []
    for m in _DATE.finditer(text):
        raw = m.group(0)
        try: d = datetime.strptime(raw, "%B %d, %Y") if "," in raw else datetime.strptime(raw, "%Y-%m-%d")
        except ValueError: continue
        if raw not in [r for r, _ in found]: found.append((raw, d))
    facts = []
    for i in range(len(found)):
        for j in range(i + 1, len(found)):
            n = (found[j][1] - found[i][1]).days
            facts.append(f"{found[j][0]} is {abs(n)} day{'s' if abs(n) != 1 else ''} {'after' if n > 0 else 'before'} {found[i][0]}." if n else f"{found[j][0]} is the same day as {found[i][0]}.")
    return " ".join(facts)


def with_date_facts(state):
    """State with a `date_facts` field (object states) or an appended paragraph (string states) when two or more absolute
    dates appear. Opt-in preprocessing (JEVANY_DATE_FACTS=1 in jevany.serve, --date_facts in jevany.benchmark)."""
    facts = date_facts(render(state))
    if not facts: return state
    if isinstance(state, dict): return {**state, "date_facts": facts}
    if isinstance(state, list): return state + [{"date_facts": facts}]
    return f"{state}\n\ndate_facts: {facts}"


def question_keys(qtype: str, criteria) -> list[str]:
    """The keys a question's probabilities are reported under, in option order: the criteria names (choice),
    ["false", "true"] (noul), the level indices as strings (score). Labels, targets and anchors use the same keys."""
    if qtype == "choice": return list(criteria)
    if qtype == "noul": return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def to_record(req: SystemOneRequest):
    """-> internal record for encode(), plus per-question metadata ({"id", "type", "keys", "legend" for score}) to map
    probabilities back."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        m = {"id": qid, "type": q.type, "keys": question_keys(q.type, q.criteria)}
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
        else:
            opts = [render(x) for x in q.criteria]
            m["legend"] = dict(zip(m["keys"], q.criteria))
        qs.append({"instr": render(q.instructions), "options": opts, "label": 0}); meta.append(m)
    record = {"state": render(req.state), "questions": qs}
    if req.media:
        record["media"] = [item.model_dump() for item in req.media]
    return record, meta


def choice_confidence(p: list[float]) -> float:
    K = len(p)
    return 1.0 if K == 1 else (max(p) - 1 / K) / (1 - 1 / K)


def score_confidence(p: list[float]) -> float:
    """Approximation of TypeSafe's 'distance from the modal level' statistic (exact formula unpublished):
    1 - E|level - mode| / (L - 1)."""
    L = len(p); mode = max(range(L), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (L - 1)


def r2(x: float) -> float:
    return round(float(x), 2)


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": r2(p[1])}
        elif m["type"] == "choice":
            dist = {k: float(v) for k, v in zip(m["keys"], p)}
            out[m["id"]] = {"type": "choice", "choice": m["keys"][max(range(len(p)), key=lambda i: p[i])], "confidence": r2(choice_confidence(p)), "probabilities": dist}
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {"type": "score", "score": r2(score), "legend": m["legend"], "probabilities": {str(i): float(v) for i, v in enumerate(p)}, "confidence": r2(score_confidence(p))}
    return out


def output_tokens(tok, answers: dict) -> int:
    """Billing-style figure: tokens of the serialised answers. Not a measure of generation (there is none)."""
    tok = getattr(tok, "tokenizer", tok)
    return len(tok(json.dumps(answers), add_special_tokens=False).input_ids)
