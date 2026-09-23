"""Composable LLM planning and JevAny decision harnesses."""
import json
import re
import urllib.request
from typing import Callable, Protocol
from urllib.parse import urlparse

from .api import SystemOneRequest, validate_response


class TextGenerator(Protocol):
    def generate(self, prompt: str, params: dict | None = None) -> dict: ...


def json_object(text: str) -> dict:
    """Parse one JSON object, accepting a fenced response from an LLM."""
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL | re.IGNORECASE)
    payload = match.group(1) if match else text[text.find("{"):text.rfind("}") + 1]
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("planner output must be a JSON object")
    return value


def normalize_questions(value):
    if isinstance(value, dict):
        return value
    if not isinstance(value, list):
        raise ValueError("planner questions must be an object or a list")
    questions = {}
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError("each planner question must be an object")
        question = dict(item)
        question_id = question.pop("id", f"question_{index}")
        if not isinstance(question_id, str) or not question_id or question_id in questions:
            raise ValueError("planner question ids must be unique non-empty strings")
        questions[question_id] = question
    return questions


class HTTPDecisionClient:
    def __init__(self, base_url="http://127.0.0.1:8008", api_key="local", timeout=120):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("decision endpoint must use HTTP or HTTPS")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("non-loopback decision endpoints must use HTTPS")
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.api_key, self.timeout = api_key, timeout

    def __call__(self, request: dict) -> dict:
        validated = SystemOneRequest.model_validate(request)
        payload = validated.model_dump(mode="json")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(), method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            return validate_response(validated, json.loads(response.read()))


class JevHarness:
    """Use an LLM as a task compiler and JevAny as the bounded decision layer."""

    def __init__(self, generator: TextGenerator, decide: Callable[[dict], dict], model="jevany-27b",
                 include_evidence_in_planner=False):
        self.generator, self.decide, self.model = generator, decide, model
        self.include_evidence_in_planner = include_evidence_in_planner

    @staticmethod
    def evidence_schema(value):
        if isinstance(value, dict):
            return {key: JevHarness.evidence_schema(item) for key, item in value.items()}
        if isinstance(value, list):
            return [JevHarness.evidence_schema(value[0])] if value else []
        return type(value).__name__

    def _compile(self, task: str, evidence) -> tuple[dict, dict]:
        evidence_context = evidence if self.include_evidence_in_planner else self.evidence_schema(evidence)
        evidence_label = "Evidence" if self.include_evidence_in_planner else "Evidence schema"
        prompt = f"""Convert the task below into JevAny questions.
Return one JSON object in this exact shape:
{{"questions":{{"execution_mode":{{"type":"choice","instructions":"Choose the execution mode.","criteria":{{"allow":"Allow it","deny":"Deny it"}}}},"rollback":{{"type":"noul","instructions":"Is a rollback checkpoint required?"}}}}}}
questions must map stable snake_case ids directly to question objects. Use type choice with a criteria object, type noul,
or type score with an ordered criteria array. Do not wrap a question under a choice, noul, or score key.
Do not answer any question or include labels. The evidence is attached as state by the caller and cannot be changed by you.

Task: {task}
{evidence_label}: {json.dumps(evidence_context, ensure_ascii=False)}"""
        generated = self.generator.generate(prompt, {"max_output_tokens": 2048})
        compiled = json_object(generated["text"])
        if set(compiled) != {"questions"}:
            raise ValueError("planner output must contain only questions")
        request = {"state": {"task": task, "evidence": evidence},
                   "questions": normalize_questions(compiled["questions"]),
                   "model": self.model}
        request = SystemOneRequest.model_validate(request).model_dump(mode="json")
        planner = {key: generated.get(key) for key in ("usage", "stop_reason") if generated.get(key) is not None}
        return request, planner

    def compile(self, task: str, evidence) -> dict:
        request, _ = self._compile(task, evidence)
        return request

    def run(self, task: str, evidence) -> dict:
        request, planner = self._compile(task, evidence)
        return {"planner": planner, "request": request,
                "decision": validate_response(request, self.decide(request))}
