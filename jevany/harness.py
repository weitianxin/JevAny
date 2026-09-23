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

    def compile(self, task: str, evidence) -> dict:
        evidence_context = evidence if self.include_evidence_in_planner else self.evidence_schema(evidence)
        evidence_label = "Evidence" if self.include_evidence_in_planner else "Evidence schema"
        prompt = f"""Convert the task below into JevAny questions.
Return JSON only with the key questions. Each question must be one of:
choice with a criteria object, noul, or score with an ordered criteria array.
Do not answer the question or include labels. The evidence is attached as state by the caller and cannot be changed by you.

Task: {task}
{evidence_label}: {json.dumps(evidence_context, ensure_ascii=False)}"""
        generated = self.generator.generate(prompt, {"max_output_tokens": 2048, "temperature": 0.0})
        compiled = json_object(generated["text"])
        if set(compiled) != {"questions"}:
            raise ValueError("planner output must contain only questions")
        request = {"state": {"task": task, "evidence": evidence}, "questions": compiled["questions"],
                   "model": self.model}
        return SystemOneRequest.model_validate(request).model_dump(mode="json")

    def run(self, task: str, evidence) -> dict:
        request = self.compile(task, evidence)
        return {"request": request, "decision": validate_response(request, self.decide(request))}
