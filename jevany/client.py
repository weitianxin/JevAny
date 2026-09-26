"""Python clients using the same System One request and response as HTTP."""
import json
import math
import urllib.request
from typing import Any
from urllib.parse import urlparse

from .api import JSONContent, Media, Question, SystemOneRequest, validate_response


class DecisionClient:
    """Common convenience methods for local and HTTP inference."""

    def system_one(
        self,
        state: JSONContent,
        questions: dict[str, Question | dict],
        *,
        model: str | None = None,
        media: list[Media | dict] | None = None,
    ) -> dict[str, Any]:
        """Evaluate typed questions; return the JSON-compatible response body."""
        request = SystemOneRequest(
            state=state, questions=questions, model=model or self.model_id,
            media=media or [],
        )
        return self(request)

    def __call__(self, request: SystemOneRequest | dict) -> dict[str, Any]:
        raise NotImplementedError


class JevClient(DecisionClient):
    """Call a JevAny server or a compatible TypeSafe endpoint.

    Invalid requests raise ValueError before sending. HTTP and connection errors
    propagate from urllib; malformed responses raise ValueError.
    """

    def __init__(
        self, base_url: str = "http://127.0.0.1:8008",
        api_key: str | None = "local", timeout: float = 120,
        model: str = "jevany-latest",
    ) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("decision endpoint must use HTTP or HTTPS and have a hostname")
        if parsed.scheme == "http" and parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("non-loopback decision endpoints must use HTTPS")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("base_url must not contain credentials, a query or a fragment")
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.api_key, self.timeout, self.model_id = api_key, timeout, model

    def __call__(self, request: SystemOneRequest | dict) -> dict[str, Any]:
        validated = request if isinstance(request, SystemOneRequest) else SystemOneRequest.model_validate(request)
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            self.url, data=validated.model_dump_json().encode(), method="POST", headers=headers,
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            return validate_response(validated, json.loads(response.read()))
