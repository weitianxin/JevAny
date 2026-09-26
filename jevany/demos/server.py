"""Serve packaged replays and optional CPU environments on the loopback interface."""
import argparse
import base64
import importlib.util
import io
import json
import mimetypes
import os
from pathlib import Path
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import unquote, urlsplit
import webbrowser

from jevany.api import validate_response
from jevany.client import DecisionClient, JevClient
from . import CASES, DemoEnvironment, make_environment
from .context import decision_request

ROOT = Path(__file__).parent


def frame_uri(image) -> str:
    """Encode an RGB array as a JPEG data URI."""
    from PIL import Image
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class DemoApplication:
    """One local interactive run; concurrent requests cannot change its state."""

    def __init__(self, client: DecisionClient | None = None, *, images: bool = True,
                 factory: Callable[[str, int], DemoEnvironment] = make_environment):
        self.client, self.images, self.factory = client, images, factory
        self.lock = threading.Lock()
        self.env, self.case = None, None
        self.revision = 0
        self.trace = []
        self.snapshot = None

    def config(self) -> dict[str, Any]:
        return {
            "cases": CASES,
            "model": self.client.model_id if self.client else None,
            "images": self.images,
            "installed": {key: all(importlib.util.find_spec(name) is not None
                                   for name in (meta["package"], "PIL", "numpy"))
                          for key, meta in CASES.items()},
        }

    def start(self, case: str, seed: int) -> dict[str, Any]:
        if case not in CASES:
            raise ValueError(f"unknown environment {case!r}")
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise ValueError("seed must be an integer between 0 and 2147483647")
        try:
            env = self.factory(case, seed)
        except ImportError as error:
            raise ValueError(f"This environment needs the {CASES[case]['extra']} extra. Run: "
                             f"python -m pip install -e '.[{CASES[case]['extra']}]'") from error
        if self.env is not None:
            self.env.close()
        self.env, self.case, self.seed = env, case, seed
        self.trace = []
        self.revision += 1
        self.snapshot = self._state([frame_uri(env.render())])
        return self.snapshot

    def _state(self, frames: list[str]) -> dict[str, Any]:
        return {
            "case": self.case, "revision": self.revision, "seed": self.seed,
            "observation": self.env.observe(), "frames": frames,
            "actions": {key: self.env.ACTION_LOOKUP[key] for key in self.env.get_all_actions()},
            "step": len(self.trace), "done": self.env.done, "success": self.env.success,
            "feedback": self.env.feedback,
        }

    def step(self, revision: int, *, action: str | None = None,
             model: bool = False) -> dict[str, Any]:
        if self.env is None:
            raise ValueError("start a new run first")
        if type(revision) is not int or revision != self.revision:
            raise ValueError("this run changed in another request; start a new run")
        if self.env.done:
            raise ValueError("this episode has ended; start a new run")
        if type(model) is not bool or (model and action is not None):
            raise ValueError("choose either a manual action or a model decision")
        before = self.env.observe()
        actions = {key: self.env.ACTION_LOOKUP[key] for key in self.env.get_all_actions()}
        probabilities = None
        started = time.monotonic()
        if model:
            if self.client is None:
                raise ValueError("restart with --base-url http://127.0.0.1:8008 --text-only to connect a model")
            request = decision_request(self.case, self.client.model_id, before, actions, self.trace)
            if self.images:
                request["media"] = [{"type": "image", "uri": frame_uri(self.env.render())}]
            response = validate_response(request, self.client(request))
            answer = response["answers"]["action"]
            action, probabilities = answer["choice"], answer["probabilities"]
        if not isinstance(action, str) or action not in actions:
            raise ValueError(f"unavailable action {action!r}; choose a displayed control")
        after, reward, done, info = self.env.step(action)
        self.revision += 1
        decision = {
            "controller": "model" if model else "manual", "action": action,
            "probabilities": probabilities, "observation": before, "next_observation": after,
            "reward": reward, "done": done, "success": info["success"],
            "feedback": self.env.feedback, "seconds": round(time.monotonic() - started, 3),
        }
        self.trace.append(decision)
        frames = [frame_uri(image) for image in (self.env.frames or [self.env.render()])]
        self.snapshot = {**self._state(frames), "decision": decision}
        return self.snapshot

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


def make_server(app: DemoApplication, port: int = 8090) -> ThreadingHTTPServer:
    """Bind only to localhost; serve assets, replay data, and one live run."""
    class Handler(BaseHTTPRequestHandler):
        def send(self, value, status=200, content_type="application/json"):
            body = value if isinstance(value, bytes) else json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            if path == "/api/config":
                return self.send(app.config())
            if path == "/api/trace":
                if not app.lock.acquire(blocking=False):
                    return self.send({"error": "wait for the current action to finish"}, 409)
                try:
                    return self.send({"case": app.case, "seed": getattr(app, "seed", None),
                                      "model": app.client.model_id if app.client else None,
                                      "steps": app.trace})
                finally:
                    app.lock.release()
            path = "/static/index.html" if path == "/" else path
            if not path.startswith(("/static/", "/recordings/")):
                return self.send({"error": "not found"}, 404)
            file = (ROOT / path.lstrip("/")).resolve()
            if not file.is_relative_to(ROOT.resolve()) or not file.is_file():
                return self.send({"error": "not found"}, 404)
            return self.send(file.read_bytes(), content_type=mimetypes.guess_type(file.name)[0] or "application/octet-stream")

        def do_POST(self):
            # JSON plus same-origin checks keep unrelated browser pages from controlling a run.
            origin = self.headers.get("Origin")
            expected_origin = f"http://{self.headers.get('Host')}"
            if origin is not None and origin != expected_origin:
                return self.send({"error": "cross-origin actions are not allowed"}, 403)
            if self.headers.get_content_type() != "application/json":
                return self.send({"error": "send application/json"}, 415)
            if self.path not in ("/api/start", "/api/step"):
                return self.send({"error": "not found"}, 404)
            if not app.lock.acquire(blocking=False):
                return self.send({"error": "another action is still running"}, 409)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("JSON action body must be between 1 and 4096 bytes")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("send a JSON object")
                if self.path == "/api/start":
                    result = app.start(body.get("case"), body.get("seed", 17))
                else:
                    result = app.step(body.get("revision"), action=body.get("action"),
                                      model=body.get("model", False))
                self.send(result)
            except (ValueError, TypeError) as error:
                self.send({"error": str(error)}, 400)
            except Exception as error:
                self.send({"error": f"{type(error).__name__}: {error}"}, 502)
            finally:
                app.lock.release()

        def log_message(self, *_):
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="jevany demo", description="Open the local JevAny playground. Replays need no GPU or model.")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--base-url", help="JevAny HTTP server for live model decisions")
    parser.add_argument("--model", default="jevany-latest", help="model identity sent to the server")
    parser.add_argument("--timeout", type=float, default=120, help="model request timeout in seconds")
    parser.add_argument("--text-only", action="store_true", help="send measured state without an image")
    parser.add_argument("--no-open", action="store_true", help="print the URL without opening a browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be between 0 and 65535")
    client = JevClient(args.base_url, timeout=args.timeout, model=args.model) if args.base_url else None
    app = DemoApplication(client, images=not args.text_only)
    try:
        server = make_server(app, args.port)
    except OSError as error:
        parser.exit(2, f"Cannot open demo port {args.port}: {error}. Try --port 8091.\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"JevAny playground: {url}\nReplays are ready. Press Ctrl+C to stop.", flush=True)
    if not args.no_open and (os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32")):
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        with app.lock:
            app.close()
