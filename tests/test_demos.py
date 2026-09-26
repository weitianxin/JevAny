import copy
import json
import subprocess
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from jevany.demos import CASES, make_environment
from jevany.demos.server import DemoApplication, ROOT, make_server


class TinyEnvironment:
    ACTION_LOOKUP = {"right": "Move right", "left": "Move left"}

    def __init__(self):
        self.position = 0
        self.done = self.success = False
        self.feedback = "Ready"
        self.frames = []
        self.closed = False

    def observe(self):
        return {"position": self.position}

    def get_all_actions(self):
        return [] if self.done else list(self.ACTION_LOOKUP)

    def render(self):
        return "fake pixels"

    def step(self, action):
        self.position += 1 if action == "right" else -1
        self.success = self.position == 2
        self.done = self.success
        self.feedback = "Moved"
        return self.observe(), float(self.success), self.done, {"success": self.success}

    def close(self):
        self.closed = True


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("jevany.demos.server.frame_uri", lambda _: "data:image/jpeg;base64,dGVzdA==")
    result = DemoApplication(factory=lambda *_: TinyEnvironment())
    yield result
    result.close()


def test_replay_server_imports_without_game_or_model_dependencies():
    script = """
import sys
import jevany.demos.server
assert not {'torch', 'boto3', 'vizdoom', 'pybullet', 'crafter', 'numpy', 'PIL'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_live_actions_are_checked_and_stale_clicks_cannot_repeat(app):
    start = app.start("arm", 17)
    with pytest.raises(ValueError, match="unavailable"):
        app.step(start["revision"], action="teleport")
    assert app.env.position == 0 and app.trace == []
    first = app.step(start["revision"], action="right")
    with pytest.raises(ValueError, match="another request"):
        app.step(start["revision"], action="right")
    assert app.env.position == 1
    final = app.step(first["revision"], action="right")
    assert final["done"] and final["success"] and final["actions"] == {}
    with pytest.raises(ValueError, match="ended"):
        app.step(final["revision"], action="left")


def test_model_receives_current_image_state_and_history(app):
    class Model:
        model_id = "test-model"

        def __call__(self, request):
            self.request = copy.deepcopy(request)
            return {"answers": {"action": {"type": "choice", "choice": "right",
                     "confidence": .6, "probabilities": {"right": .8, "left": .2}}}}

    model = Model()
    app.client = model
    start = app.start("arm", 17)
    first = app.step(start["revision"], model=True)
    assert model.request["media"][0]["type"] == "image"
    assert model.request["state"]["observation"] == {"position": 0}
    assert first["decision"]["probabilities"] == {"right": .8, "left": .2}
    app.images = False
    app.step(first["revision"], model=True)
    assert "media" not in model.request
    assert model.request["state"]["recent_actions"] == [{"action": "right", "feedback": "Moved"}]
    assert model.request["model"] == "test-model"


def test_bad_model_answer_does_not_execute_or_substitute_an_action(app):
    class BadModel:
        model_id = "bad-model"

        def __call__(self, request):
            return {"answers": {"action": {"type": "choice", "choice": "left",
                     "confidence": .6, "probabilities": {"right": .8, "left": .2}}}}

    app.client = BadModel()
    start = app.start("doom", 17)
    with pytest.raises(ValueError, match="argmax"):
        app.step(start["revision"], model=True)
    assert app.env.position == 0 and app.trace == [] and app.revision == start["revision"]


@pytest.mark.parametrize("case,seed", [("missing", 17), ("doom", -1), ("arm", True), ("crafter", "17")])
def test_start_rejects_invalid_fixtures(app, case, seed):
    with pytest.raises(ValueError):
        app.start(case, seed)
    assert app.env is None


def test_restart_closes_previous_physics_instance(app):
    app.start("arm", 17)
    old = app.env
    app.start("crafter", 19)
    assert old.closed and app.env is not old


def test_http_start_static_assets_and_rejected_requests(app):
    server = make_server(app, 0)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert b"JevAny Playground" in response.read()
        with urlopen(base + "/api/config") as response:
            assert set(json.load(response)["cases"]) == {"arm", "crafter", "doom"}
        start = Request(base + "/api/start", data=b'{"case":"arm","seed":17}',
                        headers={"Content-Type": "application/json", "Origin": base})
        with urlopen(start) as response:
            revision = json.load(response)["revision"]
        for body, origin, expected in [
            (b'[]', base, 400), (b'{"case":"arm"}', "https://unrelated.example", 403),
        ]:
            req = Request(base + "/api/start", data=body, headers={"Content-Type": "application/json", "Origin": origin})
            with pytest.raises(HTTPError) as exc:
                urlopen(req)
            assert exc.value.code == expected
        with pytest.raises(HTTPError) as exc:
            urlopen(base + "/static/%2e%2e/%2e%2e/cli.py")
        assert exc.value.code == 404
        app.lock.acquire()
        try:
            with pytest.raises(HTTPError) as exc:
                urlopen(Request(base + "/api/step", data=json.dumps({"revision": revision, "action": "right"}).encode(),
                                headers={"Content-Type": "application/json"}))
            assert exc.value.code == 409
        finally:
            app.lock.release()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_packaged_replays_are_complete_and_preserve_controller_provenance():
    for case in CASES:
        replay = json.loads((ROOT / "recordings" / case / "replay.json").read_text())
        assert replay["case"] == case and len(replay["steps"]) > 1
        assert replay["controller"] == ("model" if case == "arm" else "scripted")
        for step in replay["steps"]:
            for frame in step["frames"]:
                assert (ROOT / frame.removeprefix("/")).is_file()
            decision = step["decision"]
            if decision:
                assert decision["action"] in step["actions"]
                probabilities = decision["probabilities"]
                if replay["controller"] == "scripted":
                    assert probabilities is None
                else:
                    assert set(probabilities) == set(step["actions"])
                    assert sum(probabilities.values()) == pytest.approx(1, abs=1e-5)


@pytest.mark.demo
@pytest.mark.parametrize("case", list(CASES))
def test_native_environment_reset_step_and_invalid_action(case):
    pytest.importorskip(CASES[case]["package"])
    env = make_environment(case, seed=17)
    try:
        before = env.observe()
        with pytest.raises(ValueError, match="unavailable"):
            env.step("this_action_does_not_exist")
        assert env.observe() == before
        image = env.render()
        assert image.ndim == 3 and image.shape[2] == 3
        action = {"doom": "forward", "crafter": "move_right", "arm": "approach"}[case]
        observation, reward, done, info = env.step(action)
        assert observation != before and isinstance(done, bool)
        json.dumps(observation)
        assert isinstance(info["success"], bool)
    finally:
        env.close()


@pytest.mark.demo
def test_peg_insertion_requires_contact_alignment_and_release():
    pytest.importorskip("pybullet")
    env = make_environment("arm", seed=17)
    try:
        for action in ("approach", "grasp", "lift", "cyan_socket", "seat", "release", "finish"):
            env.step(action)
        assert env.done and not env.success
        env.reset(seed=17)
        for action in ("approach", "lower", "grasp", "lift", "cyan_socket", "seat", "release"):
            env.step(action)
        assert env.success and all(env.checks().values())
    finally:
        env.close()


@pytest.mark.demo
def test_doom_does_not_advance_while_waiting_for_inference():
    pytest.importorskip("vizdoom")
    import time
    env = make_environment("doom", seed=17)
    try:
        before = env.game.get_episode_time()
        time.sleep(.08)
        assert env.game.get_episode_time() == before
        env.step("forward")
        assert env.observe()["game_ticks"] > before
    finally:
        env.close()


@pytest.mark.demo
def test_doom_config_is_temporary(tmp_path, monkeypatch):
    pytest.importorskip("vizdoom")
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    env = make_environment("doom", seed=17)
    directory = Path(env.config_directory.name)
    try:
        env.step("forward")
        assert not (tmp_path / "_vizdoom.ini").exists()
    finally:
        env.close()
    assert not directory.exists()
