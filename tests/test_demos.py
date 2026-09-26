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
        assert replay["controller"] == "model"
        assert replay["steps"][-1]["success"]
        assert (ROOT / "recordings" / case / "000.jpg").is_file()
        for step in replay["steps"]:
            for frame in step["frames"]:
                assert (ROOT / frame.removeprefix("/")).is_file()
            decision = step["decision"]
            if decision:
                assert decision["action"] in step["actions"]
                probabilities = decision["probabilities"]
                assert set(probabilities) == set(step["actions"])
                assert sum(probabilities.values()) == pytest.approx(1, abs=1e-5)
                assert decision["action"] == max(probabilities, key=probabilities.get)
                if case == "crafter":
                    priority = decision["priority"]
                    assert priority["choice"] == max(priority["probabilities"], key=priority["probabilities"].get)


def test_playground_gifs_share_format_and_match_the_packaged_replays():
    Image = pytest.importorskip("PIL.Image")
    directory = ROOT.parents[1] / "docs" / "demos"
    manifest = json.loads((directory / "playground-format.json").read_text())
    records = {record["case"]: record for record in manifest["cases"]}
    assert set(records) == set(CASES)
    assert len({tuple(record["size"]) for record in records.values()}) == 1
    assert all(record["clip"] == records["arm"]["clip"] for record in records.values())
    for case, record in records.items():
        replay = json.loads((ROOT / "recordings" / case / "replay.json").read_text())
        assert record["decisions"] == len(replay["steps"]) - 1
        assert record["success"] == replay["steps"][-1]["success"]
        with Image.open(directory / f"playground-{case}.gif") as image:
            assert image.size == tuple(record["size"])
            assert image.is_animated and image.info["loop"] == 0
            assert image.info["comment"].startswith(manifest["format"].encode())
            assert image.info["duration"] <= 80
            image.seek(image.n_frames - 1)
            assert image.info["duration"] <= 80


def test_environment_playback_timing_is_preserved_in_snapshots(app):
    app.start("arm", 17)
    app.env.frame_duration_ms = 1000 / 30
    app.env.step_pause_ms = 0
    snapshot = app.step(app.revision, action="right")
    assert snapshot["frame_duration_ms"] == pytest.approx(1000 / 30)
    assert snapshot["step_pause_ms"] == 0


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
        action = {"doom": "forward", "crafter": "move_right", "arm": "x_plus_1cm"}[case]
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
        before = env.observe()
        env.step("close_gripper")
        assert not env.success and not env.observe()["object_between_both_fingers"]
        assert env.observe()["gripper_xyz_metres"] == pytest.approx(before["gripper_xyz_metres"], abs=.002)
        with pytest.raises(ValueError, match="unavailable"):
            env.step("cyan_socket")
        replay = json.loads((ROOT / "recordings/arm/replay.json").read_text())
        env.reset(seed=replay["seed"])
        for step in replay["steps"][1:]:
            env.step(step["decision"]["action"])
        assert env.success and all(env.checks().values())
    finally:
        env.close()


@pytest.mark.demo
@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("direction,sign", [("plus", 1), ("minus", -1)])
def test_arm_translation_changes_only_the_commanded_axis(axis, direction, sign):
    pytest.importorskip("pybullet")
    env = make_environment("arm", seed=17)
    try:
        before = env.observe()
        after, _, _, _ = env.step(f"{axis}_{direction}_1cm")
        expected = list(before["gripper_xyz_metres"])
        expected["xyz".index(axis)] += sign * .01
        assert after["gripper_xyz_metres"] == pytest.approx(expected, abs=.002)
        assert after["gripper_open"] == before["gripper_open"]
    finally:
        env.close()


@pytest.mark.parametrize("fail", [False, True])
def test_shared_camera_file_is_removed_after_inference(app, tmp_path, monkeypatch, fail):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from pathlib import Path
    from jevany.api import SystemOneRequest
    serve = pytest.importorskip("jevany.serve")
    monkeypatch.setattr(serve, "MEDIA_ROOT", str(tmp_path))

    class Model:
        model_id = "test-model"

        def __call__(self, request):
            prepared = serve.prepare(SystemOneRequest.model_validate(request))
            self.path = Path(prepared.media[0].uri)
            assert self.path.is_relative_to(tmp_path)
            with Image.open(self.path) as image:
                assert image.size == (16, 12)
            if fail:
                raise RuntimeError("test inference failure")
            return {"answers": {"action": {"type": "choice", "choice": "right",
                    "confidence": .8, "probabilities": {"right": .8, "left": .2}}}}

    app.client = Model()
    app.media_root = tmp_path
    start = app.start("arm", 17)
    app.env.render = lambda: np.zeros((12, 16, 3), dtype=np.uint8)
    if fail:
        with pytest.raises(RuntimeError, match="test inference failure"):
            app.step(start["revision"], model=True)
        assert app.env.position == 0 and app.trace == []
    else:
        app.step(start["revision"], model=True)
        assert app.env.position == 1 and len(app.trace) == 1
    assert not app.client.path.exists() and list(tmp_path.iterdir()) == []


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
