import copy

import pytest

from jevany.api import SystemOneRequest, to_record
from jevany.demos.context import decision_request
from jevany.demos.games import Crafter


def test_crafter_request_preserves_map_geometry_progress_and_all_actions():
    grid = [["grass"] * 9 for _ in range(7)]
    grid[3][4], grid[3][5] = "player", "table"
    observation = {
        "inventory": {"health": 9, "wood": 1, "wood_pickaxe": 0, "stone": 0},
        "achievements": {"collect_wood": 3, "place_table": 1, "make_wood_pickaxe": 0, "collect_stone": 0},
        "position_xy": [2, 1], "facing_xy": [1, 0], "front_tile": "table",
        "visible_grid": grid, "turn": 12, "feedback": "place_table: wood -2; achieved place_table.",
        "known_resources": [
            {"tile": "table", "position_xy": [3, 1], "last_seen_turn": 12},
            {"tile": "stone", "position_xy": [8, 2], "last_seen_turn": 8},
        ],
    }
    original = copy.deepcopy(observation)
    request = decision_request("crafter", "checkpoint", observation, Crafter.ACTION_LOOKUP, [])
    record, _ = to_record(SystemOneRequest.model_validate(request))
    assert ". . . . @ B . . ." in record["state"]
    assert "place_table=DONE" in record["state"]
    assert "make_wood_pickaxe=pending" in record["state"]
    assert "Table within crafting distance: yes" in record["state"]
    assert "6 tile(s) east and 1 tile(s) south (last seen turn 8)" in record["state"]
    assert request["questions"]["action"]["criteria"] == Crafter.ACTION_LOOKUP
    assert observation == original


@pytest.mark.demo
def test_crafter_feedback_reports_failed_crafting_and_invalid_actions():
    pytest.importorskip("crafter")
    env = Crafter(17)
    try:
        before = env.observe()
        with pytest.raises(ValueError, match="unavailable"):
            env.step("teleport")
        assert env.observe() == before
        after, _, _, _ = env.step("make_wood_pickaxe")
        assert after["inventory"] == before["inventory"]
        assert after["last_action_effect"]["inventory_changes"] == {}
        assert "no change to position, inventory, or achievements" in after["feedback"]
        assert set(env.get_all_actions()) == set(Crafter.ACTION_LOOKUP)
        assert "wood=1 (carrying 0)" in after["action_context"]["make_wood_pickaxe"]
        assert "nearby table" in after["action_context"]["make_wood_pickaxe"]
        assert "sapling" in after["action_context"]["do"]
    finally:
        env.close()


@pytest.mark.demo
def test_crafter_memory_contains_only_observed_resources_and_forgets_harvested_tree():
    crafter = pytest.importorskip("crafter")
    env = Crafter(17)
    native = crafter.Env(size=(360, 360), seed=17, length=200)
    native.reset()
    try:
        initial = env.observe()
        expected = {
            (column - 4, row - 3): tile
            for row, tiles in enumerate(initial["visible_grid"]) for column, tile in enumerate(tiles)
            if tile in ("tree", "stone", "water", "table")
        }
        assert {tuple(r["position_xy"]): r["tile"] for r in initial["known_resources"]} == expected
        for action in ("move_right", "move_right", "move_right", "do", "move_left",
                       "move_left", "move_left", "move_left", "move_left"):
            state, reward, done, _ = env.step(action)
            _, native_reward, native_done, info = native.step(native.action_names.index(action))
            assert state["inventory"] == info["inventory"]
            assert state["achievements"] == info["achievements"]
            assert reward == native_reward and done == native_done
            assert (env.env._player.pos == native._player.pos).all()
        resources = {tuple(r["position_xy"]): r for r in state["known_resources"]}
        assert (4, 0) not in resources
        assert resources[(4, 1)]["tile"] == "tree"
        assert resources[(4, 1)]["last_seen_turn"] < state["turn"]
        env.reset(41)
        assert env.observe()["position_xy"] == [0, 0]
        assert env.observe()["last_action_effect"] is None
        assert all(r["last_seen_turn"] == 0 for r in env.observe()["known_resources"])
    finally:
        env.close()


@pytest.mark.demo
def test_crafter_repeated_screenshots_do_not_change_simulation_rng():
    pytest.importorskip("crafter")
    import numpy as np
    env = Crafter(17)
    try:
        env.env._world.daylight = .1
        before = env.env._world.random.get_state()
        first, second = env.render(), env.render()
        after = env.env._world.random.get_state()
        assert np.array_equal(first, second)
        assert all(np.array_equal(a, b) for a, b in zip(before, after))
    finally:
        env.close()


@pytest.mark.demo
@pytest.mark.parametrize("invalid_stage", [None, "priority", "action"])
def test_crafter_model_selects_both_objective_and_native_action(tmp_path, invalid_stage):
    pytest.importorskip("crafter")
    from pathlib import Path
    from jevany.demos.server import DemoApplication

    class Model:
        model_id = "test-checkpoint"

        def __init__(self):
            self.requests = []

        def __call__(self, request):
            self.requests.append(copy.deepcopy(request))
            assert Path(request["media"][0]["uri"]).read_bytes().startswith(b"\x89PNG")
            question, spec = next(iter(request["questions"].items()))
            choice = "gather_wood" if question == "priority" else "move_right"
            winner = next(key for key in spec["criteria"] if key != choice) if question == invalid_stage else choice
            return {"answers": {question: {
                "type": "choice", "choice": choice, "confidence": 1,
                "probabilities": {key: float(key == winner) for key in spec["criteria"]},
            }}}

    client = Model()
    app = DemoApplication(client, media_root=tmp_path)
    try:
        start = app.start("crafter", 17)
        if invalid_stage:
            with pytest.raises(ValueError, match="argmax"):
                app.step(start["revision"], model=True)
            assert app.env.steps == 0 and app.trace == []
            assert app.revision == start["revision"]
        else:
            result = app.step(start["revision"], model=True)
            assert app.env.steps == 1
            assert result["decision"]["action"] == "move_right"
            assert result["decision"]["priority"]["choice"] == "gather_wood"
            assert client.requests[0]["media"] == client.requests[1]["media"]
            assert set(client.requests[1]["questions"]["action"]["criteria"]) == set(Crafter.ACTION_LOOKUP)
            assert "Current objective chosen by the checkpoint: gather_wood" in client.requests[1]["state"]
        assert all(not Path(request["media"][0]["uri"]).exists() for request in client.requests)
    finally:
        app.close()


@pytest.mark.demo
def test_recorder_keeps_existing_replay_if_model_request_fails(tmp_path):
    pytest.importorskip("crafter")
    from pathlib import Path
    import runpy

    class Offline:
        model_id = "offline-test"

        def __call__(self, request):
            raise RuntimeError("server unavailable")

    directory = tmp_path / "crafter"
    directory.mkdir()
    (directory / "000.jpg").write_bytes(b"existing frame")
    (directory / "replay.json").write_bytes(b"existing replay")
    record = runpy.run_path(str(Path(__file__).parents[1] / "scripts/record_demo_previews.py"))["record"]
    with pytest.raises(RuntimeError, match="server unavailable"):
        record(Offline(), "crafter", 17, tmp_path, images=False)
    assert (directory / "000.jpg").read_bytes() == b"existing frame"
    assert (directory / "replay.json").read_bytes() == b"existing replay"


@pytest.mark.demo
def test_recorder_publishes_a_complete_replay_and_removes_old_frames(tmp_path):
    pytest.importorskip("crafter")
    import json
    from pathlib import Path
    import runpy
    from jevany.demos.server import ROOT

    source = json.loads((ROOT / "recordings/crafter/replay.json").read_text())

    class RecordedAnswers:
        model_id = "recorded-answers-test"
        step = 1

        def __call__(self, request):
            question = next(iter(request["questions"]))
            decision = source["steps"][self.step]["decision"]
            if question == "priority":
                answer = decision["priority"]
            else:
                answer = {"type": "choice", "choice": decision["action"],
                          "confidence": max(decision["probabilities"].values()),
                          "probabilities": decision["probabilities"]}
                self.step += 1
            return {"answers": {question: answer}}

    directory = tmp_path / "crafter"
    directory.mkdir()
    (directory / "obsolete.jpg").write_bytes(b"old frame")
    record = runpy.run_path(str(Path(__file__).parents[1] / "scripts/record_demo_previews.py"))["record"]
    result = record(RecordedAnswers(), "crafter", source["seed"], tmp_path, images=False)
    assert result["steps"][-1]["success"]
    assert json.loads((directory / "replay.json").read_text()) == result
    assert not (directory / "obsolete.jpg").exists()
    assert not list(tmp_path.glob(".crafter-*"))
    assert len(list(directory.glob("*.jpg"))) == len(result["steps"])


@pytest.mark.demo
def test_doom_marks_the_cached_image_when_the_engine_finishes():
    pytest.importorskip("vizdoom")
    from jevany.demos.games import Doom

    env = Doom(17)
    try:
        assert "image_is_current" not in env.observe()
        while not env.done:
            observation, _, _, _ = env.step("forward")
        # Reaching the armor without shooting either enemy is no longer success.
        assert not env.success and observation["armor"] > 0
        assert observation["hitcount"] == 0
        assert observation["image_is_current"] is False
    finally:
        env.close()


@pytest.mark.demo
def test_doom_final_room_observations_and_request_match_native_targets():
    pytest.importorskip("vizdoom")
    from jevany.demos.games import Doom

    env = Doom(17)
    try:
        before = env.observe()
        assert before["position_x"] == 896 and before["position_y"] == 0
        assert before["killcount"] == before["hitcount"] == 0
        enemies = {item["object"]: item for item in before["visible_objects"]
                   if item["object"].endswith("Guy")}
        assert set(enemies) == {"ChaingunGuy", "ShotgunGuy"}
        assert enemies["ChaingunGuy"]["bearing_degrees"] > 0
        assert enemies["ShotgunGuy"]["bearing_degrees"] < 0
        request = decision_request("doom", "checkpoint", before, env.ACTION_LOOKUP, [])
        record, _ = to_record(SystemOneRequest.model_validate(request))
        assert "Kill both enemies" in record["state"]
        assert "LIVING enemy: ChaingunGuy" in record["state"]
        assert "26.6 degrees left" in record["state"]
        assert "26.6 degrees right" in record["state"]
        assert request["questions"]["action"]["criteria"] == env.ACTION_LOOKUP
        with pytest.raises(ValueError, match="unavailable"):
            env.step("auto_aim")
        assert env.observe() == before
    finally:
        env.close()


@pytest.mark.demo
def test_doom_recorded_actions_kill_both_enemies_before_advancing():
    pytest.importorskip("vizdoom")
    import json
    from jevany.demos.games import Doom
    from jevany.demos.server import ROOT

    replay = json.loads((ROOT / "recordings/doom/replay.json").read_text())
    env = Doom(replay["seed"])
    killed = set()
    cleared_step = None
    try:
        for step in replay["steps"][1:]:
            before = env.observe()
            observation, _, _, _ = env.step(step["decision"]["action"])
            # Bundled Freedoom sprites differ between supported ViZDoom versions.
            for key in ("ammo2", "hitcount", "killcount"):
                assert observation[key] == step["observation"][key]
            if observation["hitcount"] > before["hitcount"]:
                assert step["decision"]["action"] == "attack"
                assert observation["ammo2"] < before["ammo2"]
                newly_dead = {item["id"] for item in observation["visible_objects"]
                              if item["object"].startswith("Dead")} - killed
                assert len(newly_dead) == 1
                killed |= newly_dead
            if len(killed) == 2 and cleared_step is None:
                cleared_step = step["step"]
                assert not env.success
            if env.success:
                assert step["step"] > cleared_step
                assert observation["position_x"] >= max(1184, env.cleared_at_x + 64)
        assert env.success and len(killed) == 2
    finally:
        env.close()
