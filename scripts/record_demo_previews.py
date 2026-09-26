"""Record explicitly scripted game tours for offline playback, without a model.

Run with the demo extra installed:
    python scripts/record_demo_previews.py --out jevany/demos/recordings
The robot replay is an existing model recording and is not replaced here.
"""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np
from PIL import Image

from jevany.demos import CASES, make_environment


class Recorder:
    def __init__(self, case, seed, root):
        self.case, self.seed = case, seed
        self.directory = root / case
        self.directory.mkdir(parents=True, exist_ok=True)
        self.env = make_environment(case, seed)
        self.steps, self.frame_index = [], 0
        self.capture()

    def capture(self, action=None, reward=0):
        frames = []
        for image in self.env.frames or [self.env.render()]:
            name = f"{self.frame_index:03d}.jpg"
            Image.fromarray(image).save(self.directory / name, quality=85)
            frames.append(f"/recordings/{self.case}/{name}")
            self.frame_index += 1
        self.steps.append({
            "case": self.case, "seed": self.seed, "step": len(self.steps),
            "observation": self.env.observe(), "frames": frames,
            "actions": dict(self.env.ACTION_LOOKUP),
            "feedback": self.env.feedback, "done": self.env.done, "success": self.env.success,
            "decision": None if action is None else {
                "controller": "scripted", "action": action,
                "probabilities": None, "reward": reward,
            },
        })

    def act(self, action):
        if self.env.done:
            return
        _, reward, _, _ = self.env.step(action)
        self.capture(action, reward)

    def save(self):
        value = {
            "case": self.case, "controller": "scripted", "seed": self.seed,
            "note": "Scripted environment preview, recorded on CPU. These actions are not JevAny "
                    "predictions and do not measure model performance. Start a new run to try your model.",
            "environment": "Crafter 1.8.3" if self.case == "crafter" else "ViZDoom 1.3.0 / Freedoom 2 / deadly_corridor",
            "steps": self.steps,
        }
        (self.directory / "replay.json").write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps({"case": self.case, "steps": len(self.steps)-1,
                          "frames": self.frame_index, "success": self.env.success,
                          "controller": "scripted"}), flush=True)


DIRECTIONS = {"move_left": (-1, 0), "move_right": (1, 0), "move_up": (0, -1), "move_down": (0, 1)}


def path_to_resource(env, material):
    """Fixture-only planner: uses the generated map to record a reproducible tour."""
    start = tuple(env._player.pos)
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        pos, path = queue.popleft()
        for action, delta in DIRECTIONS.items():
            neighbor = tuple(a+b for a, b in zip(pos, delta))
            if not all(0 <= v < bound for v, bound in zip(neighbor, env._world.area)):
                continue
            tile, obj = env._world[neighbor]
            if tile == material and obj is None:
                return path + [action, "do"]
            if neighbor not in seen and tile in ("grass", "path", "sand") and obj is None:
                seen.add(neighbor)
                queue.append((neighbor, path + [action]))
    raise RuntimeError(f"no reachable {material} in this preview fixture")


def crafter_preview(recorder):
    env = recorder.env.env
    for _ in range(3):
        for action in path_to_resource(env, "tree"):
            recorder.act(action)
    for action, delta in DIRECTIONS.items():
        if all(env._world[env._player.pos + np.array(delta) * n][0] in ("grass", "path", "sand")
               and env._world[env._player.pos + np.array(delta) * n][1] is None for n in (1, 2)):
            recorder.act(action)
            recorder.act("place_table")
            recorder.act("make_wood_pickaxe")
            break
    for action in path_to_resource(env, "stone"):
        recorder.act(action)
    if not recorder.env.success:
        raise RuntimeError("the scripted crafting fixture did not reach its goal")


def doom_preview(recorder):
    enemies = {"Zombieman", "ShotgunGuy", "DoomImp", "ChaingunGuy", "Demon", "Spectre"}
    for _ in range(48):
        if recorder.env.done:
            break
        objects = recorder.env.observe()["visible_objects"]
        targets = [obj for obj in objects if obj["object"] in enemies]
        if targets:
            target = max(targets, key=lambda obj: obj["screen_box_xywh"][2] * obj["screen_box_xywh"][3])
            x, _, width, _ = target["screen_box_xywh"]
            offset = x + width/2 - 320
            action = "attack" if abs(offset) < 65 else "turn_left" if offset < 0 else "turn_right"
        else:
            action = "forward"
        recorder.act(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("jevany/demos/recordings"))
    parser.add_argument("--cases", nargs="+", choices=["crafter", "doom"], default=["crafter", "doom"])
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    for case in args.cases:
        recorder = Recorder(case, args.seed, args.out)
        try:
            (crafter_preview if case == "crafter" else doom_preview)(recorder)
            recorder.save()
        finally:
            recorder.env.close()


if __name__ == "__main__":
    main()
