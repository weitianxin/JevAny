"""CPU adapters for Crafter and ViZDoom, with the simulation paused between steps."""
from pathlib import Path
from math import atan2, degrees
import re
import struct
from tempfile import TemporaryDirectory
from typing import Any

from . import CASES


class Crafter:
    """Expose Crafter's 17 native actions and only the player's visible surroundings."""

    ACTION_LOOKUP = {
        "noop": "Wait one turn.",
        "move_left": "Move or face west.",
        "move_right": "Move or face east.",
        "move_up": "Move or face north.",
        "move_down": "Move or face south.",
        "do": "Interact with the tile in front: collect, drink, eat, or attack.",
        "sleep": "Sleep to restore energy; the world continues to advance.",
        "place_stone": "Place stone in front; consumes one stone.",
        "place_table": "Place a crafting table in front; consumes two wood.",
        "place_furnace": "Place a furnace in front; consumes four stone.",
        "place_plant": "Plant a sapling in front.",
        "make_wood_pickaxe": "Make a wood pickaxe near a table; consumes one wood.",
        "make_stone_pickaxe": "Make a stone pickaxe near a table; consumes one wood and one stone.",
        "make_iron_pickaxe": "Make an iron pickaxe near a table and furnace; consumes wood, coal, and iron.",
        "make_wood_sword": "Make a wood sword near a table; consumes one wood.",
        "make_stone_sword": "Make a stone sword near a table; consumes one wood and one stone.",
        "make_iron_sword": "Make an iron sword near a table and furnace; consumes wood, coal, and iron.",
    }

    def __init__(self, seed: int = 17):
        self.reset(seed)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        import crafter
        self.env = crafter.Env(size=(360, 360), seed=17 if seed is None else seed,
                               length=CASES["crafter"]["limit"])
        self.image = self.env.reset()
        self.names = list(self.env.action_names)
        self.steps, self.done, self.success = 0, False, False
        self.feedback = "Collect wood from a tree with Interact, then build your tools."
        self.origin = self.env._player.pos.copy()
        self.seen_resources = {}
        self.visits = {(0, 0): 1}
        self.last_action_effect = None
        self.frames = []
        return self.observe()

    def observe(self) -> dict[str, Any]:
        # Crafter 1.8.3 exposes inventory through info after a step, but not reset.
        # Crop its semantic state to the exact 9 × 7 terrain viewport before use.
        player = self.env._player
        cells = []
        for dy in range(-3, 4):
            row = []
            for dx in range(-4, 5):
                position = player.pos + (dx, dy)
                if not all(0 <= v < bound for v, bound in zip(position, self.env._world.area)):
                    row.append("boundary")
                    continue
                material, obj = self.env._world[position]
                row.append(type(obj).__name__.lower() if obj is not None else material)
            cells.append(row)
        position = [int(v) for v in player.pos - self.origin]
        for row, tiles in enumerate(cells):
            for column, tile in enumerate(tiles):
                coordinate = (position[0] + column - 4, position[1] + row - 3)
                if tile in ("tree", "stone", "table", "water"):
                    self.seen_resources[coordinate] = (tile, self.steps)
                else:
                    self.seen_resources.pop(coordinate, None)
        facing = [int(v) for v in player.facing]
        return {
            "inventory": dict(player.inventory),
            "achievements": dict(player.achievements),
            "position_xy": position,
            "position_visits": self.visits.get(tuple(position), 0),
            "adjacent_visits": {
                name: self.visits.get((position[0] + dx, position[1] + dy), 0)
                for name, (dx, dy) in {"west": (-1, 0), "east": (1, 0),
                                       "north": (0, -1), "south": (0, 1)}.items()
            },
            "facing_xy": facing,
            "front_tile": cells[3 + facing[1]][4 + facing[0]],
            "visible_grid": cells,
            "grid_orientation": "Rows run north to south, columns west to east; player is at row 3, column 4.",
            "known_resources": [
                {"tile": tile, "position_xy": list(coordinate), "last_seen_turn": turn}
                for coordinate, (tile, turn) in self.seen_resources.items()
            ],
            "last_action_effect": self.last_action_effect,
            "action_context": self._action_context(cells),
            "turn": self.steps, "feedback": self.feedback,
        }

    def _action_context(self, cells: list[list[str]]) -> dict[str, str]:
        from crafter import constants
        inventory = self.env._player.inventory
        facing = self.env._player.facing
        front = cells[3 + facing[1]][4 + facing[0]]
        nearby = {tile for row in cells[2:5] for tile in row[3:6]}
        context = {}
        x, y = (int(v) for v in self.env._player.pos - self.origin)
        for action, (dx, dy) in {
            "move_left": (-1, 0), "move_right": (1, 0),
            "move_up": (0, -1), "move_down": (0, 1),
        }.items():
            tile = cells[3 + dy][4 + dx]
            result = ("move one tile" if tile in constants.walkable else
                      "enter lava and die" if tile == "lava" else "turn to face it without moving")
            visits = self.visits.get((x + dx, y + dy), 0)
            context[action] = f"Adjacent destination is {tile}: {result}. Previously visited {visits} time(s)."
        collect = constants.collect.get(front)
        context["do"] = f"Current interaction target is the adjacent {front}."
        if collect:
            context["do"] += " Can yield " + ", ".join(
                f"{amount} {item}" for item, amount in collect["receive"].items()) + "."
            missing = [f"{item}={amount} (carrying {inventory[item]})"
                       for item, amount in collect["require"].items() if inventory[item] < amount]
            if missing:
                context["do"] += " Requirements NOT met: " + ", ".join(missing) + "."
            if collect.get("probability", 1) < 1:
                context["do"] += f" Yield probability is {collect['probability']:g}."
        elif front in ("table", "furnace", "sand", "path", "boundary"):
            context["do"] += " Interacting with this tile produces no item."
        for prefix, recipes in (("place", constants.place), ("make", constants.make)):
            for item, recipe in recipes.items():
                action = f"{prefix}_{item}"
                if action not in self.names:
                    continue
                missing = [f"{key}={amount} (carrying {inventory[key]})"
                           for key, amount in recipe["uses"].items() if inventory[key] < amount]
                missing += [f"nearby {utility}" for utility in recipe.get("nearby", []) if utility not in nearby]
                if "where" in recipe and front not in recipe["where"]:
                    missing.append(f"empty {'/'.join(recipe['where'])} in front (currently {front})")
                context[action] = ("Requirements NOT met: " + ", ".join(missing) + "." if missing
                                   else "Requirements are met.")
                count = self.env._player.achievements.get(action, 0)
                context[action] += f" Already completed {count} time(s) this episode."
        return context

    def get_all_actions(self) -> list[str]:
        return [] if self.done else self.names[:]

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if action not in self.get_all_actions():
            raise ValueError(f"unavailable Crafter action: {action!r}")
        before = self.observe()
        self.image, reward, terminal, info = self.env.step(self.names.index(action))
        self.steps += 1
        unlocked = [key for key, value in info["achievements"].items()
                    if value > before["achievements"][key]]
        goals = ("collect_wood", "place_table", "make_wood_pickaxe", "collect_stone")
        self.success = all(info["achievements"][key] > 0 for key in goals) and info["inventory"]["health"] > 0
        self.done = bool(terminal or self.success)
        movement = [int(v) - start for v, start in
                    zip(self.env._player.pos - self.origin, before["position_xy"])]
        changes = {key: value - before["inventory"][key]
                   for key, value in info["inventory"].items()
                   if value != before["inventory"][key]}
        self.last_action_effect = {
            "action": action, "front_tile_before": before["front_tile"],
            "movement_xy": movement, "inventory_changes": changes, "unlocked": unlocked,
        }
        effects = []
        if any(movement):
            position = tuple(int(v) for v in self.env._player.pos - self.origin)
            self.visits[position] = self.visits.get(position, 0) + 1
            effects.append(f"moved ({movement[0]:+d}, {movement[1]:+d})")
        elif action.startswith("move_"):
            effects.append("faced that direction but did not move")
        if changes:
            effects.append(", ".join(f"{key} {value:+d}" for key, value in changes.items()))
        if unlocked:
            effects.append("achieved " + ", ".join(unlocked))
        if not effects:
            effects.append("no change to position, inventory, or achievements")
        target = "" if action.startswith("move_") else f" facing {before['front_tile']}"
        x, y = self.env._player.pos - self.origin
        self.feedback = f"{action}{target}: " + "; ".join(effects) + f". Position ({x}, {y})."
        if self.done:
            self.feedback = ("Stone collected with a crafted pickaxe." if self.success else
                             "Episode ended before the crafting goal was completed.")
        self.frames = [self.render()]
        return self.observe(), float(reward), self.done, {"success": self.success}

    def render(self):
        # Crafter's night lighting consumes the simulation RNG when rendered.
        # Reuse the native observation so reading an image cannot change a run.
        return self.image

    def close(self) -> None:
        pass


class Doom:
    """Clear the final room of Freedoom's corridor, then advance past its enemies."""

    frame_duration_ms = 2000 / 35
    step_pause_ms = 0

    ACTION_LOOKUP = {
        "forward": "Move forward for 8 game ticks.",
        "backward": "Move backward for 8 game ticks.",
        "strafe_left": "Strafe left for 8 game ticks without changing aim.",
        "strafe_right": "Strafe right for 8 game ticks without changing aim.",
        "turn_left": "Turn left for 4 game ticks.",
        "turn_right": "Turn right for 4 game ticks.",
        "attack": "Fire for 8 game ticks in the current direction.",
        "wait": "Wait for 8 game ticks.",
    }
    CONTROLS = {
        "forward": "MOVE_FORWARD", "backward": "MOVE_BACKWARD",
        "strafe_left": "MOVE_LEFT", "strafe_right": "MOVE_RIGHT",
        "turn_left": "TURN_LEFT", "turn_right": "TURN_RIGHT", "attack": "ATTACK",
    }

    def __init__(self, seed: int = 17):
        self.game = None
        self.config_directory = None
        self.reset(seed)

    @staticmethod
    def _final_room(source: Path, destination: Path) -> None:
        """Move the native map's spawn to its final room and keep that room's pair."""
        data = source.read_bytes()
        magic, count, directory = struct.unpack_from("<4sii", data)
        lumps = []
        for index in range(count):
            offset, size, name = struct.unpack_from("<ii8s", data, directory + index * 16)
            content = data[offset:offset + size]
            if name.rstrip(b"\0") == b"TEXTMAP":
                def thing(match):
                    block = match[0]
                    identity = re.search(r"\bid\s*=\s*(\d+);", block)
                    identity = int(identity[1]) if identity else None
                    if identity in (10, 11, 12, 13):
                        return ""
                    if identity == 1:
                        return re.sub(r"\bx\s*=\s*[^;]+;", "x = 896.0;", block)
                    return block
                content = re.sub(r"\bthing\s*\{[^}]*\}", thing, content.decode()).encode()
            lumps.append((name, content))
        output, entries = bytearray(12), bytearray()
        for name, content in lumps:
            entries.extend(struct.pack("<ii8s", len(output), len(content), name))
            output.extend(content)
        struct.pack_into("<4sii", output, 0, magic, count, len(output))
        destination.write_bytes(output + entries)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        import vizdoom as vzd
        self.close()
        self.vzd = vzd
        self.game = vzd.DoomGame()
        self.config_directory = TemporaryDirectory(prefix="jevany-doom-")
        try:
            self.game.set_doom_config_path(str(Path(self.config_directory.name) / "vizdoom.ini"))
            self.game.load_config(str(Path(vzd.scenarios_path) / "deadly_corridor.cfg"))
            scenario = Path(self.config_directory.name) / "final_room.wad"
            self._final_room(Path(vzd.scenarios_path) / "deadly_corridor.wad", scenario)
            self.game.set_doom_scenario_path(str(scenario))
            self.game.set_window_visible(False)
            self.game.set_sound_enabled(False)
            self.game.set_mode(vzd.Mode.PLAYER)
            self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)
            self.game.set_render_weapon(True)
            self.game.set_render_hud(True)
            self.game.set_render_crosshair(True)
            self.game.set_labels_buffer_enabled(True)
            self.game.set_doom_skill(1)
            self.game.set_seed(17 if seed is None else seed)
            self.game.set_episode_start_time(14)
            self.game.set_episode_timeout(CASES["doom"]["limit"] * 8 + 1)
            self.game.init()
            self.game.new_episode()
        except Exception:
            self.close()
            raise
        self.buttons = list(self.game.get_available_buttons())
        self.steps, self.done, self.success = 0, False, False
        self.feedback = "Kill the enemy on the left and the enemy on the right, then advance through the room."
        self.cleared_at_x = None
        self.frames = []
        self.image = self.game.get_state().screen_buffer.copy()
        self.last_ticks = self.game.get_episode_time()
        return self.observe()

    def observe(self) -> dict[str, Any]:
        state = self.game.get_state()
        variables = {name.lower(): float(self.game.get_game_variable(getattr(self.vzd.GameVariable, name)))
                     for name in ("HEALTH", "AMMO2", "KILLCOUNT", "ARMOR", "HITCOUNT",
                                  "POSITION_X", "POSITION_Y", "ANGLE", "ATTACK_READY")}
        def bearing(x, y):
            direction = degrees(atan2(y - variables["position_y"], x - variables["position_x"]))
            return (direction - variables["angle"] + 180) % 360 - 180

        visible = [] if state is None else [
            {"object": label.object_name, "id": label.object_id,
             "screen_box_xywh": [label.x, label.y, label.width, label.height],
             "bearing_degrees": bearing(label.object_position_x, label.object_position_y)}
            for label in state.labels if label.object_name != "DoomPlayer"
        ]
        observation = {**variables, "visible_objects": visible, "image_size": [640, 480],
                       "game_ticks": self.last_ticks,
                       "crosshair_x": 320, "room_exit_x": 1184,
                       "exit_bearing_degrees": bearing(1312, 0),
                       "enemies_to_kill": 2,
                       "cleared_at_x": self.cleared_at_x,
                       "decisions": self.steps, "feedback": self.feedback}
        if state is None:
            observation["image_is_current"] = False
        return observation

    def get_all_actions(self) -> list[str]:
        return [] if self.done else list(self.ACTION_LOOKUP)

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if action not in self.get_all_actions():
            raise ValueError(f"unavailable Doom action: {action!r}")
        before = self.observe()
        target = self.CONTROLS.get(action)
        control = [target is not None and button.name == target for button in self.buttons]
        self.frames, reward = [], 0.0
        for _ in range(2 if action.startswith("turn_") else 4):
            reward += self.game.make_action(control, 2)
            state = self.game.get_state()
            if state is not None:
                self.last_ticks = self.game.get_episode_time()
                self.image = state.screen_buffer.copy()
                self.frames.append(self.image)
            if self.game.is_episode_finished():
                break
        self.steps += 1
        after = self.observe()
        if self.cleared_at_x is None and after["hitcount"] >= 2 and after["killcount"] >= 2:
            self.cleared_at_x = after["position_x"]
        self.success = bool(
            not self.game.is_player_dead() and self.cleared_at_x is not None
            and after["position_x"] >= max(1184, self.cleared_at_x + 64)
        )
        self.done = self.success or self.game.is_episode_finished() or self.steps >= CASES["doom"]["limit"]
        self.feedback = (
            f"{action}: ammo {after['ammo2'] - before['ammo2']:+g}, "
            f"hits {after['hitcount'] - before['hitcount']:+g}, "
            f"kills {after['killcount'] - before['killcount']:+g}; "
            f"position ({after['position_x']:.1f}, {after['position_y']:.1f}), "
            f"heading {after['angle']:.1f} degrees."
        )
        if self.done:
            self.feedback += (" Both enemies killed; advanced through the cleared room." if self.success else
                              " Run ended before killing both enemies and advancing.")
        return self.observe(), float(reward), bool(self.done), {"success": self.success}

    def render(self):
        return self.image

    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None
        if self.config_directory is not None:
            self.config_directory.cleanup()
            self.config_directory = None
