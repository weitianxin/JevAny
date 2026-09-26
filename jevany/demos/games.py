"""CPU adapters for Crafter and ViZDoom, with the simulation paused between steps."""
from pathlib import Path
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
        self.env.reset()
        self.names = list(self.env.action_names)
        self.steps, self.done, self.success = 0, False, False
        self.feedback = "Collect wood from a tree with Interact, then build your tools."
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
        return {
            "inventory": dict(player.inventory),
            "achievements": dict(player.achievements),
            "facing_xy": [int(v) for v in player.facing],
            "visible_grid": cells,
            "grid_orientation": "Rows run north to south, columns west to east; player is at row 3, column 4.",
            "turn": self.steps, "feedback": self.feedback,
        }

    def get_all_actions(self) -> list[str]:
        return [] if self.done else self.names[:]

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if action not in self.get_all_actions():
            raise ValueError(f"unavailable Crafter action: {action!r}")
        before = self.env._player.achievements.copy()
        _, reward, terminal, info = self.env.step(self.names.index(action))
        self.steps += 1
        unlocked = [key for key, value in info["achievements"].items() if value > before[key]]
        goals = ("collect_wood", "place_table", "make_wood_pickaxe", "collect_stone")
        self.success = all(info["achievements"][key] > 0 for key in goals) and info["inventory"]["health"] > 0
        self.done = bool(terminal or self.success)
        self.feedback = ("Unlocked: " + ", ".join(unlocked).replace("_", " ") if unlocked
                         else f"{action.replace('_', ' ').capitalize()} completed.")
        if self.done:
            self.feedback = ("Stone collected with a crafted pickaxe." if self.success else
                             "Episode ended before the crafting goal was completed.")
        self.frames = [self.render()]
        return self.observe(), float(reward), self.done, {"success": self.success}

    def render(self):
        return self.env.render()

    def close(self) -> None:
        pass


class Doom:
    """Run Freedoom's corridor scenario in synchronous mode with bounded controls."""

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

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        import vizdoom as vzd
        self.close()
        self.vzd = vzd
        self.game = vzd.DoomGame()
        self.config_directory = TemporaryDirectory(prefix="jevany-doom-")
        try:
            self.game.set_doom_config_path(str(Path(self.config_directory.name) / "vizdoom.ini"))
            self.game.load_config(str(Path(vzd.scenarios_path) / "deadly_corridor.cfg"))
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
            self.game.set_episode_timeout(CASES["doom"]["limit"] * 8 + 1)
            self.game.init()
            self.game.new_episode()
        except Exception:
            self.close()
            raise
        self.buttons = list(self.game.get_available_buttons())
        self.steps, self.done, self.success = 0, False, False
        self.feedback = "Reach the green armor at the far end of the corridor."
        self.frames = []
        self.image = self.game.get_state().screen_buffer.copy()
        self.last_ticks = self.game.get_episode_time()
        return self.observe()

    def observe(self) -> dict[str, Any]:
        state = self.game.get_state()
        variables = {name.lower(): float(self.game.get_game_variable(getattr(self.vzd.GameVariable, name)))
                     for name in ("HEALTH", "AMMO2", "KILLCOUNT", "ARMOR")}
        visible = [] if state is None else [
            {"object": label.object_name,
             "screen_box_xywh": [label.x, label.y, label.width, label.height]}
            for label in state.labels if label.object_name != "DoomPlayer"
        ]
        return {**variables, "visible_objects": visible, "image_size": [640, 480],
                "game_ticks": self.last_ticks,
                "decisions": self.steps, "feedback": self.feedback}

    def get_all_actions(self) -> list[str]:
        return [] if self.done else list(self.ACTION_LOOKUP)

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if action not in self.get_all_actions():
            raise ValueError(f"unavailable Doom action: {action!r}")
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
        self.done = self.game.is_episode_finished() or self.steps >= CASES["doom"]["limit"]
        # The scenario exits when armor is reached. Death and timeout are failures.
        armor = self.game.get_game_variable(self.vzd.GameVariable.ARMOR)
        self.success = bool(self.done and not self.game.is_player_dead() and armor > 0)
        self.feedback = f"{action.replace('_', ' ').capitalize()} executed; reward {reward:+.1f}."
        if self.done:
            self.feedback = "Corridor completed." if self.success else "Run ended before reaching the armor."
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
