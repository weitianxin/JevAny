"""Format observed game state for the checkpoint's next native action."""
from typing import Any

from . import CASES

CRAFTER_RULES = (
    "Each decision executes one native game turn. Movement also changes facing, even when "
    "a tree, rock, table or water blocks movement. Interact (do) affects ONLY the adjacent "
    "tile you face; it does not move you toward a distant resource. A tree gives one wood "
    "and becomes grass. A table costs two wood; a wood pickaxe costs one MORE wood, so "
    "the crafting task needs three wood total. Crafting works within one tile of a table "
    "(including diagonals); facing the table is unnecessary. Interacting with a table "
    "does not craft. A wood pickaxe lets you mine stone. Placing a table needs an empty "
    "grass, sand, or path tile in front. Completed milestones remain completed; a table "
    "and pickaxe can be reused. All native actions remain selectable, including actions "
    "whose ingredients or surroundings are missing."
)

DIRECTIONS = {(-1, 0): "west", (1, 0): "east", (0, -1): "north", (0, 1): "south"}
SYMBOLS = {
    "player": "@", "grass": ".", "path": ":", "sand": ",", "tree": "T",
    "stone": "R", "water": "~", "table": "B", "coal": "c", "iron": "i",
    "diamond": "d", "lava": "!", "furnace": "F", "plant": "p",
    "cow": "C", "zombie": "Z", "skeleton": "S", "arrow": "a", "boundary": "#",
}


def crafter_state(observation: dict[str, Any]) -> str:
    """Describe the current viewport and resources remembered from earlier views."""
    inventory, achievements = observation["inventory"], observation["achievements"]
    x, y = observation["position_xy"]
    grid = observation["visible_grid"]
    facing = DIRECTIONS[tuple(observation["facing_xy"])]
    goals = ("collect_wood", "place_table", "make_wood_pickaxe", "collect_stone")
    wood_needed = (0 if achievements["place_table"] else 2) + (0 if inventory["wood_pickaxe"] else 1)
    lines = [
        f"Goal: {CASES['crafter']['goal']}",
        "Goal progress: " + "; ".join(f"{key}={'DONE' if achievements[key] else 'pending'}" for key in goals),
        f"Turn {observation['turn']}. Position ({x}, {y}); +x east/right, +y south/down.",
        f"Facing {facing}. The adjacent tile in front is {observation['front_tile']}.",
        "Inventory: " + ", ".join(f"{key}={value}" for key, value in inventory.items()
                                 if value or key in ("health", "food", "drink", "energy",
                                                    "wood", "wood_pickaxe", "stone")),
        f"Remaining construction costs {wood_needed} wood; carrying {inventory['wood']}; "
        f"wood shortfall={max(0, wood_needed - inventory['wood'])}.",
        "Adjacent tiles: " + "; ".join(
            f"{name}={grid[3 + dy][4 + dx]}" for (dx, dy), name in DIRECTIONS.items()),
        "Current 9-column × 7-row view (north at top, player @ in center):",
        *(" ".join(SYMBOLS.get(tile, "?") for tile in row) for row in grid),
        "Legend: " + ", ".join(f"{symbol}={tile}" for tile, symbol in SYMBOLS.items()
                              if any(tile in row for row in grid)),
        "Observed resource locations (offsets relative to you; older sightings may have changed):",
    ]
    if "adjacent_visits" in observation:
        lines.insert(3, f"Arrivals at this position: {observation['position_visits']}. "
                     "Arrivals at adjacent positions: " + ", ".join(
                         f"{direction}={count}" for direction, count in observation["adjacent_visits"].items()))
    for kind in ("tree", "table", "stone", "water"):
        resources = sorted(
            (item for item in observation["known_resources"] if item["tile"] == kind),
            key=lambda item: (abs(item["position_xy"][0] - x) + abs(item["position_xy"][1] - y),
                              item["position_xy"]),
        )[:6]
        descriptions = []
        for resource in resources:
            rx, ry = resource["position_xy"]
            seen = ("visible now" if resource["last_seen_turn"] == observation["turn"]
                    else f"last seen turn {resource['last_seen_turn']}")
            offset = []
            if rx != x:
                offset.append(f"{abs(rx-x)} tile(s) {'east' if rx > x else 'west'}")
            if ry != y:
                offset.append(f"{abs(ry-y)} tile(s) {'south' if ry > y else 'north'}")
            descriptions.append(f"{' and '.join(offset) or 'here'} ({seen})")
        lines.append(f"{kind}: " + (", ".join(descriptions) or "none observed"))
    near_table = any("table" in row[3:6] for row in grid[2:5])
    lines.extend([
        f"Table within crafting distance: {'yes' if near_table else 'no'}.",
        "Last action result: " + observation["feedback"],
    ])
    return "\n".join(lines)


def decision_request(case: str, model: str, observation: dict[str, Any],
                     actions: dict[str, str], history: list[dict]) -> dict[str, Any]:
    """Build a request without filtering candidates or selecting an action."""
    recent = [{"action": entry["action"], "feedback": entry["feedback"]} for entry in history[-8:]]
    instructions = (
        "Choose the next available action using the current view, measurements, and recent "
        "feedback. Actions execute exactly as described; the environment is paused while "
        "you choose. Do not finish before the goal is achieved."
    )
    state: Any = {"goal": CASES[case]["goal"], "observation": observation, "recent_actions": recent}
    if case == "crafter" and "position_xy" in observation:
        state = crafter_state(observation)
        state += "\nRecent turns:\n" + ("\n".join(entry["feedback"] for entry in recent) or "None.")
        instructions = CRAFTER_RULES + "\nSelect the single next action that advances the unfinished goal while staying alive."
        actions = {key: (description + " " + observation.get("action_context", {}).get(key, "")).rstrip()
                   for key, description in actions.items()}
    return {
        "model": model, "state": state,
        "questions": {"action": {"type": "choice", "instructions": instructions, "criteria": actions}},
    }
