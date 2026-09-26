"""Local browser demos; simulation and model dependencies load only on request."""
from typing import Any, Protocol


class DemoEnvironment(Protocol):
    ACTION_LOOKUP: dict[str, str]
    done: bool
    success: bool
    feedback: str
    frames: list[Any]

    def reset(self, seed: int | None = None) -> dict[str, Any]: ...
    def observe(self) -> dict[str, Any]: ...
    def get_all_actions(self) -> list[str]: ...
    def step(self, action: str) -> tuple[dict, float, bool, dict]: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...

CASES = {
    "doom": {
        "title": "Doom corridor",
        "category": "3D GAME",
        "description": "Navigate a hostile corridor with limited health and ammunition.",
        "goal": "Reach the green armor at the end of the corridor. Stay alive, aim before shooting, and conserve ammunition.",
        "package": "vizdoom",
        "extra": "demo",
        "limit": 160,
    },
    "crafter": {
        "title": "Crafter survival",
        "category": "2D GAME",
        "description": "Gather materials, build tools, and survive a changing world.",
        "goal": "Collect wood, place a crafting table, make a wood pickaxe, and collect stone while staying alive.",
        "package": "crafter",
        "extra": "demo",
        "limit": 200,
    },
    "arm": {
        "title": "Robot peg insertion",
        "category": "ROBOTICS",
        "description": "Grasp a peg, align it with the cyan socket, and release it.",
        "goal": "Lift the green peg, align it with the cyan socket, lower it, and release it upright inside the socket.",
        "package": "pybullet",
        "extra": "robotics",
        "limit": 24,
    },
}


def make_environment(case: str, seed: int = 17) -> DemoEnvironment:
    """Create a CPU environment with reset/step/render and finite named actions."""
    if case not in CASES:
        raise ValueError(f"unknown environment {case!r}; choose {', '.join(CASES)}")
    if case == "arm":
        from .arm import PegInsertion
        return PegInsertion(seed)
    if case == "crafter":
        from .games import Crafter
        return Crafter(seed)
    from .games import Doom
    return Doom(seed)
