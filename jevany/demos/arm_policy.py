"""Physical subgoals for the peg-insertion demo's primitive action policy."""
from typing import Any


def _at_target(target: list[float], current: list[float]) -> bool:
    return all(abs(round((t - c) * 100, 2)) <= .55 for t, c in zip(target, current))


class ArmHarness:
    """Track measured task progress while leaving every motor command to Jev."""

    def __init__(self) -> None:
        self.stage = "open"
        self.last_action: str | None = None
        self.anchor: list[float] | None = None

    def context(self, observation: dict[str, Any]) -> dict[str, Any]:
        tip = observation["gripper_xyz_metres"]
        peg = observation["peg_xyz_metres"]
        socket = observation["destinations_xyz_metres"]["cyan_socket"]
        holding = observation["object_between_both_fingers"]
        opened = observation["gripper_open"]
        if self.stage in ("lift", "transport", "lower") and not holding:
            self.stage = "open"
        if self.stage == "grasp" and self.last_action == "close_gripper" and not holding:
            self.stage = "open"
        for _ in range(8):
            if self.stage == "open":
                target, fingers = list(tip), "open"
                if opened:
                    self.stage = "align"
                    continue
            elif self.stage == "align":
                target, fingers = [peg[0], peg[1], .31], "open"
                if _at_target(target, tip) and opened:
                    self.stage = "descend"
                    continue
            elif self.stage == "descend":
                target, fingers = [peg[0], peg[1], .085], "open"
                if _at_target(target, tip) and opened:
                    self.stage = "grasp"
                    continue
            elif self.stage == "grasp":
                target, fingers = list(tip), "closed"
                if holding:
                    self.anchor = list(tip[:2])
                    self.stage = "lift"
                    continue
            elif self.stage == "lift":
                target, fingers = [*self.anchor, .31], "closed"
                if _at_target([.31], [tip[2]]) and holding:
                    self.stage = "transport"
                    continue
            elif self.stage == "transport":
                target, fingers = [socket[0], socket[1], .31], "closed"
                if _at_target(target, tip) and holding:
                    self.stage = "lower"
                    continue
            elif self.stage == "lower":
                target, fingers = [socket[0], socket[1], .14], "closed"
                if _at_target(target, tip) and holding:
                    self.stage = "release"
                    continue
            else:
                target, fingers = list(tip), "open"
            break
        error_cm = [round((t - c) * 100, 2) for t, c in zip(target, tip)]
        moving = [axis for axis in range(3) if abs(error_cm[axis]) > .55]
        previous_axis = (self.last_action or "").split("_")[0]
        preferred_axis = (
            "xyz".index(previous_axis)
            if previous_axis in ("x", "y", "z") and "xyz".index(previous_axis) in moving
            else max(moving, key=lambda axis: abs(error_cm[axis]), default=None)
        )
        return {
            "stage": self.stage,
            "target_xyz_cm": [round(t * 100, 2) for t in target],
            "current_xyz_cm": [round(t * 100, 2) for t in tip],
            "target_minus_current_cm": error_cm,
            "preferred_motion_axis": "XYZ"[preferred_axis] if preferred_axis is not None else None,
            "required_fingers": fingers, "fingers_now": "open" if opened else "closed",
            "holding_peg_now": holding,
            "peg_xyz_cm": [round(v * 100, 2) for v in peg],
            "stage_goal": {
                "open": "Open the fingers in place before approaching the peg.",
                "align": "Position the OPEN gripper above the peg at safe height.",
                "descend": "Lower the OPEN gripper to grasp height, keeping XY aligned with the peg.",
                "grasp": "Close the aligned fingers around the peg. Confirm current two-finger contact.",
                "lift": "Lift the HELD peg vertically to transport height. Keep the fingers closed.",
                "transport": "Move the HELD peg above the cyan socket. Keep the fingers closed.",
                "lower": "Lower the HELD peg into the cyan socket. Keep the fingers closed.",
                "release": "Open the fingers in place and release the peg.",
            }[self.stage],
        }

    def request(self, observation: dict[str, Any], actions: dict[str, str],
                history: list[dict[str, Any]], model: str) -> dict[str, Any]:
        """Build a choice request; the caller attaches the current camera image."""
        context = self.context(observation)
        return {
            "model": model,
            "state": {
                "task": "Place the green peg upright in the cyan socket, then release it.",
                "current_subgoal": context["stage_goal"], "phase": context["stage"],
                "units": "centimetres in world XYZ; +Z is up",
                "gripper_xyz": context["current_xyz_cm"],
                "desired_gripper_xyz": context["target_xyz_cm"],
                "fingers_now": context["fingers_now"],
                "required_fingers": context["required_fingers"],
                "holding_peg_now": context["holding_peg_now"],
                "peg_xyz": context["peg_xyz_cm"],
                "recent_actions": history[-3:],
                "remaining_motion_xyz_cm": context["target_minus_current_cm"],
                "preferred_motion_axis": context["preferred_motion_axis"],
                "position_tolerance_cm": .55,
            },
            "questions": {"action": {
                "type": "choice",
                "instructions": (
                    "Select ONE primitive action to complete the CURRENT subgoal. "
                    "The image shows the real scene; world-coordinate measurements are authoritative for direction. "
                    "Match the required finger state. When position is outside tolerance, reduce the remaining "
                    "position error and preserve the required finger state. "
                    "A positive remaining coordinate requires a plus move; a negative coordinate requires a minus move. "
                    "Each move changes only one coordinate by 1 or 5 cm. "
                    "Continue along preferred_motion_axis until that coordinate is within tolerance, "
                    "then let the harness select the next axis. Use 5 cm when that axis has at least "
                    "4.45 cm remaining; otherwise use 1 cm. "
                    "An absolute error of 0.55 cm or less is already aligned: leave that axis alone. "
                    "Do not reverse direction to correct an already aligned axis. "
                    "Do not repeat open or close when the fingers already have the required state. "
                    "Do not close before the grasp phase; "
                    "do not open while holding the peg unless the phase is release. "
                    "The harness updates subgoals only after measured physical conditions are met."
                ),
                "criteria": dict(actions),
            }},
        }

    def record(self, action: str) -> None:
        """Remember the executed action so a failed grasp can trigger recovery."""
        self.last_action = action
