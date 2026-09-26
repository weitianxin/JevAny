"""Subgoal requests must reflect current contact and preserve model choice."""
from jevany.demos.arm_policy import ArmHarness


def observation(*, height=.31, holding=False, opened=True):
    return {
        "gripper_xyz_metres": [.47, -.19, height],
        "peg_xyz_metres": [.47, -.19, .055],
        "destinations_xyz_metres": {"cyan_socket": [.61, .17, .14]},
        "object_between_both_fingers": holding,
        "gripper_open": opened,
        "checks": {"peg_grasped": True},
    }


def test_request_uses_signed_centimetres_and_preserves_all_candidates():
    harness = ArmHarness()
    actions = {"down": "Move down", "up": "Move up", "close": "Close fingers"}
    request = harness.request(observation(), actions, [], "fixture")
    state = request["state"]
    assert state["phase"] == "descend"
    assert state["remaining_motion_xyz_cm"] == [0., 0., -22.5]
    assert state["required_fingers"] == "open"
    assert not state["holding_peg_now"]
    assert "checks" not in state
    assert request["questions"]["action"]["criteria"] == actions
    assert request["model"] == "fixture"


def test_past_contact_does_not_advance_or_preserve_a_failed_grasp():
    harness = ArmHarness()
    harness.context(observation())
    assert harness.context(observation(height=.085))["stage"] == "grasp"
    harness.record("close_gripper")
    failed = harness.context(observation(height=.085, opened=False))
    assert failed["stage"] == "open" and failed["required_fingers"] == "open"


def test_lost_contact_returns_to_recovery_before_transporting():
    harness = ArmHarness()
    harness.context(observation())
    harness.context(observation(height=.085))
    harness.record("close_gripper")
    assert harness.context(observation(height=.085, holding=True, opened=False))["stage"] == "lift"
    failed = harness.context(observation(height=.14, holding=False, opened=False))
    assert failed["stage"] == "open"
    assert not failed["holding_peg_now"]
