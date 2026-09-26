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


def test_motion_keeps_the_previous_axis_until_it_is_aligned():
    harness = ArmHarness()
    state = observation()
    state["gripper_xyz_metres"] = [.45, -.10, .35]
    harness.record("x_plus_1cm")
    context = harness.context(state)
    assert context["preferred_motion_axis"] == "X"
    # X is now within tolerance. A larger Y error should be corrected next,
    # rather than reversing X to chase an already acceptable position.
    state["gripper_xyz_metres"][0] = .475
    context = harness.context(state)
    assert context["target_minus_current_cm"][0] == -.5
    assert context["preferred_motion_axis"] == "Y"


def test_grasp_subgoal_has_no_motion_axis():
    harness = ArmHarness()
    request = harness.request(observation(height=.085), {}, [], "fixture")
    # Alignment must first be reached at transport height.
    assert request["state"]["preferred_motion_axis"] == "Z"
    harness.context(observation())
    request = harness.request(observation(height=.085), {}, [], "fixture")
    assert request["state"]["phase"] == "grasp"
    assert request["state"]["preferred_motion_axis"] is None


def test_tolerance_boundary_advances_instead_of_stalling_without_an_axis():
    state = observation()
    state["gripper_xyz_metres"][0] = .4755
    context = ArmHarness().context(state)
    assert context["stage"] == "descend"
    assert context["target_minus_current_cm"][0] == -.55
    assert context["preferred_motion_axis"] == "Z"
