"""Franka peg insertion, adapted from JevAny's recorded PyBullet environment."""
import math
import random
from typing import Any

import numpy as np
import pybullet
import pybullet_data
from pybullet_utils.bullet_client import BulletClient

from .arm_policy import ArmHarness


class PegInsertion:
    """Move joint motors and evaluate contact, insertion depth, and release."""

    DELTAS = {
        f"{axis}_{direction}_{centimetres}cm": (index, sign * centimetres / 100)
        for index, axis in enumerate("xyz")
        for direction, sign in (("plus", 1), ("minus", -1))
        for centimetres in (5, 1)
    }
    ACTION_LOOKUP = {
        action: (
            f"Move the gripper {abs(delta) * 100:g} cm along world "
            f"{'+' if delta > 0 else '-'}{'XYZ'[axis]}; keep the other two "
            "coordinates and finger opening unchanged."
        )
        for action, (axis, delta) in DELTAS.items()
    } | {
        "close_gripper": "Close both fingers in place. No translation or automatic alignment.",
        "open_gripper": "Open both fingers in place. No translation or automatic retraction.",
    }
    LOWER = np.array([.20, -.40, .045])
    UPPER = np.array([.75, .40, .50])
    LIMIT = 120

    def __init__(self, seed: int = 17):
        self.p = None
        self.reset(seed)

    def _box(self, half, position, color, mass=0):
        p = self.p
        shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        body = p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=shape,
                                 baseVisualShapeIndex=visual, basePosition=position)
        p.changeDynamics(body, -1, lateralFriction=1.0, restitution=0,
                         spinningFriction=.01, rollingFriction=.001)
        return body

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self.close()
        self.p = BulletClient(connection_mode=pybullet.DIRECT)
        p = self.p
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        self._box([.66, .49, .035], [.35, 0, -.035], [.13, .18, .24, 1])
        self._box([30, 30, .025], [.3, 0, -.105], [.055, .075, .105, 1])
        self.robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        for joint, angle in enumerate([0, -.4, 0, -2.2, 0, 1.9, math.pi / 4]):
            p.resetJointState(self.robot, joint, angle)
        for joint in (9, 10):
            p.resetJointState(self.robot, joint, .04)
            p.changeDynamics(self.robot, joint, lateralFriction=3.0, spinningFriction=.1)
        for link in range(-1, p.getNumJoints(self.robot)):
            if link not in (9, 10):
                p.changeVisualShape(self.robot, link, rgbaColor=[.85, .90, .96, 1],
                                    specularColor=[.6, .6, .6])
        self.orientation = p.getQuaternionFromEuler([math.pi, 0, 0])
        self.finger_target = .04
        shift = random.Random(17 if seed is None else seed).uniform(-.004, .004)
        visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.017, length=.11,
                                    rgbaColor=[.12, .87, .65, 1])
        shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=.017, height=.11)
        self.peg = p.createMultiBody(baseMass=.06, baseCollisionShapeIndex=shape,
                                    baseVisualShapeIndex=visual, basePosition=[.47 + shift, -.19, .059])
        p.changeDynamics(self.peg, -1, lateralFriction=1.3, spinningFriction=.01, restitution=0)
        self.destinations = {"cyan_socket": [.61, .17, .14], "red_socket": [.39, .23, .14]}
        for name, (x, y, _) in self.destinations.items():
            color = [.17, .77, .87, 1] if name == "cyan_socket" else [.98, .35, .29, 1]
            for half, pos in [
                ([.025, .077, .035], [x - .05, y, .035]),
                ([.025, .077, .035], [x + .05, y, .035]),
                ([.025, .025, .035], [x, y - .05, .035]),
                ([.025, .025, .035], [x, y + .05, .035]),
            ]:
                self._box(half, pos, color)
        self.frames, self.ticks = [], 0
        self.recording, self.ever_held = False, False
        self.steps, self.done, self.success = 0, False, False
        self.harness = ArmHarness()
        self.feedback = "Ready. Close the fingers only after aligning at grasp height."
        self._move([.36, -.02, .4], 240)
        self._hold(120)
        self.recording = True
        return self.observe()

    def _position(self):
        return np.asarray(self.p.getBasePositionAndOrientation(self.peg)[0])

    def _tip(self):
        return np.asarray(self.p.getLinkState(self.robot, 11)[0])

    def _held(self) -> bool:
        contacts = self.p.getContactPoints(self.robot, self.peg)
        return all(any(c[3] == finger and c[9] > .05 for c in contacts) for finger in (9, 10))

    def _command(self, position):
        p = self.p
        joints = p.calculateInverseKinematics(
            self.robot, 11, position, self.orientation, maxNumIterations=100,
            residualThreshold=1e-5)
        p.setJointMotorControlArray(self.robot, list(range(7)), p.POSITION_CONTROL,
                                   targetPositions=joints[:7], forces=[200] * 7,
                                   positionGains=[.20] * 7)
        for joint in (9, 10):
            p.setJointMotorControl2(self.robot, joint, p.POSITION_CONTROL,
                                   targetPosition=self.finger_target, force=65)

    def _tick(self):
        self.p.stepSimulation()
        self.ticks += 1
        if self.ticks % 4 == 0:
            self.ever_held = self.ever_held or self._held()
        if self.recording and self.ticks % 90 == 0:
            self.frames.append(self.render())

    def _move(self, target, steps=216):
        start = self._tip()
        target = np.asarray(target)
        for index in range(steps):
            t = (index + 1) / steps
            self._command(start + (target - start) * (t * t * (3 - 2 * t)))
            self._tick()
        self._hold(36)

    def _hold(self, steps):
        target = self._tip()
        for _ in range(steps):
            self._command(target)
            self._tick()

    def checks(self) -> dict[str, bool]:
        p = self.p
        pos = self._position()
        rotation = p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self.peg)[1])
        return {
            "peg_grasped": bool(self.ever_held),
            "centered_in_cyan_socket": bool(np.linalg.norm(pos[:2] - self.destinations["cyan_socket"][:2]) < .017),
            "seated_depth": bool(.05 < pos[2] < .073),
            "upright": bool(rotation[8] > .97),
            "released": bool(p.getJointState(self.robot, 9)[0] > .030),
            "settled": bool(np.linalg.norm(p.getBaseVelocity(self.peg)[0]) < .05),
        }

    def observe(self) -> dict[str, Any]:
        tip, pos = self._tip(), self._position()
        return {
            "gripper_xyz_metres": tip.round(4).tolist(),
            "peg_xyz_metres": pos.round(4).tolist(),
            "destinations_xyz_metres": self.destinations,
            "gripper_open": bool(self.p.getJointState(self.robot, 9)[0] > .030),
            "object_between_both_fingers": self._held(),
            "aligned_above_peg": bool(np.linalg.norm(tip[:2] - pos[:2]) < .035),
            "gripper_height_above_peg_metres": round(float(tip[2] - pos[2]), 4),
            "checks": self.checks(), "feedback": self.feedback,
            "cartesian_controls": {
                "coordinate_frame": "world XYZ in metres; +Z is up, -Z is down",
                "translation_steps_metres": [.05, .01],
                "workspace_lower_xyz_metres": self.LOWER.tolist(),
                "workspace_upper_xyz_metres": self.UPPER.tolist(),
                "remaining_decisions": self.LIMIT - self.steps,
            },
        }

    def decision_request(self, model: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Give Jev a measured subgoal while retaining all primitive choices."""
        return self.harness.request(self.observe(), self.ACTION_LOOKUP, history, model)

    def get_all_actions(self) -> list[str]:
        return [] if self.done else list(self.ACTION_LOOKUP)

    def step(self, action: str) -> tuple[dict, float, bool, dict]:
        if action not in self.get_all_actions():
            raise ValueError(f"unavailable arm action: {action!r}")
        self.frames = []
        self.feedback = f"{action} executed."
        if action in self.DELTAS:
            axis, distance = self.DELTAS[action]
            target = self._tip().copy()
            target[axis] += distance
            bounded = np.clip(target, self.LOWER, self.UPPER)
            if not np.allclose(target, bounded):
                self.feedback = f"{action}: motion clipped at the workspace boundary."
            self._move(bounded, 120)
        else:
            self.finger_target = 0 if action == "close_gripper" else .04
            self._hold(180)
            self.feedback = (
                "Both fingers contact the peg." if self._held()
                else "Fingers closed without a two-finger grasp." if self.finger_target == 0
                else "Fingers opened in place."
            )
        self._hold(48)
        self.frames.append(self.render())
        self.steps += 1
        self.success = all(self.checks().values())
        self.done = bool(self.success or self.steps >= self.LIMIT)
        if self.done:
            self.feedback += (
                " Peg seated upright in cyan socket and released."
                if self.success else " Decision limit reached; insertion checks failed."
            )
        self.harness.record(action)
        return self.observe(), float(self.success), self.done, {"success": self.success}

    def render(self):
        p = self.p
        view = p.computeViewMatrix([1.18, -1.24, 1.12], [.42, .015, .15], [0, 0, 1])
        projection = p.computeProjectionMatrixFOV(43, 640 / 432, .02, 100)
        rgba = p.getCameraImage(
            640, 432, view, projection, renderer=p.ER_TINY_RENDERER, shadow=1,
            lightDirection=[-3, -4, 9], lightColor=[1, .97, .93],
            lightAmbientCoeff=.40, lightDiffuseCoeff=.77, lightSpecularCoeff=.32)[2]
        return np.asarray(rgba, dtype=np.uint8).reshape(432, 640, 4)[:, :, :3].copy()

    def close(self) -> None:
        if self.p is not None:
            self.p.disconnect()
            self.p = None
