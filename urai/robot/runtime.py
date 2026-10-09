"""The robot runtime: two arm drivers, their calibrated models, the cameras and the motion gate.

:class:`RobotRuntime` is what the backend talks to. The same class wraps the real hardware
(:func:`build_robot_runtime`: PiPER arms over CAN and RealSense cameras) and the kinematic simulator
(:func:`urai.robot.simulated.build_simulated_runtime`).

Safety properties of :func:`build_robot_runtime`:

* Start-up never sends a motion target other than the measured pose: there is no homing.
* The motion gate is closed unless ``allow_motion=True`` is passed explicitly. With the gate closed the
  arm drivers are read-only and every hardware writer refuses.
* With the gate open, start-up takes control of each arm only after its CAN feedback is fresh, holding
  the measured joints, and enables the gripper at its measured opening.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from .arm_model import ArmModel
from .calibration import ARMS, Calibration
from .cameras import CameraFrame, CameraSpec, ColorFrame, RealSenseRig
from .driver import ArmState, MotionDisabledError, PiperArmDriver
from .rig import RigConfig, build_models

__all__ = [
    "CAMERAS",
    "FEEDBACK_FRESH_S",
    "ArmDriver",
    "ArmState",
    "CameraFrame",
    "CameraRig",
    "ColorFrame",
    "MotionDisabledError",
    "RobotRuntime",
    "build_robot_runtime",
]

#: Camera names of the rig: the fixed head camera and one wrist camera per arm.
CAMERAS = ("head", "hand_left", "hand_right")
#: Feedback older than this is "no data", not a reading.
FEEDBACK_FRESH_S = 0.5


class ArmDriver(Protocol):
    """Operations the runtime and the backend use on one arm (real or simulated)."""

    def state(self) -> ArmState: ...
    def feedback_age_s(self) -> float: ...
    def joint_feedback_age_s(self) -> float: ...
    def joints(self) -> np.ndarray: ...
    def set_joint_mode(self, speed_percent: int) -> None: ...
    def command_joints(self, q_deg: Any) -> None: ...
    def set_gripper(self, opening_mm: float, effort: int) -> None: ...
    def exit_teaching(self) -> None: ...
    def take_control(self) -> None: ...
    def clear_motor_faults(self) -> list[int]: ...
    def init_gripper(self, max_opening_mm: float) -> bool: ...
    def close(self) -> None: ...


class CameraRig(Protocol):
    """Named RGB-D cameras (real or synthetic)."""

    def capture(self, name: str, extrinsics: Any) -> CameraFrame: ...
    def streaming(self, name: str) -> bool: ...
    def wait_color(self, name: str, after_seq: int | None, timeout_s: float) -> ColorFrame: ...
    def close(self) -> None: ...


class RobotRuntime:
    """Arms, arm models, cameras and the motion gate of one dual-arm rig.

    Args:
        name: Human-readable name of the rig.
        arms: ``{"left": driver, "right": driver}``.
        models: ``{"left": ArmModel, "right": ArmModel}`` in the shared world frame.
        cameras: Camera rig with the cameras of :data:`CAMERAS`.
        calibration: Camera and base calibration of the rig.
        motion_allowed: State of the motion gate (fixed for the lifetime of the runtime).
        open_effort: Gripper torque limit for opening (0.001 N·m).
        grip_effort: Gripper torque limit for closing (0.001 N·m).
    """

    def __init__(self, *, name: str, arms: Mapping[str, ArmDriver], models: Mapping[str, ArmModel],
                 cameras: CameraRig, calibration: Calibration, motion_allowed: bool, open_effort: int,
                 grip_effort: int) -> None:
        if sorted(arms) != sorted(ARMS) or sorted(models) != sorted(ARMS):
            raise ValueError(f"expected drivers and models for {ARMS}, got {sorted(arms)} and {sorted(models)}")
        self.name = name
        self.arms = dict(arms)
        self.models = dict(models)
        self.cameras = cameras
        self.calibration = calibration
        self.open_effort = int(open_effort)
        self.grip_effort = int(grip_effort)
        self.last_gripper_command: dict[str, float | None] = dict.fromkeys(ARMS)
        self.execution_lock = threading.RLock()
        self._motion_allowed = bool(motion_allowed)

    # ------------------------------------------------------------------ motion gate

    @property
    def motion_allowed(self) -> bool:
        """Whether hardware writes (motion, gripper, control takeover) are allowed."""
        return self._motion_allowed

    def require_motion(self, reason: str) -> None:
        """Raise :class:`MotionDisabledError` (a ``RuntimeError``) when motion is not allowed."""
        if not self._motion_allowed:
            raise MotionDisabledError(f"motion is not allowed on {self.name} (start with --allow-motion); refused: {reason}")

    # ------------------------------------------------------------------ state

    def joints_deg(self, arm: str) -> np.ndarray:
        """Measured joints of ``arm`` in degrees."""
        return np.asarray(self.arms[arm].joints(), dtype=np.float64)

    def gripper_mm(self, arm: str) -> float:
        """Measured gripper opening of ``arm`` in mm."""
        return float(self.arms[arm].state().gripper_mm)

    def require_fresh_feedback(self, arm: str) -> None:
        """Raise ``RuntimeError`` unless both the joint and the driver feedback of ``arm`` are fresh.

        A silent bus reads as all-zero joints; commanding a hold from such a reading would drive the arm
        to the zero pose.
        """
        driver = self.arms[arm]
        ages = {"joint": driver.joint_feedback_age_s(), "driver": driver.feedback_age_s()}
        stale = {k: v for k, v in ages.items() if not v <= FEEDBACK_FRESH_S}
        if stale:
            text = ", ".join(f"{k} feedback {v:.2f} s old" for k, v in stale.items())
            raise RuntimeError(f"{arm} arm: {text} (limit {FEEDBACK_FRESH_S} s); check power, emergency stop and CAN")

    # ------------------------------------------------------------------ cameras

    def camera_pose(self, camera: str) -> np.ndarray:
        """Camera-to-world transform (metres) of ``camera`` now; wrist cameras follow the measured joints."""
        if camera == "head":
            return self.calibration.t_world_head_camera.copy()
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; choose from {CAMERAS}")
        arm = camera.removeprefix("hand_")
        return self.calibration.t_world_wrist_camera(arm, self.models[arm].kin.fk(self.joints_deg(arm)))

    def capture(self, camera: str) -> CameraFrame:
        """One fresh RGB-D frame of ``camera`` with its pose at the time of the picture."""
        if camera not in CAMERAS:
            raise ValueError(f"unknown camera {camera!r}; choose from {CAMERAS}")
        return self.cameras.capture(camera, lambda: self.camera_pose(camera))

    # ------------------------------------------------------------------ recovery

    def take_control(self, arm: str) -> None:
        """Acquire motion control of ``arm`` holding its measured pose (requires fresh feedback)."""
        self.require_motion(f"take control of the {arm} arm")
        with self.execution_lock:
            self.require_fresh_feedback(arm)
            self.arms[arm].take_control()

    def clear_motor_faults(self, arm: str) -> list[int]:
        """Re-enable a protection-stopped ``arm``, bracketing the enable with a hold at the measured joints.

        Returns:
            The joint numbers that were disabled (empty when all motors were enabled).
        """
        self.require_motion(f"re-enable the {arm} arm")
        with self.execution_lock:
            self.require_fresh_feedback(arm)
            return self.arms[arm].clear_motor_faults()

    def start_control(self, max_opening_mm: float, speed_percent: int) -> None:
        """Bring both arms under CAN joint-position control without moving them.

        For each arm: take control holding the measured joints, re-enable motors a protection stop
        disabled, write the gripper parameters and enable the gripper at its measured opening, and
        select joint-position mode at ``speed_percent``.
        """
        self.require_motion("start-up control takeover")
        with self.execution_lock:
            for arm in ARMS:
                self.require_fresh_feedback(arm)
                driver = self.arms[arm]
                driver.take_control()
                driver.clear_motor_faults()
                driver.init_gripper(max_opening_mm)
                driver.set_joint_mode(speed_percent)

    def close(self) -> None:
        """Stop the cameras and disconnect the arms."""
        try:
            self.cameras.close()
        finally:
            for driver in self.arms.values():
                driver.close()


def camera_specs(calibration: Calibration, rig: RigConfig) -> dict[str, CameraSpec]:
    """RealSense stream configuration of the rig's three cameras, in opening order."""
    depth = {"depth_width": rig.depth_width, "depth_height": rig.depth_height}
    return {
        "head": CameraSpec(calibration.head_serial, rig.head_width, rig.head_height, stream=rig.head_stream, **depth),
        "hand_left": CameraSpec(calibration.wrist_cameras["left"].serial, rig.wrist_width, rig.wrist_height, **depth),
        "hand_right": CameraSpec(calibration.wrist_cameras["right"].serial, rig.wrist_width, rig.wrist_height, **depth),
    }


def build_robot_runtime(calibration: Calibration, rig: RigConfig, *, allow_motion: bool = False) -> RobotRuntime:
    """Connect the two PiPER arms and the RealSense cameras of a real rig.

    The arms are connected first (feedback only). With ``allow_motion`` both arms are then brought under
    control without moving (:meth:`RobotRuntime.start_control`). Finally the cameras are opened; the head
    camera runs a frame pump when ``rig.head_stream`` is set. Anything opened is closed again when a later
    step fails.
    """
    models = build_models(calibration, rig)
    interfaces = {"left": rig.left_interface, "right": rig.right_interface}
    arms: dict[str, PiperArmDriver] = {}
    cameras = RealSenseRig(camera_specs(calibration, rig))
    try:
        for arm in ARMS:
            driver = PiperArmDriver(interfaces[arm], joint_limits_deg=models[arm].kin.limits_deg,
                                    speed_percent=rig.speed_percent, allow_motion=allow_motion, name=f"{arm} arm")
            arms[arm] = driver.connect()
        runtime = RobotRuntime(name="AgileX PiPER-X dual arm", arms=arms, models=models, cameras=cameras,
                               calibration=calibration, motion_allowed=allow_motion, open_effort=rig.open_effort,
                               grip_effort=rig.grip_effort)
        if allow_motion:
            runtime.start_control(rig.max_opening_mm, rig.speed_percent)
        cameras.open()
    except BaseException:
        for driver in arms.values():
            driver.close()
        raise
    return runtime
