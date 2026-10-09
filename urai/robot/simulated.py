"""Kinematic simulator of the dual PiPER-X rig: simulated arm drivers and a synthetic RGB-D scene.

The simulator runs the real planning and execution stack (the backend's IK, collision checks, retiming
and 50 Hz control loop) without hardware:

* :class:`SimulatedArmDriver` is an ideal joint position servo. Each joint moves towards the latest
  reference at most at the controller speed percentage of the firmware's 3.0 rad/s joint speed limit.
  The arm is always under CAN control, enabled and reporting fresh feedback. The gripper moves towards
  its commanded opening, stops on an object between the fingers (the measured opening is then the
  object's width) and carries that object until it opens again; a released object settles upright on
  the surface below it.
* :class:`SyntheticSceneRig` renders the scene (table, objects and both arms drawn as capsules) for the
  head camera and the two wrist cameras by ray casting, with nominal RealSense D435 colour intrinsics.

Time is wall-clock time: the simulated arms move while the caller sleeps, exactly like the real ones.
"""
from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from .arm_model import ArmModel
from .calibration import ARMS, Calibration, example_calibration_path, load_calibration
from .cameras import CameraFrame, ColorFrame
from .driver import (
    CTRL_MODE_CAN_TEXT,
    GRIPPER_RANGE_MM,
    ArmState,
    MotionDisabledError,
    check_gripper_command,
    joint_command_units,
)
from .kinematics import GRIPPER_FINGER_TIP_MM, mm_to_m_matrix
from .rig import RigConfig, build_models
from .runtime import RobotRuntime
from .scene import Box, Capsule, Cup, Cylinder, Scene, SceneObject, Table

__all__ = [
    "FIRMWARE_MAX_JOINT_SPEED_DEG_S",
    "GRIPPER_SPEED_MM_S",
    "SimulatedArmDriver",
    "SyntheticCamera",
    "SyntheticSceneRig",
    "arm_capsules",
    "build_simulated_runtime",
    "default_scene",
    "nominal_intrinsics",
]

#: Firmware joint speed limit (3.0 rad/s, the documented range of the PiPER joint speed setting).
FIRMWARE_MAX_JOINT_SPEED_DEG_S = math.degrees(3.0)
#: Nominal speed of the simulated gripper jaws (mm/s of opening).
GRIPPER_SPEED_MM_S = 100.0
#: Joints of the folded rest pose the simulated arms start in.
REST_JOINTS_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#: Finger pad samples along link6 z (m, measured from the link6 origin) and across the finger (link6 y).
_PAD_DEPTHS_M = tuple((GRIPPER_FINGER_TIP_MM - d) / 1000.0 for d in (2.0, 10.0, 20.0, 30.0))
_PAD_OFFSETS_M = (-0.008, 0.0, 0.008)
#: An opening this much wider than the held object's width releases it (mm).
_RELEASE_MARGIN_MM = 0.5
_LINK_RGB = (150, 153, 160)
_FINGER_RGB = (45, 45, 50)
_FINGER_RADIUS_M = 0.008


def nominal_intrinsics(width: int, height: int) -> np.ndarray:
    """Pinhole intrinsics close to a RealSense D435 colour stream (69 x 42 degree field of view)."""
    f = 0.711 * width
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


class SimulatedArmDriver:
    """An ideal PiPER arm: rate-limited joint position servo and a gripper that grasps scene objects.

    Args:
        name: Arm name (``"left"`` or ``"right"``).
        model: The arm's model (kinematics and base pose).
        scene: Scene whose objects the gripper can grasp.
        speed_percent: Initial controller speed percentage.
        allow_motion: When false every command raises :class:`MotionDisabledError`.
        joints_deg: Initial joints (clamped into the joint limits).
        gripper_mm: Initial gripper opening.
    """

    def __init__(self, name: str, model: ArmModel, scene: Scene, *, speed_percent: int, allow_motion: bool = True,
                 joints_deg: Sequence[float] = REST_JOINTS_DEG, gripper_mm: float = 0.0) -> None:
        self.name = name
        self.model = model
        self.scene = scene
        self.allow_motion = bool(allow_motion)
        self.gripper_range_mm = GRIPPER_RANGE_MM
        self._lower = np.array([model.kin.limits_deg[i][0] for i in range(1, 7)])
        self._upper = np.array([model.kin.limits_deg[i][1] for i in range(1, 7)])
        self._q = np.clip(np.asarray(joints_deg, dtype=np.float64), self._lower, self._upper)
        self._target = self._q.copy()
        self._rate_deg_s = _joint_rate(speed_percent)
        self._opening = float(gripper_mm)
        self._opening_target = self._opening
        self._held: tuple[str, float] | None = None
        self._stamp = time.monotonic()
        self._lock = threading.Lock()
        #: World pose of link6 (m) at the last update; read without locking by objects held in the hand.
        self.hand_pose = self._link6_world(self._q)
        self._holder: Callable[[], np.ndarray] = lambda: self.hand_pose

    # ------------------------------------------------------------------ connection

    def connect(self) -> SimulatedArmDriver:
        return self

    def close(self) -> None:
        """Nothing to release."""

    # ------------------------------------------------------------------ feedback

    def joints(self) -> np.ndarray:
        with self._lock:
            self._advance()
            return self._q.copy()

    def state(self) -> ArmState:
        with self._lock:
            self._advance()
            return ArmState(joints_deg=self._q.copy(), gripper_mm=self._opening, enabled=True,
                            ctrl_mode=CTRL_MODE_CAN_TEXT, arm_status=0, err_code=0)

    def feedback_age_s(self) -> float:
        return 0.0

    def joint_feedback_age_s(self) -> float:
        return 0.0

    def motor_enable_status(self) -> list[bool]:
        return [True] * 6

    def held_object(self) -> str | None:
        """Name of the object in the gripper, if any."""
        with self._lock:
            self._advance()
            return None if self._held is None else self._held[0]

    # ------------------------------------------------------------------ commands

    def set_joint_mode(self, speed_percent: int) -> None:
        rate = _joint_rate(speed_percent)
        self._require_motion("set joint mode")
        with self._lock:
            self._advance()
            self._rate_deg_s = rate

    def command_joints(self, q_deg: Sequence[float] | np.ndarray) -> None:
        units = joint_command_units(q_deg)
        self._require_motion("joint reference")
        with self._lock:
            self._advance()
            self._target = np.clip(np.asarray(units, dtype=np.float64) / 1000.0, self._lower, self._upper)

    def set_gripper(self, opening_mm: float, effort: int) -> None:
        opening, _ = check_gripper_command(opening_mm, effort, self.gripper_range_mm, self.name)
        self._require_motion("gripper command")
        with self._lock:
            self._advance()
            self._opening_target = opening

    def exit_teaching(self) -> None:
        self._require_motion("exit drag-teach mode")

    def take_control(self) -> None:
        self._require_motion("take control")
        with self._lock:
            self._advance()
            self._target = self._q.copy()

    def clear_motor_faults(self) -> list[int]:
        return []

    def init_gripper(self, max_opening_mm: float = GRIPPER_RANGE_MM) -> bool:
        self._require_motion("initialise gripper")
        with self._lock:
            self._advance()
            self.gripper_range_mm = float(round(float(max_opening_mm)))
            self._opening_target = self._opening
        return True

    # ------------------------------------------------------------------ simulation

    def _require_motion(self, what: str) -> None:
        if not self.allow_motion:
            raise MotionDisabledError(f"{self.name}: motion is not allowed (refused: {what})")

    def _link6_world(self, q: np.ndarray) -> np.ndarray:
        return self.model.t_world_base @ mm_to_m_matrix(self.model.kin.fk(q))

    def _advance(self) -> None:
        """Move joints and jaws from the last update to now (caller holds the lock)."""
        now = time.monotonic()
        dt = now - self._stamp
        self._stamp = now
        if dt <= 0.0:
            return
        step = self._rate_deg_s * dt
        self._q = self._q + np.clip(self._target - self._q, -step, step)
        self.hand_pose = self._link6_world(self._q)
        self._advance_gripper(GRIPPER_SPEED_MM_S * dt)

    def _advance_gripper(self, step_mm: float) -> None:
        opening, target = self._opening, self._opening_target
        if target < opening:
            new = max(opening - step_mm, target)
            if self._held is None:
                found = self.scene.width_between_fingers(self.hand_pose, opening / 1000.0, _PAD_DEPTHS_M, _PAD_OFFSETS_M)
                if found is not None and new <= found[1] * 1000.0:
                    self.scene.attach(found[0], self._holder)
                    self._held = (found[0], found[1] * 1000.0)
            if self._held is not None:
                new = max(new, self._held[1])
        elif target > opening:
            new = min(opening + step_mm, target)
            if self._held is not None and new > self._held[1] + _RELEASE_MARGIN_MM:
                self.scene.release(self._held[0])
                self._held = None
        else:
            return
        self._opening = new


def _joint_rate(speed_percent: int) -> float:
    value = int(speed_percent)
    if not 1 <= value <= 100:
        raise ValueError(f"controller speed percentage must be within 1-100, got {speed_percent}")
    return FIRMWARE_MAX_JOINT_SPEED_DEG_S * value / 100.0


@dataclasses.dataclass(frozen=True)
class SyntheticCamera:
    """Pinhole model of a synthetic camera: intrinsics ``k`` and image size."""

    k: np.ndarray
    width: int
    height: int


class SyntheticSceneRig:
    """Renders RGB-D frames of a :class:`~urai.robot.scene.Scene` for the named cameras.

    Args:
        scene: The scene.
        cameras: Camera name to pinhole model.
        bodies: Returns the capsules to draw in addition to the scene (the simulated arms).
    """

    def __init__(self, scene: Scene, cameras: Mapping[str, SyntheticCamera],
                 bodies: Callable[[], Sequence[Capsule]]) -> None:
        self.scene = scene
        self.cameras = dict(cameras)
        self._bodies = bodies

    def _camera(self, name: str) -> SyntheticCamera:
        if name not in self.cameras:
            raise ValueError(f"unknown camera {name!r}; the rig has {tuple(self.cameras)}")
        return self.cameras[name]

    def capture(self, name: str, extrinsics: Callable[[], np.ndarray]) -> CameraFrame:
        """Render camera ``name`` at the pose ``extrinsics()`` returns now."""
        camera = self._camera(name)
        t_world_cam = np.asarray(extrinsics(), dtype=np.float64)
        stamp = time.time()
        rgb, depth = self.scene.render(camera.k, t_world_cam, camera.width, camera.height, self._bodies())
        return CameraFrame(rgb=rgb, depth_m=depth, k=camera.k.copy(), t_world_cam=t_world_cam, camera=name,
                           timestamp=stamp)

    def streaming(self, name: str) -> bool:
        """Synthetic cameras have no frame pump; pictures are rendered on demand."""
        self._camera(name)
        return False

    def wait_color(self, name: str, after_seq: int | None, timeout_s: float) -> ColorFrame:
        self._camera(name)
        raise TimeoutError(f"synthetic camera {name} is not streaming; use capture()")

    def close(self) -> None:
        """Nothing to release."""


def arm_capsules(model: ArmModel, joints_deg: np.ndarray, opening_mm: float) -> list[Capsule]:
    """Capsules (world, metres) drawing one arm: its link skeleton and two fingers at ``opening_mm``."""
    caps = [Capsule(p / 1000.0, q / 1000.0, r / 1000.0, _LINK_RGB)
            for name, p, q, r in model.body_capsules(joints_deg, world=True) if not name.endswith("finger")]
    rot, shift = model.t_world_base[:3, :3], model.t_world_base[:3, 3]
    tips_left, tips_right, roots_left, roots_right = model.kin.finger_corners(joints_deg, opening_mm) / 1000.0 @ rot.T + shift
    caps.append(Capsule(roots_left, tips_left, _FINGER_RADIUS_M, _FINGER_RGB))
    caps.append(Capsule(roots_right, tips_right, _FINGER_RADIUS_M, _FINGER_RGB))
    return caps


def default_scene(models: Mapping[str, ArmModel]) -> Scene:
    """A table with two blocks, a bowl, a cup and a capped bottle in front of and between the two arms.

    The objects are placed relative to the arm bases: ``forward`` is the world x axis, ``across`` points
    from the right base to the left base, and the table top is the left arm model's table height.
    """
    left, right = models["left"].base_xy, models["right"].base_xy
    mid = (left + right) / 2.0
    across = (left - right) / np.linalg.norm(left - right)
    forward = np.array([1.0, 0.0])
    z = models["left"].table_z_mm / 1000.0
    lo, hi = np.minimum(left, right), np.maximum(left, right)
    table = Table(z=z, x_range=(float(lo[0]) - 0.25, float(hi[0]) + 0.85), y_range=(float(lo[1]) - 0.45, float(hi[1]) + 0.45))

    def at(ahead: float, side: float, yaw_deg: float = 0.0) -> np.ndarray:
        xy = mid + ahead * forward + side * across
        yaw = math.radians(yaw_deg)
        t = np.eye(4)
        t[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        t[:3, 3] = [xy[0], xy[1], z]
        return t

    objects = [
        SceneObject("red_block", (Box((0.04, 0.04, 0.04), (200, 45, 40)),), at(0.32, 0.08, 15.0)),
        SceneObject("blue_block", (Box((0.035, 0.035, 0.035), (45, 85, 200)),), at(0.40, -0.04, -20.0)),
        SceneObject("bowl", (Cup(0.075, 0.055, 0.004, 0.006, (225, 222, 210)),), at(0.30, -0.17)),
        SceneObject("cup", (Cup(0.04, 0.095, 0.003, 0.006, (60, 165, 95)),), at(0.44, 0.20)),
        SceneObject("bottle", (Cylinder(0.032, 0.16, (95, 145, 205)), Cylinder(0.016, 0.022, (235, 200, 45), z0=0.16)),
                    at(0.46, -0.22)),
    ]
    return Scene(table, objects)


def build_simulated_runtime(calibration: Calibration | None = None, rig: RigConfig | None = None,
                            scene: Scene | None = None, *, allow_motion: bool = True) -> RobotRuntime:
    """A :class:`RobotRuntime` on simulated arms and synthetic cameras.

    Args:
        calibration: Rig calibration; defaults to the bundled nominal example.
        rig: Rig constants; defaults to :class:`RigConfig()`.
        scene: Scene to grasp and render; defaults to :func:`default_scene`.
        allow_motion: State of the motion gate.
    """
    calibration = calibration or load_calibration(example_calibration_path())
    rig = rig or RigConfig()
    models = build_models(calibration, rig)
    scene = scene or default_scene(models)
    arms = {arm: SimulatedArmDriver(arm, models[arm], scene, speed_percent=rig.speed_percent, allow_motion=allow_motion)
            for arm in ARMS}

    def bodies() -> list[Capsule]:
        caps: list[Capsule] = []
        for arm in ARMS:
            state = arms[arm].state()
            caps += arm_capsules(models[arm], state.joints_deg, state.gripper_mm)
        return caps

    cameras = {
        "head": SyntheticCamera(nominal_intrinsics(rig.head_width, rig.head_height), rig.head_width, rig.head_height),
        **{f"hand_{arm}": SyntheticCamera(nominal_intrinsics(rig.wrist_width, rig.wrist_height), rig.wrist_width,
                                          rig.wrist_height) for arm in ARMS},
    }
    return RobotRuntime(name="Simulated PiPER-X dual arm", arms=arms, models=models,
                        cameras=SyntheticSceneRig(scene, cameras, bodies), calibration=calibration,
                        motion_allowed=allow_motion, open_effort=rig.open_effort, grip_effort=rig.grip_effort)
