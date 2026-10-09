"""AgileX PiPER arm driver over the official ``piper_sdk`` CAN interface.

One :class:`PiperArmDriver` owns one arm on one CAN bus. It exposes the small set of operations the
runtime and the backend need (state, freshness of the feedback, joint position mode, joint references,
gripper commands, control takeover and fault recovery) and keeps every wire-level detail here:

* joint angles travel as integers in 0.001 degree, gripper openings in 0.001 mm and gripper torque
  limits in 0.001 N·m;
* ``ctrl_mode`` is reported exactly as ``piper_sdk`` prints its enum (``CAN_CTRL(0x1)``,
  ``TEACHING_MODE(0x2)``, ...), so callers can test for the substrings ``'CAN'`` and ``'TEACHING'``.

Hardware behaviours this driver relies on:

1. An arm configured as a teach-input arm (role ``0xFA``) silently ignores joint references. Taking
   control therefore first switches the arm to the motion-output role (``0xFC``); the role switch takes
   about a second, so the mode is polled with retries.
2. Never switch an arm that is under CAN control to the teach-input role: it stops broadcasting
   feedback and drops its holding torque. This driver never sends ``0xFA``.
3. The gripper does not report or respond until the end-effector parameters are written after every
   power-up (:meth:`PiperArmDriver.init_gripper`).
4. All-zero joints and all-``False`` enable flags are what the SDK returns when nothing is being
   received. Only the age of the low-speed feedback frames tells "disabled" apart from "silent bus",
   see :meth:`PiperArmDriver.feedback_age_s`.

A driver constructed with ``allow_motion=False`` never sends a command that moves the arm or changes its
configuration: every command method raises :class:`MotionDisabledError`. Connecting still sends the
parameter queries that ``piper_sdk`` issues on ``ConnectPort``; reading state is always allowed.
"""
from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "CAN_BITRATE",
    "CTRL_MODE_CAN_TEXT",
    "DEFAULT_GRIPPER_EFFORT",
    "GRIPPER_RANGE_MM",
    "MAX_GRIPPER_EFFORT",
    "MOTION_OUTPUT_ROLE",
    "ArmState",
    "MotionDisabledError",
    "PiperArmDriver",
    "check_gripper_command",
    "joint_command_units",
]

#: Bit rate of the PiPER CAN bus.
CAN_BITRATE = 1_000_000
#: Mechanical stroke of the PiPER-X gripper (mm); also the ``max_range_config`` written at start-up.
GRIPPER_RANGE_MM = 70.0
#: Gripper torque limit used by the official examples (0.001 N·m).
DEFAULT_GRIPPER_EFFORT = 1000
#: Upper bound of the gripper torque limit accepted by the firmware (0.001 N·m).
MAX_GRIPPER_EFFORT = 5000
#: Linkage role "motion-output arm": the arm accepts joint and pose references.
MOTION_OUTPUT_ROLE = 0xFC
#: How ``piper_sdk`` prints the controller mode "CAN command control".
CTRL_MODE_CAN_TEXT = "CAN_CTRL(0x1)"

_CTRL_MODE_CAN = 0x01
_MOVE_MODE_JOINT = 0x01
_FOLLOW_NORMAL = 0x00
_TEACH_EXIT = 0x02
_ALL_MOTORS = 7
_MOTOR_ENABLE = 0x02
_GRIPPER_DISABLE_CLEAR = 0x02
_GRIPPER_ENABLE = 0x01
_CLEAR_ERROR = 0xAE
_TEACHING_PENDANT_RANGE_PERCENT = 100
_TEACHING_FRICTION = 1
_ENQUIRE_GRIPPER_PARAMS = 0x04

_TAKEOVER_TRIES = 8
_TAKEOVER_WAIT_S = 0.8
_TAKEOVER_MAX_DRIFT_DEG = 2.0
_FRESH_FEEDBACK_S = 0.5


class MotionDisabledError(RuntimeError):
    """A hardware write was requested while motion is not allowed."""


@dataclasses.dataclass(frozen=True)
class ArmState:
    """One snapshot of an arm.

    Attributes:
        joints_deg: Six joint angles in degrees.
        gripper_mm: Measured gripper opening in mm.
        enabled: Whether all six joint motors report enabled.
        ctrl_mode: Controller mode as ``piper_sdk`` prints it, e.g. ``'CAN_CTRL(0x1)'``.
        arm_status: Controller status code (0 is normal).
        err_code: Controller fault bit field (0 is no fault).
    """

    joints_deg: np.ndarray
    gripper_mm: float
    enabled: bool
    ctrl_mode: str
    arm_status: int
    err_code: int


def check_gripper_command(opening_mm: float, effort: int, range_mm: float, name: str) -> tuple[float, int]:
    """Validated ``(opening_mm, effort)`` of a gripper command.

    Raises:
        ValueError: The opening is outside ``0..range_mm`` (an over-stroke command drives the jaws into the
            mechanical stop at full torque) or the effort is outside the firmware range.
    """
    opening = float(opening_mm)
    if not 0.0 <= opening <= range_mm:
        raise ValueError(f"{name}: gripper opening {opening:.1f} mm is outside 0-{range_mm:.0f} mm")
    torque = int(effort)
    if not 0 <= torque <= MAX_GRIPPER_EFFORT:
        raise ValueError(f"{name}: gripper effort {torque} is outside 0-{MAX_GRIPPER_EFFORT}")
    return opening, torque


def joint_command_units(joints_deg: Sequence[float] | np.ndarray) -> list[int]:
    """Six joint angles in degrees as the integer 0.001 degree units ``JointCtrl`` sends.

    Raises:
        ValueError: The input is not six finite numbers.
    """
    q = np.asarray(joints_deg, dtype=np.float64)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError(f"expected six finite joint angles in degrees, got {q!r}")
    return [round(v * 1000.0) for v in q.tolist()]


class PiperArmDriver:
    """One PiPER arm on one CAN interface.

    Args:
        interface: CAN channel name (for SocketCAN the network interface, e.g. ``can_left``).
        joint_limits_deg: Joint limits ``{1: (lo, hi), ..., 6: (lo, hi)}`` in degrees; the hold target
            sent while taking control is clamped into them.
        speed_percent: Controller speed percentage (1-100) used when taking control.
        allow_motion: When false the driver is read-only and never writes to the bus.
        bustype: ``python-can`` interface type. ``'socketcan'`` (the default) verifies that the
            interface exists, is up and runs at :data:`CAN_BITRATE`; other types such as ``'slcan'`` or
            ``'virtual'`` are opened without that check.
        name: Label used in error messages.
    """

    def __init__(
        self,
        interface: str,
        *,
        joint_limits_deg: Mapping[int, tuple[float, float]],
        speed_percent: int,
        allow_motion: bool,
        bustype: str = "socketcan",
        name: str | None = None,
    ) -> None:
        self.interface = str(interface)
        self.bustype = str(bustype)
        self.name = name or self.interface
        self.speed_percent = _check_speed_percent(speed_percent)
        self.allow_motion = bool(allow_motion)
        self.gripper_range_mm = GRIPPER_RANGE_MM
        self._limits = np.array([joint_limits_deg[i] for i in range(1, 7)], dtype=np.float64)
        self._piper: Any = None

    # ------------------------------------------------------------------ connection

    def connect(self, settle_s: float = 1.2) -> PiperArmDriver:
        """Open the CAN channel, start the SDK receive thread and wait for feedback frames to arrive."""
        from piper_sdk import C_PiperInterface_V2

        piper = C_PiperInterface_V2(self.interface, judge_flag=False, can_auto_init=False)
        piper.CreateCanBus(
            self.interface,
            bustype=self.bustype,
            expected_bitrate=CAN_BITRATE,
            judge_flag=self.bustype == "socketcan",
        )
        piper.ConnectPort()
        self._piper = piper
        time.sleep(settle_s)
        return self

    def close(self) -> None:
        """Stop the receive thread and close the CAN channel."""
        if self._piper is not None:
            self._piper.DisconnectPort()
            self._piper = None

    @property
    def piper(self) -> Any:
        """The underlying ``C_PiperInterface_V2``."""
        if self._piper is None:
            raise RuntimeError(f"{self.name}: CAN interface {self.interface} is not connected")
        return self._piper

    # ------------------------------------------------------------------ feedback

    def joints(self) -> np.ndarray:
        """Measured joint angles in degrees."""
        j = self.piper.GetArmJointMsgs().joint_state
        return np.array([getattr(j, f"joint_{i}") / 1000.0 for i in range(1, 7)], dtype=np.float64)

    def motor_enable_status(self) -> list[bool]:
        """Per-motor enable flags from the low-speed driver feedback (joint 1 first)."""
        return [bool(v) for v in self.piper.GetArmEnableStatus()]

    def state(self) -> ArmState:
        """Read joints, gripper, enable flags and controller status in one go."""
        status = self.piper.GetArmStatus().arm_status
        gripper = self.piper.GetArmGripperMsgs().gripper_state
        return ArmState(
            joints_deg=self.joints(),
            gripper_mm=gripper.grippers_angle / 1000.0,
            enabled=all(self.motor_enable_status()),
            ctrl_mode=str(status.ctrl_mode),
            arm_status=int(status.arm_status),
            err_code=int(status.err_code),
        )

    def feedback_age_s(self) -> float:
        """Seconds since the last low-speed driver feedback frame; ``inf`` before the first one.

        The SDK stamps frames with the receive time of the CAN message (epoch seconds). This is the
        only way to distinguish real readings from the defaults left behind by a silent bus.
        """
        return _age_s(self.piper.GetArmLowSpdInfoMsgs().time_stamp)

    def joint_feedback_age_s(self) -> float:
        """Seconds since the last joint feedback frame; ``inf`` before the first one."""
        return _age_s(self.piper.GetArmJointMsgs().time_stamp)

    def gripper_feedback_age_s(self) -> float:
        """Seconds since the last gripper feedback frame; ``inf`` before the first one."""
        return _age_s(self.piper.GetArmGripperMsgs().time_stamp)

    # ------------------------------------------------------------------ commands

    def set_joint_mode(self, speed_percent: int) -> None:
        """CAN control, joint position mode (MOVE J, normal follow) at ``speed_percent``.

        Must be sent before streaming :meth:`command_joints`: the firmware executes joint references
        with the last mode and speed it received.
        """
        self._require_motion("set joint mode")
        self.piper.MotionCtrl_2(
            ctrl_mode=_CTRL_MODE_CAN,
            move_mode=_MOVE_MODE_JOINT,
            move_spd_rate_ctrl=_check_speed_percent(speed_percent),
            is_mit_mode=_FOLLOW_NORMAL,
        )

    def command_joints(self, q_deg: Sequence[float] | np.ndarray) -> None:
        """Send one joint position reference (degrees; 0.001 degree units on the wire)."""
        units = joint_command_units(q_deg)
        self._require_motion("joint reference")
        self.piper.JointCtrl(*units)

    def set_gripper(self, opening_mm: float, effort: int) -> None:
        """Command the gripper opening (mm) with a torque limit ``effort`` (0.001 N·m).

        Raises:
            ValueError: See :func:`check_gripper_command`.
        """
        opening, torque = check_gripper_command(opening_mm, effort, self.gripper_range_mm, self.name)
        self._require_motion("gripper command")
        self.piper.GripperCtrl(round(opening * 1000.0), torque, _GRIPPER_ENABLE, 0)

    def exit_teaching(self) -> None:
        """Leave drag-teach mode (``MotionCtrl_1(0x00, 0x00, 0x02)``: end teach recording)."""
        self._require_motion("exit drag-teach mode")
        self.piper.MotionCtrl_1(0x00, 0x00, _TEACH_EXIT)

    def take_control(self, settle_s: float = 0.5) -> None:
        """Acquire motion control while holding the current pose.

        Sequence: motion-output role, exit drag-teach, enable all motors, CAN joint position mode
        (retried until the controller leaves teaching mode), then one joint reference equal to the
        measured joints clamped into the joint limits so no stale controller target is executed.

        Raises:
            RuntimeError: The arm stays in teaching mode, or the joints drift by more than 2 degrees
                after the hold reference.
        """
        self._require_motion("take control")
        hold = self._measured_hold()
        self.piper.MasterSlaveConfig(MOTION_OUTPUT_ROLE, 0, 0, 0)
        time.sleep(0.5)
        self.piper.MotionCtrl_1(0x00, 0x00, _TEACH_EXIT)
        time.sleep(0.3)
        self.piper.EnableArm(_ALL_MOTORS, _MOTOR_ENABLE)
        time.sleep(0.3)
        for _ in range(_TAKEOVER_TRIES):
            self.set_joint_mode(self.speed_percent)
            time.sleep(_TAKEOVER_WAIT_S)
            if "TEACHING" not in str(self.piper.GetArmStatus().arm_status.ctrl_mode):
                break
        else:
            raise RuntimeError(f"{self.name}: still in teaching mode after {_TAKEOVER_TRIES} attempts")
        self.piper.JointCtrl(*joint_command_units(hold))
        time.sleep(settle_s)
        drift = float(np.abs(self.joints() - hold).max())
        if drift > _TAKEOVER_MAX_DRIFT_DEG:
            raise RuntimeError(f"{self.name}: joints drifted {drift:.2f} degrees after taking control")

    def clear_motor_faults(self, settle_s: float = 0.3) -> list[int]:
        """Re-enable motors that a protection stop disabled; no-op when all motors are enabled.

        Clears the error of each disabled joint driver, then brackets ``EnableArm`` with a joint
        reference at the measured (limit-clamped) joints so the arm cannot jump to an old target.
        ``MotionCtrl_1`` resume is deliberately not used: it disables all six motors.

        Returns:
            The joint numbers (1-6) that were disabled before the recovery.

        Raises:
            RuntimeError: Some motors remain disabled (driver fault or over-temperature).
        """
        disabled = [i + 1 for i, ok in enumerate(self.motor_enable_status()) if not ok]
        if not disabled:
            return []
        self._require_motion("clear motor faults")
        for joint in disabled:
            self.piper.JointConfig(joint_num=joint, set_zero=0, acc_param_is_effective=0, clear_err=_CLEAR_ERROR)
        time.sleep(settle_s)
        hold = joint_command_units(self._measured_hold())
        self.piper.JointCtrl(*hold)
        time.sleep(0.05)
        self.piper.EnableArm(_ALL_MOTORS, _MOTOR_ENABLE)
        time.sleep(0.15)
        self.piper.JointCtrl(*hold)
        time.sleep(settle_s)
        still = [i + 1 for i, ok in enumerate(self.motor_enable_status()) if not ok]
        if still:
            raise RuntimeError(f"{self.name}: motors {still} remain disabled after clearing their faults")
        return disabled

    def init_gripper(self, max_opening_mm: float = GRIPPER_RANGE_MM, settle_s: float = 1.5) -> bool:
        """Write the gripper parameters, then enable the gripper holding its measured opening.

        Required after every power-up: until the end-effector parameters are written the gripper sends
        no feedback and ignores commands. The parameters are read back after ``settle_s``. The
        gripper is then disabled with error clearing and enabled again, as in the official example,
        but with the measured opening as its target so that enabling does not close the jaws.

        Args:
            max_opening_mm: Gripper stroke written as ``max_range_config`` (the firmware accepts 0,
                70 or 100).
            settle_s: Wait for the parameter read-back.

        Returns:
            Whether the parameter read-back matched.

        Raises:
            RuntimeError: The gripper does not report its opening, so it cannot be enabled in place.
        """
        self._require_motion("initialise gripper")
        stroke = round(float(max_opening_mm))
        self.piper.GripperTeachingPendantParamConfig(_TEACHING_PENDANT_RANGE_PERCENT, stroke, _TEACHING_FRICTION)
        time.sleep(0.5)
        self.piper.ArmParamEnquiryAndConfig(_ENQUIRE_GRIPPER_PARAMS)
        time.sleep(settle_s)
        echo = self.piper.GetGripperTeachingPendantParamFeedback().arm_gripper_teaching_param_feedback
        matched = int(echo.max_range_config) == stroke
        age = self.gripper_feedback_age_s()
        if age > _FRESH_FEEDBACK_S:
            raise RuntimeError(
                f"{self.name}: no gripper feedback ({age:.1f} s old) after writing the gripper parameters; "
                "check the wrist connector"
            )
        self.gripper_range_mm = float(stroke)
        measured = self.piper.GetArmGripperMsgs().gripper_state.grippers_angle / 1000.0
        hold = round(min(max(measured, 0.0), self.gripper_range_mm) * 1000.0)
        self.piper.GripperCtrl(hold, DEFAULT_GRIPPER_EFFORT, _GRIPPER_DISABLE_CLEAR, 0)
        time.sleep(0.3)
        self.piper.GripperCtrl(hold, DEFAULT_GRIPPER_EFFORT, _GRIPPER_ENABLE, 0)
        time.sleep(0.3)
        return matched

    # ------------------------------------------------------------------ helpers

    def _measured_hold(self) -> np.ndarray:
        """Measured joints clamped into the joint limits.

        After a power loss the arm can sag slightly past a nominal limit; sending that reading back
        unclamped would be rejected, while the clamped target only moves the arm back to the limit.
        """
        return np.clip(self.joints(), self._limits[:, 0], self._limits[:, 1])

    def _require_motion(self, what: str) -> None:
        if not self.allow_motion:
            raise MotionDisabledError(f"{self.name}: motion is not allowed (refused: {what})")


def _age_s(stamp: float) -> float:
    """Age of an SDK receive stamp (epoch seconds); ``inf`` when nothing was received yet."""
    value = float(stamp)
    if value <= 0.0:
        return float("inf")
    return max(0.0, time.time() - value)


def _check_speed_percent(speed_percent: int) -> int:
    value = int(speed_percent)
    if not 1 <= value <= 100:
        raise ValueError(f"controller speed percentage must be within 1-100, got {speed_percent}")
    return value
