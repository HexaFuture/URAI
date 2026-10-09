# SPDX-License-Identifier: Apache-2.0
"""Constants of a dual PiPER-X rig and construction of its two arm models."""

from __future__ import annotations

from dataclasses import dataclass

from .arm_model import ArmModel
from .calibration import ARMS, Calibration
from .kinematics import PiperXKinematics

__all__ = ["RigConfig", "build_models"]


@dataclass(frozen=True)
class RigConfig:
    """Rig constants that are not part of the calibration file.

    Geometry, consumed by :func:`build_models`:
        table_z_mm: table surface height in each arm's base frame (mm).
        link_clearance_mm: minimum height of the joint1..joint5 origins above the table (mm).
        fingertip_bias_mm_left, fingertip_bias_mm_right: real minus model fingertip height per arm
            (mm); positive when the real fingertip sits higher than the model.
        fingertip_below_table_mm: depth the bias-corrected fingertip may reach below the table (mm).
        ik_position_tol_mm: TCP residual above which ``ArmModel.ik_tcp_world`` reports no solution (mm).

    Drivers, consumed by the arm and camera drivers:
        left_interface, right_interface: SocketCAN interface names of the left and right arm.
        speed_percent: controller speed percentage set when entering CAN joint-position mode.
        grip_effort, open_effort: gripper effort limits for closing and opening, in 0.001 N·m.
        max_opening_mm: gripper stroke configured when the gripper is initialised (mm).
        head_width, head_height: head camera colour resolution (pixels).
        wrist_width, wrist_height: wrist camera colour resolution (pixels).
        depth_width, depth_height: depth resolution of all cameras (pixels); depth is aligned to colour.
        head_stream: keep the head camera streaming continuously instead of capturing on demand.
    """

    table_z_mm: float = 3.5
    link_clearance_mm: float = 60.0
    fingertip_bias_mm_left: float = 0.0
    fingertip_bias_mm_right: float = 0.0
    fingertip_below_table_mm: float = 5.0
    ik_position_tol_mm: float = 2.0
    left_interface: str = "canleft"
    right_interface: str = "canright"
    speed_percent: int = 20
    grip_effort: int = 1000
    open_effort: int = 1000
    max_opening_mm: float = 70.0
    head_width: int = 1280
    head_height: int = 720
    wrist_width: int = 1280
    wrist_height: int = 720
    depth_width: int = 848
    depth_height: int = 480
    head_stream: bool = True

    def fingertip_bias_for(self, arm: str) -> float:
        """Fingertip height bias (mm) of ``arm``."""
        if arm == "left":
            return self.fingertip_bias_mm_left
        if arm == "right":
            return self.fingertip_bias_mm_right
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def build_models(calibration: Calibration, rig: RigConfig) -> dict[str, ArmModel]:
    """Arm models of the rig, ``{"left": ArmModel, "right": ArmModel}``.

    The world frame is the left arm's base frame; the right arm is placed with the calibrated
    base transform. Both models share one :class:`PiperXKinematics` instance.
    """
    kin = PiperXKinematics()
    return {
        arm: ArmModel(
            arm,
            kin,
            calibration.t_world_base(arm),
            table_z_mm=rig.table_z_mm,
            link_clearance_mm=rig.link_clearance_mm,
            fingertip_bias_mm=rig.fingertip_bias_for(arm),
            fingertip_below_table_mm=rig.fingertip_below_table_mm,
            ik_position_tol_mm=rig.ik_position_tol_mm,
        )
        for arm in ARMS
    }
