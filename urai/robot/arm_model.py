# SPDX-License-Identifier: Apache-2.0
"""Geometry of one PiPER-X arm placed in the world frame.

:class:`ArmModel` wraps :class:`~urai.robot.kinematics.PiperXKinematics` with the arm's base pose
and TCP offset, and provides the collision skeleton (capsules) and the table, joint-limit,
self-collision and inter-arm checks used to validate planned and measured configurations.

Units: world-frame transforms are in metres; kinematics, capsules and clearances are in millimetres;
angles are in degrees.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .kinematics import (
    GRIPPER_FINGER_TIP_MM,
    PiperXKinematics,
    clamp_joints,
    m_to_mm_matrix,
    mm_to_m_matrix,
    within_limits,
)

__all__ = [
    "INTER_ARM_MARGIN_MM",
    "ArmModel",
    "Capsule",
    "approach_pose_dir",
    "capsules_clearance_mm",
    "inter_arm_reason",
    "joint_limit_reason",
    "m_to_mm_matrix",
    "mm_to_m_matrix",
    "segment_distance_mm",
    "segments_distance_matrix_mm",
]

#: Tool +z pointing straight down.
TOOL_DOWN = np.array([0.0, 0.0, -1.0])

# Collision skeleton (mm). Segments: base plate (base origin to the plate top), base column
# (plate top to the joint2 origin), upper arm (joint2 to joint3), forearm (joint3 to joint5),
# wrist (joint5 to joint6, including a wrist camera), tool housing, and the two open fingers.
SELF_COLLISION_BASE_PLATE_TOP_MM = 40.0
SELF_COLLISION_BASE_PLATE_RADIUS_MM = 57.0
SELF_COLLISION_BASE_COLUMN_RADIUS_MM = 45.0
SELF_COLLISION_UPPER_ARM_RADIUS_MM = 40.0
SELF_COLLISION_TOOL_RADIUS_MM = 35.0
SELF_COLLISION_FINGER_RADIUS_MM = 10.0
#: Minimum gripper-to-own-base clearance accepted by :meth:`ArmModel.self_collision_check`.
SELF_COLLISION_MARGIN_MM = 15.0
#: Fingers are modelled fully open, as they are while approaching.
SELF_COLLISION_FINGER_OPENING_MM = 70.0
INTER_ARM_FOREARM_RADIUS_MM = 40.0
INTER_ARM_WRIST_RADIUS_MM = 45.0
#: Minimum clearance between the two arms' capsules.
INTER_ARM_MARGIN_MM = 30.0

#: An IK solution whose j1 differs from the target azimuth by more than this is a flipped solution.
FACING_TOL_DEG = 90.0
#: Targets closer than this to the base axis have no meaningful azimuth.
FACING_MIN_RADIUS_MM = 50.0
#: Forward-facing rule: |j1| bound and minimum base-frame x of joint4..joint6 and the fingertip.
FRONT_J1_LIMIT_DEG = 90.0
FRONT_MIN_X_MM = -50.0

#: Planned joints are checked against the nominal URDF limits without margin; the home pose
#: (all zeros) lies exactly on the j2 and j3 bounds.
JOINT_LIMIT_MARGIN_DEG = 0.0

#: ``(segment name, end point p, end point q, radius)``, mm.
Capsule = tuple[str, np.ndarray, np.ndarray, float]


def _tilted_approach(up: np.ndarray, outward: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Unit approach direction tilted ``tilt_deg`` from straight down towards ``outward``."""
    u = np.asarray(up, dtype=np.float64).reshape(3)
    u = u / np.linalg.norm(u)
    angle = np.radians(float(tilt_deg))
    if abs(angle) < 1e-9:
        return -u
    o = np.asarray(outward, dtype=np.float64).reshape(3)
    o = o - u * float(np.dot(o, u))
    norm = float(np.linalg.norm(o))
    if norm < 1e-6:
        raise ValueError("tilt direction is parallel to the vertical; cannot tilt the approach")
    return -u * np.cos(angle) + (o / norm) * np.sin(angle)


def approach_pose_dir(xyz_m: np.ndarray, jaw_yaw_deg: float, tilt_deg: float, dir_xy: np.ndarray) -> np.ndarray:
    """TCP pose (4x4, m) at ``xyz_m`` with a tilted approach and a horizontal closing direction.

    The tool +z starts straight down and leans ``tilt_deg`` towards the horizontal direction
    ``dir_xy`` (the wrist stays on the ``-dir_xy`` side). The closing axis (tool +x) is the
    horizontal direction at ``jaw_yaw_deg`` projected onto the plane normal to the approach.
    Raises ValueError when ``tilt_deg`` is non-zero and ``dir_xy`` has no horizontal extent.
    """
    p = np.asarray(xyz_m, dtype=np.float64)
    d = np.asarray(dir_xy, dtype=np.float64)[:2]
    outward = np.array([d[0], d[1], 0.0])
    z_axis = _tilted_approach(np.array([0.0, 0.0, 1.0]), outward, tilt_deg) if abs(tilt_deg) > 1e-9 else TOOL_DOWN.copy()
    yaw = math.radians(jaw_yaw_deg)
    x_axis = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    x_axis = x_axis - z_axis * float(x_axis @ z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    t = np.eye(4)
    t[:3, 0], t[:3, 1], t[:3, 2], t[:3, 3] = x_axis, y_axis, z_axis, p
    return t


def segment_distance_mm(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Shortest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection,
    section 5.1.9); degenerate segments (points) are allowed."""
    p1, q1, p2, q2 = (np.asarray(v, dtype=np.float64) for v in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    eps = 1e-9
    if a < eps and e < eps:
        return float(np.linalg.norm(r))
    if a < eps:
        s, t = 0.0, min(max(f / e, 0.0), 1.0)
    else:
        c = float(d1 @ r)
        if e < eps:
            t, s = 0.0, min(max(-c / a, 0.0), 1.0)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s = min(max((b * f - c * e) / den, 0.0), 1.0) if den > eps else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, min(max(-c / a, 0.0), 1.0)
            elif t > 1.0:
                t, s = 1.0, min(max((b - c) / a, 0.0), 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def segments_distance_matrix_mm(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Pairwise shortest distances (N, M) between segments ``p1[i]-q1[i]`` and ``p2[j]-q2[j]``.

    Vectorised form of :func:`segment_distance_mm`, including the point and parallel cases.
    """
    p1 = np.asarray(p1, dtype=np.float64)[:, None, :]
    q1 = np.asarray(q1, dtype=np.float64)[:, None, :]
    p2 = np.asarray(p2, dtype=np.float64)[None, :, :]
    q2 = np.asarray(q2, dtype=np.float64)[None, :, :]
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = np.sum(d1 * d1, axis=-1), np.sum(d2 * d2, axis=-1)
    f, c, b = np.sum(d2 * r, axis=-1), np.sum(d1 * r, axis=-1), np.sum(d1 * d2, axis=-1)
    eps = 1e-12
    a_ok, e_ok = a > eps, e > eps
    denom = a * e - b * b
    safe = np.where(denom > eps, denom, 1.0)
    s = np.where(denom > eps, np.clip((b * f - c * e) / safe, 0.0, 1.0), 0.0)
    t = np.where(e_ok, (b * s + f) / np.where(e_ok, e, 1.0), 0.0)
    t_cl = np.clip(t, 0.0, 1.0)
    redo = (t != t_cl) | ~e_ok
    s = np.where(redo, np.clip((b * t_cl - c) / np.where(a_ok, a, 1.0), 0.0, 1.0), s)
    s = np.where(a_ok, s, 0.0)
    t_cl = np.where(a_ok, t_cl, np.where(e_ok, np.clip(f / np.where(e_ok, e, 1.0), 0.0, 1.0), 0.0))
    c1 = p1 + d1 * s[..., None]
    c2 = p2 + d2 * t_cl[..., None]
    return np.linalg.norm(c1 - c2, axis=-1)


def _capsule_arrays(caps: Sequence[Capsule]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array([c[1] for c in caps], dtype=np.float64),
        np.array([c[2] for c in caps], dtype=np.float64),
        np.array([c[3] for c in caps], dtype=np.float64),
    )


def capsules_clearance_mm(caps_a: Sequence[Capsule], caps_b: Sequence[Capsule]) -> tuple[float, str, str]:
    """Smallest surface clearance between two capsule sets (mm, negative when they intersect).

    Returns:
        ``(clearance, name in caps_a, name in caps_b)`` of the closest pair.
    """
    pa, qa, ra = _capsule_arrays(caps_a)
    pb, qb, rb = _capsule_arrays(caps_b)
    d = segments_distance_matrix_mm(pa, qa, pb, qb) - ra[:, None] - rb[None, :]
    i, j = np.unravel_index(int(d.argmin()), d.shape)
    return float(d[i, j]), caps_a[i][0], caps_b[j][0]


@dataclass
class ArmModel:
    """One arm: kinematics, base pose in the world frame and table/clearance constants.

    Attributes:
        name: arm label used in messages (``"left"`` / ``"right"``).
        kin: kinematics of the arm.
        t_world_base: base pose in the world frame, 4x4 in metres.
        tcp_offset_mm: TCP distance from the link6 origin along link6 +z (default: fingertips).
        table_z_mm: table surface height in the base frame.
        link_clearance_mm: minimum height of the joint1..joint5 origins above the table.
        fingertip_bias_mm: real fingertip height minus model fingertip height.
        fingertip_below_table_mm: how far the bias-corrected fingertip may go below the table.
        max_reach_m: coarse horizontal reach callers may use to skip hopeless IK; not a check.
        ik_position_tol_mm: TCP position residual above which :meth:`ik_tcp_world` has no solution.
        t_base_world, t_link6_tcp, t_tcp_link6: derived transforms (base in metres, TCP in mm).
    """

    name: str
    kin: PiperXKinematics
    t_world_base: np.ndarray
    tcp_offset_mm: float = GRIPPER_FINGER_TIP_MM
    table_z_mm: float = 3.5
    link_clearance_mm: float = 60.0
    fingertip_bias_mm: float = 0.0
    fingertip_below_table_mm: float = 5.0
    max_reach_m: float = 0.85
    ik_position_tol_mm: float = 2.0

    def __post_init__(self) -> None:
        self.t_world_base = np.asarray(self.t_world_base, dtype=np.float64)
        if self.t_world_base.shape != (4, 4):
            raise ValueError("t_world_base must be 4x4")
        self.t_base_world = np.linalg.inv(self.t_world_base)
        self.t_link6_tcp = np.eye(4)
        self.t_link6_tcp[2, 3] = float(self.tcp_offset_mm)
        self.t_tcp_link6 = np.linalg.inv(self.t_link6_tcp)
        #: Why the last :meth:`ik_tcp_world` call rejected a low-residual solution, if it did.
        self.last_reject_reason: str | None = None

    @property
    def base_xy(self) -> np.ndarray:
        """Base origin in the world xy plane (m)."""
        return self.t_world_base[:2, 3]

    def fk_tcp_world(self, joints_deg: np.ndarray) -> np.ndarray:
        """TCP pose in the world frame, 4x4 in metres."""
        t_base_tcp_mm = self.kin.fk(joints_deg) @ self.t_link6_tcp
        return self.t_world_base @ mm_to_m_matrix(t_base_tcp_mm)

    def ik_tcp_world(self, t_world_tcp: np.ndarray, seed_deg: np.ndarray) -> tuple[np.ndarray | None, float, float]:
        """World TCP pose (m) to joints (deg).

        Returns ``(None, position_error_mm, rotation_error_deg)`` when the position residual exceeds
        :attr:`ik_position_tol_mm`, when the solution lies within 0.5 deg of a joint limit after
        being clamped 1 deg inside, or when it breaks the forward-facing or facing rules (the
        reason is stored in :attr:`last_reject_reason`).
        """
        self.last_reject_reason = None
        t_base_tcp_mm = m_to_mm_matrix(self.t_base_world @ np.asarray(t_world_tcp, dtype=np.float64))
        t_base_link6 = t_base_tcp_mm @ self.t_tcp_link6
        q, pos_err_mm, rot_err_deg = self.kin.ik_best_effort(t_base_link6, seed_deg=seed_deg)
        if pos_err_mm > self.ik_position_tol_mm:
            return None, float(pos_err_mm), float(rot_err_deg)
        q = clamp_joints(q, margin_deg=1.0)
        if within_limits(q, margin_deg=0.5):
            return None, float(pos_err_mm), float(rot_err_deg)
        front = self.front_violation(q)
        if front is not None:
            self.last_reject_reason = front
            return None, float(pos_err_mm), float(rot_err_deg)
        facing = self.facing_violation(q, t_base_tcp_mm[:3, 3])
        if facing is not None:
            self.last_reject_reason = facing
            return None, float(pos_err_mm), float(rot_err_deg)
        return q, float(pos_err_mm), float(rot_err_deg)

    def front_violation(self, joints_deg: np.ndarray, *, origins: np.ndarray | None = None) -> str | None:
        """Reason the arm faces backwards (|j1| beyond the limit, or joint4..fingertip behind the base
        plane), else None. The elbow (joint3) is exempt: it folds behind the base in normal poses."""
        j1 = float(joints_deg[0])
        if abs(j1) > FRONT_J1_LIMIT_DEG:
            return (f"{self.name} j1={j1:.0f} deg turns the arm past the side line (|j1| > {FRONT_J1_LIMIT_DEG:.0f}); "
                    f"the robot must keep facing forward")
        pts = self.kin.link_origins(joints_deg) if origins is None else origins
        xs = pts[3:, 0]
        if xs.min() < FRONT_MIN_X_MM:
            i = int(xs.argmin()) + 4
            return (f"{self.name} {'fingertip' if i == 7 else f'joint{i}'} x={xs.min():.0f} mm is behind the base plane "
                    f"(< {FRONT_MIN_X_MM:.0f}); the robot must keep facing forward")
        return None

    def facing_violation(self, joints_deg: np.ndarray, p_base_mm: np.ndarray) -> str | None:
        """Reason j1 faces away from the target at ``p_base_mm`` (a flipped IK branch), else None."""
        x, y = float(p_base_mm[0]), float(p_base_mm[1])
        if math.hypot(x, y) < FACING_MIN_RADIUS_MM:
            return None
        az = math.degrees(math.atan2(y, x))
        diff = (float(joints_deg[0]) - az + 180.0) % 360.0 - 180.0
        if abs(diff) <= FACING_TOL_DEG:
            return None
        return (f"{self.name} flipped IK solution: j1={float(joints_deg[0]):.0f} deg faces away from the target "
                f"(azimuth {az:.0f} deg, diff {abs(diff):.0f} > {FACING_TOL_DEG:.0f}); the arm would swing around "
                f"behind itself — target is out of normal reach")

    def table_check(self, joints_deg: np.ndarray, *, origins: np.ndarray | None = None) -> str | None:
        """Reason the arm violates the table (joint origins too low or the bias-corrected fingertip
        below the allowed depth), else None."""
        origins = self.kin.link_origins(joints_deg) if origins is None else origins
        # joint1..joint5 keep link_clearance_mm above the table; joint6 only has to stay above it.
        joints = np.asarray(origins[:-1], dtype=np.float64).reshape(-1, 3)
        floor = np.full(joints.shape[0], float(self.table_z_mm) + float(self.link_clearance_mm))
        floor[-1] = float(self.table_z_mm)
        margin = joints[:, 2] - floor
        if margin.min() < 0.0:
            j = int(margin.argmin()) + 1
            return f"{self.name} joint{j} z={origins[j - 1, 2]:.0f}mm below table clearance {self.link_clearance_mm:.0f}mm"
        tip_z = float(origins[-1, 2]) + self.fingertip_bias_mm
        if tip_z < self.table_z_mm - self.fingertip_below_table_mm:
            return f"{self.name} fingertip z={tip_z:.1f}mm (bias-corrected) is below the table"
        return None

    def floor_check(self, joints_deg: np.ndarray, *, self_collision: bool = True) -> str | None:
        """:meth:`table_check`, then :meth:`front_violation`, then (optionally)
        :meth:`self_collision_check`; the first reason found, else None."""
        origins = self.kin.link_origins(joints_deg)
        table = self.table_check(joints_deg, origins=origins)
        if table is not None:
            return table
        front = self.front_violation(joints_deg, origins=origins)
        if front is not None:
            return front
        return self.self_collision_check(joints_deg, origins=origins) if self_collision else None

    def self_collision_clearance(self, joints_deg: np.ndarray, *, origins: np.ndarray | None = None) -> tuple[float, str, str]:
        """Smallest clearance (mm, negative when intersecting) of the gripper against the arm's own
        base plate, base column and upper arm, with the segment names ``(clearance, mover, target)``.

        Here the tool segment runs from the joint6 origin to the fingertip.
        """
        q = np.asarray(joints_deg, dtype=np.float64)
        pts = self.kin.link_origins(q) if origins is None else origins
        plate_top = np.array([0.0, 0.0, SELF_COLLISION_BASE_PLATE_TOP_MM])
        corners = self.kin.finger_corners(q, SELF_COLLISION_FINGER_OPENING_MM)
        movers = [("tool body", (pts[5], pts[6]), SELF_COLLISION_TOOL_RADIUS_MM),
                  ("left finger", (corners[2], corners[0]), SELF_COLLISION_FINGER_RADIUS_MM),
                  ("right finger", (corners[3], corners[1]), SELF_COLLISION_FINGER_RADIUS_MM)]
        targets = [("base plate", (np.zeros(3), plate_top), SELF_COLLISION_BASE_PLATE_RADIUS_MM),
                   ("base column", (plate_top, pts[1]), SELF_COLLISION_BASE_COLUMN_RADIUS_MM),
                   ("upper arm", (pts[1], pts[2]), SELF_COLLISION_UPPER_ARM_RADIUS_MM)]
        return min(
            ((segment_distance_mm(*m_seg, *t_seg) - m_r - t_r, m_name, t_name)
             for m_name, m_seg, m_r in movers for t_name, t_seg, t_r in targets),
            key=lambda item: item[0],
        )

    def self_collision_check(self, joints_deg: np.ndarray, *, origins: np.ndarray | None = None) -> str | None:
        """Reason the gripper comes within :data:`SELF_COLLISION_MARGIN_MM` of its own base or upper
        arm, else None."""
        worst = self.self_collision_clearance(joints_deg, origins=origins)
        if worst[0] < SELF_COLLISION_MARGIN_MM:
            return (f"{self.name} self-collision: {worst[1]} vs {worst[2]} (clearance {worst[0]:.0f} mm < "
                    f"{SELF_COLLISION_MARGIN_MM:.0f}); target too close to the arm's own base — pick a point "
                    f"farther out or let the other arm take it")
        return None

    def body_capsules(self, joints_deg: np.ndarray, *, world: bool = True) -> list[Capsule]:
        """Capsule skeleton of the whole arm (mm): base plate, base column, upper arm, forearm, wrist,
        tool body and the two open fingers; in the world frame unless ``world=False``.

        The tool body ends at the finger roots, so the gap between the open fingers stays free.
        """
        q = np.asarray(joints_deg, dtype=np.float64)
        pts = self.kin.link_origins(q)
        plate_top = np.array([0.0, 0.0, SELF_COLLISION_BASE_PLATE_TOP_MM])
        corners = self.kin.finger_corners(q, SELF_COLLISION_FINGER_OPENING_MM)
        caps: list[Capsule] = [
            ("base plate", np.zeros(3), plate_top, SELF_COLLISION_BASE_PLATE_RADIUS_MM),
            ("base column", plate_top, pts[1], SELF_COLLISION_BASE_COLUMN_RADIUS_MM),
            ("upper arm", pts[1], pts[2], SELF_COLLISION_UPPER_ARM_RADIUS_MM),
            ("forearm", pts[2], pts[4], INTER_ARM_FOREARM_RADIUS_MM),
            ("wrist", pts[4], pts[5], INTER_ARM_WRIST_RADIUS_MM),
            ("tool body", pts[5], (corners[2] + corners[3]) / 2., SELF_COLLISION_TOOL_RADIUS_MM),
            ("left finger", corners[2], corners[0], SELF_COLLISION_FINGER_RADIUS_MM),
            ("right finger", corners[3], corners[1], SELF_COLLISION_FINGER_RADIUS_MM),
        ]
        if not world:
            return caps
        rot, t_mm = self.t_world_base[:3, :3], self.t_world_base[:3, 3] * 1000.0
        return [(n, rot @ p + t_mm, rot @ qq + t_mm, rad) for n, p, qq, rad in caps]


def joint_limit_reason(joints_deg: np.ndarray) -> str | None:
    """Reason a joint vector leaves the URDF limits, else None."""
    bad = within_limits(joints_deg, margin_deg=JOINT_LIMIT_MARGIN_DEG)
    if bad:
        q = np.asarray(joints_deg, dtype=np.float64)
        which = ", ".join(f"j{i}={q[i - 1]:.1f} deg" for i in bad)
        return f"joint limit exceeded ({which})"
    return None


def inter_arm_reason(model: ArmModel, joints_deg: np.ndarray, other: ArmModel, other_joints_deg: np.ndarray, *,
                     margin_mm: float = INTER_ARM_MARGIN_MM, other_caps: Sequence[Capsule] | None = None) -> str | None:
    """Reason ``model`` at ``joints_deg`` comes within ``margin_mm`` of ``other`` at
    ``other_joints_deg`` (world-frame capsules), else None. ``other_caps`` reuses precomputed
    capsules of the other arm."""
    caps_o = other.body_capsules(other_joints_deg) if other_caps is None else other_caps
    clearance, part, other_part = capsules_clearance_mm(model.body_capsules(joints_deg), caps_o)
    if clearance >= margin_mm:
        return None
    return (f"{model.name} would hit the {other.name} arm: {model.name} {part} vs {other.name} {other_part} "
            f"(clearance {clearance:.0f} mm < {margin_mm:.0f}); the {other.name} arm is in the way — park('{other.name}') or home() it "
            f"first, or choose a target farther from it")
