# SPDX-License-Identifier: Apache-2.0
"""Forward and inverse kinematics of the AgileX PiPER-X arm.

Model
-----
The chain is read from the official PiPER-X URDF bundled as ``assets/piper_x_description.urdf``
(``piper_x/urdf/piper_x_description.urdf`` of github.com/agilexrobotics/agx_arm_urdf at commit
f539fee, MIT license, see ``assets/LICENSE.agilex-urdf``). The standard PiPER URDF and the DH
table shipped with ``piper_sdk`` describe a different arm and do not fit the PiPER-X.

Frames
------
``base``
    The URDF ``base_link`` of the arm.
``link6``
    The URDF ``link6`` frame. It is the frame of the end pose reported by the arm firmware, so
    ``matrix_to_pose(fk(q))`` is directly comparable with the firmware end pose. The gripper
    closes along link6 +x and approaches along link6 +z.

Units
-----
Millimetres and degrees throughout. Firmware poses ``[x, y, z, RX, RY, RZ]`` use intrinsic ZYX
Euler angles, ``R = Rz(RZ) @ Ry(RY) @ Rx(RX)``.
"""

from __future__ import annotations

import functools
import pathlib
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

__all__ = [
    "EULER_ORDER",
    "GRIPPER_FINGER_ROOT_MM",
    "GRIPPER_FINGER_TIP_MM",
    "IK_SEED_STEP_DEG",
    "JOINT_LIMITS_DEG",
    "URDF_PATH",
    "PiperXKinematics",
    "clamp_joints",
    "m_to_mm_matrix",
    "matrix_to_pose",
    "mm_to_m_matrix",
    "pose_to_matrix",
    "within_limits",
]

#: Euler convention of firmware poses: intrinsic ZYX (upper case in scipy means intrinsic).
EULER_ORDER = "ZYX"

#: Joint sampling step (deg) of the IK seed library over joints 2-5.
IK_SEED_STEP_DEG = 8.0

#: The bundled PiPER-X URDF.
URDF_PATH = pathlib.Path(__file__).parent / "assets" / "piper_x_description.urdf"

#: Distance (mm) from the link6 origin to the fingertips along link6 +z.
GRIPPER_FINGER_TIP_MM = 142.5

#: Distance (mm) from the link6 origin to the finger roots along link6 +z.
GRIPPER_FINGER_ROOT_MM = 66.0


def pose_to_matrix(pose: np.ndarray | list[float]) -> np.ndarray:
    """Firmware pose ``[x, y, z, RX, RY, RZ]`` (mm, deg) to a 4x4 transform (translation in mm)."""
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (6,):
        raise ValueError(f"pose must have 6 elements, got shape {p.shape}")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler(EULER_ORDER, [p[5], p[4], p[3]], degrees=True).as_matrix()
    t[:3, 3] = p[:3]
    return t


def matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    """4x4 transform (translation in mm) to a firmware pose ``[x, y, z, RX, RY, RZ]`` (mm, deg)."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got shape {m.shape}")
    rz, ry, rx = Rotation.from_matrix(m[:3, :3]).as_euler(EULER_ORDER, degrees=True)
    return np.array([m[0, 3], m[1, 3], m[2, 3], rx, ry, rz], dtype=np.float64)


def mm_to_m_matrix(t_mm: np.ndarray) -> np.ndarray:
    """Copy of a homogeneous transform with its translation converted from mm to m."""
    out = np.array(t_mm, dtype=np.float64, copy=True)
    out[:3, 3] /= 1000.0
    return out


def m_to_mm_matrix(t_m: np.ndarray) -> np.ndarray:
    """Copy of a homogeneous transform with its translation converted from m to mm."""
    out = np.array(t_m, dtype=np.float64, copy=True)
    out[:3, 3] *= 1000.0
    return out


def _joint_vector(joints_deg: np.ndarray | list[float]) -> np.ndarray:
    q = np.asarray(joints_deg, dtype=np.float64)
    if q.shape != (6,):
        raise ValueError(f"joint angles must have 6 elements, got shape {q.shape}")
    return q


class PiperXKinematics:
    """Forward kinematics, spatial Jacobian and bounded numerical IK of one PiPER-X arm.

    Attributes:
        urdf_path: URDF the chain was read from.
        joint_origins: (6, 4, 4) fixed transforms parent -> joint frame of joint1..joint6, in metres.
        joint_axes: (6, 3) unit rotation axes of joint1..joint6 in their joint frames.
        limits_deg: ``{1..6: (lower, upper)}`` joint limits in degrees.
    """

    def __init__(self, urdf_path: pathlib.Path | str = URDF_PATH) -> None:
        self.urdf_path = pathlib.Path(urdf_path)
        chain = _parse_urdf_chain(self.urdf_path)
        self.joint_origins = np.stack([c[0] for c in chain])
        self.joint_axes = np.stack([c[1] for c in chain])
        self.limits_deg = {i + 1: chain[i][2] for i in range(len(chain))}

    # ------------------------------------------------------------------ forward kinematics

    def fk(self, joints_deg: np.ndarray | list[float]) -> np.ndarray:
        """link6 pose in the base frame, 4x4 with translation in mm."""
        q = _joint_vector(joints_deg)
        t = np.eye(4)
        for i in range(6):
            step = np.eye(4)
            step[:3, :3] = Rotation.from_rotvec(self.joint_axes[i] * np.deg2rad(q[i])).as_matrix()
            t = t @ self.joint_origins[i] @ step
        return m_to_mm_matrix(t)

    def fk_with_spatial_jacobian(
        self, joints_deg: np.ndarray | list[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward kinematics with the analytic positional Jacobian.

        Returns:
            ``(pose, jacobian, axes)``: ``pose`` equals :meth:`fk` bit for bit; ``jacobian`` (3, 6)
            is the derivative of the link6 origin in mm per degree of each joint; ``axes`` (3, 6)
            holds the unit joint axes in the base frame.
        """
        q = _joint_vector(joints_deg)
        transform = np.eye(4)
        joint_positions = np.empty((6, 3), dtype=np.float64)
        joint_axes = np.empty((6, 3), dtype=np.float64)
        for i in range(6):
            # Frame of joint i before it rotates; its axis is expressed in this frame.
            pre = transform @ self.joint_origins[i]
            joint_positions[i] = pre[:3, 3]
            joint_axes[i] = pre[:3, :3] @ self.joint_axes[i]
            step = np.eye(4)
            step[:3, :3] = Rotation.from_rotvec(self.joint_axes[i] * np.deg2rad(q[i])).as_matrix()
            transform = pre @ step
        end_position_m = transform[:3, 3]
        # Revolute joint: w x (p - o), converted from m/rad to mm/deg.
        position_jacobian = (
            np.cross(joint_axes, end_position_m[None, :] - joint_positions) * (1000.0 * np.deg2rad(1.0))
        ).T
        return m_to_mm_matrix(transform), position_jacobian, joint_axes.T

    def link_poses(self, joints_deg: np.ndarray | list[float]) -> np.ndarray:
        """Poses of base_link and link1..link6 in the base frame, shape (7, 4, 4), translation in mm."""
        q = _joint_vector(joints_deg)
        poses = np.empty((7, 4, 4), dtype=np.float64)
        transform = np.eye(4)
        poses[0] = m_to_mm_matrix(transform)
        for i in range(6):
            step = np.eye(4)
            step[:3, :3] = Rotation.from_rotvec(self.joint_axes[i] * np.deg2rad(q[i])).as_matrix()
            transform = transform @ self.joint_origins[i] @ step
            poses[i + 1] = m_to_mm_matrix(transform)
        return poses

    def link_origins(self, joints_deg: np.ndarray | list[float]) -> np.ndarray:
        """Origins of joint1..joint6 followed by the fingertip, shape (7, 3), mm, base frame."""
        poses = self.link_poses(joints_deg)
        points = poses[1:, :3, 3]
        tip = points[-1] + poses[6, :3, 2] * GRIPPER_FINGER_TIP_MM
        return np.vstack([points, tip])

    def finger_corners(self, joints_deg: np.ndarray | list[float], opening_mm: float) -> np.ndarray:
        """Finger corner points in the base frame, shape (4, 3), mm.

        Order: left tip, right tip, left root, right root. Each corner lies on the tool axis
        (link6 +z) at the root or tip distance, offset by half the opening along link6 x.
        """
        pose = self.fk(joints_deg)
        rotation = pose[:3, :3]
        origin = pose[:3, 3]
        approach = rotation[:, 2]
        closing = rotation[:, 0] * (0.5 * float(opening_mm))
        return np.stack(
            [
                origin + approach * GRIPPER_FINGER_TIP_MM + closing,
                origin + approach * GRIPPER_FINGER_TIP_MM - closing,
                origin + approach * GRIPPER_FINGER_ROOT_MM + closing,
                origin + approach * GRIPPER_FINGER_ROOT_MM - closing,
            ]
        )

    # ------------------------------------------------------------------ inverse kinematics

    def ik_best_effort(
        self,
        target: np.ndarray | list[float],
        seed_deg: np.ndarray | list[float] | None = None,
        position_tol_mm: float = 0.5,
        rotation_tol_deg: float = 0.1,
        rotation_weight: float = 2.0,
        max_restarts: int = 6,
    ) -> tuple[np.ndarray, float, float]:
        """Joint angles within the limits whose link6 pose is closest to ``target``.

        Bounded least squares from ``seed_deg`` first, then from up to ``max_restarts`` seeds of
        the seed library. The search stops at the first start whose residuals are within
        ``position_tol_mm`` and ``rotation_tol_deg``; otherwise it returns the start with the
        smallest summed residual. The solution always respects the joint limits, so an
        unreachable target comes back as the closest admissible configuration together with
        its residuals; callers decide whether that counts as reached.

        Args:
            target: link6 pose in the base frame, either a firmware pose ``[x, y, z, RX, RY, RZ]``
                (mm, deg) or a 4x4 transform with translation in mm.
            seed_deg: first start, usually the current joints. ``None`` starts from the library.
            position_tol_mm: position residual that ends the search early.
            rotation_tol_deg: orientation residual that ends the search early.
            rotation_weight: weight of the orientation residual (deg) relative to position (mm).
            max_restarts: number of library seeds tried after ``seed_deg``.

        Returns:
            ``(joints_deg, position_error_mm, rotation_error_deg)``.
        """
        goal = np.asarray(target, dtype=np.float64)
        if goal.shape == (4, 4):
            goal_matrix = goal
        elif goal.shape == (6,):
            goal_matrix = pose_to_matrix(goal)
        else:
            raise ValueError(f"target must be a 6-element pose or a 4x4 transform, got shape {goal.shape}")

        lower = np.array([self.limits_deg[i + 1][0] for i in range(6)])
        upper = np.array([self.limits_deg[i + 1][1] for i in range(6)])
        goal_rotation = Rotation.from_matrix(goal_matrix[:3, :3])
        best = (float("inf"), float("inf"))
        best_q = np.clip(
            np.asarray(seed_deg, dtype=np.float64) if seed_deg is not None else np.zeros(6),
            np.array([self.limits_deg[i + 1][0] for i in range(6)]),
            np.array([self.limits_deg[i + 1][1] for i in range(6)]),
        )

        def residual(q: np.ndarray) -> np.ndarray:
            current, _, _ = self.fk_with_spatial_jacobian(q)
            position_error = current[:3, 3] - goal_matrix[:3, 3]
            rotation_error = np.rad2deg((Rotation.from_matrix(current[:3, :3]) * goal_rotation.inv()).as_rotvec())
            return np.concatenate([position_error, rotation_error * rotation_weight])

        def jacobian(q: np.ndarray) -> np.ndarray:
            current, position_jacobian, joint_axes = self.fk_with_spatial_jacobian(q)
            rotation_error = (Rotation.from_matrix(current[:3, :3]) * goal_rotation.inv()).as_rotvec()
            # The residual's rad -> deg factor cancels the deg -> rad factor of the joint rates.
            orientation_jacobian = _so3_log_left_jacobian_inverse(rotation_error) @ joint_axes
            return np.vstack((position_jacobian, orientation_jacobian * rotation_weight))

        def attempts() -> Iterator[np.ndarray]:
            # The seed library lookup is deferred until the given seed has failed.
            budget = max_restarts + 1
            if seed_deg is not None:
                yield np.clip(np.asarray(seed_deg, dtype=np.float64), lower, upper)
                budget -= 1
            if budget <= 0:
                return
            yield from self.seed_candidates(goal_matrix, max_restarts + 1)[:budget]

        for start in attempts():
            solution = least_squares(
                residual, start, jac=jacobian, bounds=(lower, upper), xtol=1e-12, ftol=1e-12, gtol=1e-12
            )
            achieved = self.fk(solution.x)
            position_error = float(np.linalg.norm(achieved[:3, 3] - goal_matrix[:3, 3]))
            rotation_error = float(
                np.rad2deg(
                    np.linalg.norm((Rotation.from_matrix(achieved[:3, :3]) * goal_rotation.inv()).as_rotvec())
                )
            )
            if position_error <= position_tol_mm and rotation_error <= rotation_tol_deg:
                return solution.x, position_error, rotation_error
            if position_error + rotation_error < best[0] + best[1]:
                best = (position_error, rotation_error)
                best_q = solution.x
        return best_q, best[0], best[1]

    def seed_candidates(self, target: np.ndarray | list[float], count: int = 8) -> list[np.ndarray]:
        """Up to ``count`` IK starts from the seed library, closest to ``target`` first.

        The library samples joints 2-5 with j1 = j6 = 0. Targets are matched on features that are
        invariant to a rotation about the base z axis; j1 is then set from the target azimuth and
        j6 from the signed angle between the library and target x axes about the approach axis.
        Candidates are picked greedily so that they differ in joints 2-5, which spreads them over
        the elbow and wrist branches.

        Args:
            target: link6 pose, 6-element firmware pose or 4x4 transform (mm).
            count: number of starts to return.

        Returns:
            Joint vectors (deg) clipped to the limits.
        """
        goal = np.asarray(target, dtype=np.float64)
        if goal.shape == (6,):
            goal = pose_to_matrix(goal)
        joints, features, azimuths = _seed_library(str(self.urdf_path), IK_SEED_STEP_DEG)
        goal_azimuth = float(np.degrees(np.arctan2(goal[1, 3], goal[0, 3])))
        goal_feature = _cylindrical_features(
            goal[None, :3, 3], goal[None, :3, 2], goal[None, :3, 0], np.radians([goal_azimuth])
        )[0].astype(np.float32)
        # Radius and height in mm; direction components weighted so that 0.1 equals 30 mm.
        weights = np.array([1.0, 1.0, 300.0, 300.0, 300.0, 100.0, 100.0, 100.0], dtype=np.float32)
        cost = np.abs(features - goal_feature) @ weights
        pool = int(min(max(count * 40, 200), len(cost)))
        ranked = np.argpartition(cost, pool - 1)[:pool]
        ranked = ranked[np.argsort(cost[ranked])]
        lower = np.array([self.limits_deg[i + 1][0] for i in range(6)])
        upper = np.array([self.limits_deg[i + 1][1] for i in range(6)])
        out: list[np.ndarray] = []
        chosen: list[np.ndarray] = []
        for spread in (60.0, 30.0, 0.0):
            for index in ranked:
                if len(out) >= count:
                    break
                candidate = joints[index].astype(np.float64)
                if spread > 0.0 and any(
                    float(np.abs(candidate[1:5] - taken[1:5]).max()) < spread for taken in chosen
                ):
                    continue
                chosen.append(candidate)
                seed = candidate.copy()
                seed[0] = goal_azimuth - float(azimuths[index])
                seed_x, approach_axis = features[index][5:8], features[index][2:5]
                seed[5] = np.degrees(
                    np.arctan2(
                        float(np.dot(np.cross(seed_x, goal_feature[5:8]), approach_axis)),
                        float(np.dot(seed_x, goal_feature[5:8])),
                    )
                )
                out.append(np.clip(seed, lower, upper))
            if len(out) >= count:
                break
        return out


@functools.lru_cache(maxsize=4)
def _parse_urdf_chain(urdf_path: pathlib.Path) -> tuple[tuple[np.ndarray, np.ndarray, tuple[float, float]], ...]:
    """Fixed origin (metres), axis and limits (deg) of joint1..joint6 from a URDF."""
    root = ET.parse(urdf_path).getroot()
    by_name = {j.get("name"): j for j in root.iter("joint")}
    chain = []
    for idx in range(1, 7):
        joint = by_name[f"joint{idx}"]
        origin = joint.find("origin")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy").split()]).as_matrix()
        transform[:3, 3] = [float(v) for v in origin.get("xyz").split()]
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()], dtype=np.float64)
        limit = joint.find("limit")
        bounds = (float(np.rad2deg(float(limit.get("lower")))), float(np.rad2deg(float(limit.get("upper")))))
        chain.append((transform, axis, bounds))
    return tuple(chain)


@functools.lru_cache(maxsize=2)
def _seed_library(urdf_path: str, step_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid over joints 2-5 (j1 = j6 = 0): ``(joints, cylindrical features, azimuth deg)`` as float32."""
    chain = _parse_urdf_chain(pathlib.Path(urdf_path))
    grids = [np.arange(chain[i][2][0], chain[i][2][1] + 1e-9, step_deg) for i in (1, 2, 3, 4)]
    mesh = np.meshgrid(*grids, indexing="ij")
    count = mesh[0].size
    joints = np.zeros((count, 6), dtype=np.float32)
    for slot, values in zip((1, 2, 3, 4), mesh, strict=True):
        joints[:, slot] = values.ravel()
    transform = np.tile(np.eye(4), (count, 1, 1))
    for index in range(6):
        step = np.tile(np.eye(4), (count, 1, 1))
        step[:, :3, :3] = Rotation.from_rotvec(
            np.outer(np.deg2rad(joints[:, index].astype(np.float64)), chain[index][1])
        ).as_matrix()
        transform = transform @ chain[index][0] @ step
    origins = transform[:, :3, 3] * 1000.0
    azimuth = np.arctan2(origins[:, 1], origins[:, 0])
    features = _cylindrical_features(origins, transform[:, :3, 2], transform[:, :3, 0], azimuth)
    return (
        np.ascontiguousarray(joints),
        np.ascontiguousarray(features.astype(np.float32)),
        np.ascontiguousarray(np.degrees(azimuth).astype(np.float32)),
    )


def _cylindrical_features(
    origins: np.ndarray, approaches: np.ndarray, x_axes: np.ndarray, azimuth: np.ndarray
) -> np.ndarray:
    """Pose features invariant to a rotation about the base z axis, shape (N, 8).

    ``[radius, height, approach (radial, tangential, z), x axis (radial, tangential, z)]``,
    expressed in the cylindrical basis at each point's own azimuth (radians).
    """
    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    radial = np.stack([cos_a, sin_a, np.zeros_like(cos_a)], axis=-1)
    tangent = np.stack([-sin_a, cos_a, np.zeros_like(cos_a)], axis=-1)
    return np.stack(
        [
            np.hypot(origins[:, 0], origins[:, 1]),
            origins[:, 2],
            np.einsum("ij,ij->i", approaches, radial),
            np.einsum("ij,ij->i", approaches, tangent),
            approaches[:, 2],
            np.einsum("ij,ij->i", x_axes, radial),
            np.einsum("ij,ij->i", x_axes, tangent),
            x_axes[:, 2],
        ],
        axis=-1,
    )


def _so3_log_left_jacobian_inverse(rotvec: np.ndarray) -> np.ndarray:
    """Inverse left Jacobian of the SO(3) logarithm at ``rotvec``.

    For ``R' = Exp(delta) @ R`` and ``phi = log(R)``: ``log(R') - log(R) ~= J_l(phi)^-1 @ delta``.
    A series expansion is used for small angles.
    """
    phi = np.asarray(rotvec, dtype=np.float64)
    if phi.shape != (3,):
        raise ValueError(f"rotation vector must have 3 elements, got shape {phi.shape}")
    theta = float(np.linalg.norm(phi))
    skew = np.array(
        [[0.0, -phi[2], phi[1]], [phi[2], 0.0, -phi[0]], [-phi[1], phi[0], 0.0]],
        dtype=np.float64,
    )
    theta2 = theta * theta
    if theta < 1e-4:
        coefficient = 1.0 / 12.0 + theta2 / 720.0 + theta2 * theta2 / 30240.0
    else:
        coefficient = (1.0 - 0.5 * theta / np.tan(0.5 * theta)) / theta2
    return np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)


#: Joint limits (deg) of the bundled URDF, ``{1..6: (lower, upper)}``.
JOINT_LIMITS_DEG: dict[int, tuple[float, float]] = {
    index: bounds for index, (_, _, bounds) in enumerate(_parse_urdf_chain(URDF_PATH), start=1)
}


def within_limits(joints_deg: np.ndarray | list[float], margin_deg: float = 0.0) -> list[int]:
    """Indices (1..6) of joints outside the URDF limits shrunk by ``margin_deg``; empty when all fit.

    A positive margin tightens the limits, a negative one widens them.
    """
    q = np.asarray(joints_deg, dtype=np.float64)
    out = []
    for i in range(6):
        low, high = JOINT_LIMITS_DEG[i + 1]
        if not (low + margin_deg <= q[i] <= high - margin_deg):
            out.append(i + 1)
    return out


def clamp_joints(joints_deg: np.ndarray | list[float], margin_deg: float = 0.0) -> np.ndarray:
    """Copy of the joints clamped into the URDF limits shrunk by ``margin_deg``."""
    q = np.asarray(joints_deg, dtype=np.float64)
    out = np.empty(6, dtype=np.float64)
    for i in range(6):
        low, high = JOINT_LIMITS_DEG[i + 1]
        out[i] = min(max(q[i], low + margin_deg), high - margin_deg)
    return out
