# SPDX-License-Identifier: Apache-2.0
"""PiPER-X kinematics: model data, pose conventions, FK, Jacobian, IK and seeds (real computation)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.robot import kinematics
from urai.robot.kinematics import (
    GRIPPER_FINGER_ROOT_MM,
    GRIPPER_FINGER_TIP_MM,
    JOINT_LIMITS_DEG,
    URDF_PATH,
    PiperXKinematics,
    clamp_joints,
    m_to_mm_matrix,
    matrix_to_pose,
    mm_to_m_matrix,
    pose_to_matrix,
    within_limits,
)

#: link_origins of two configurations, computed from the bundled URDF (mm).
GOLDEN_LINK_ORIGINS_MM = {
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0): [
        [0.0, 0.0, 123.0],
        [0.0, 0.0, 123.0],
        [-282.40634122469174, -8.13837810183905e-16, 161.58444422410358],
        [-12.606357180182105, 2.191393069779594e-06, 207.2661496386598],
        [61.77067308391244, 2.6335810990825677e-06, 213.7602214810374],
        [96.6380190945343, 3.7786966451786904e-06, 216.80459040998974],
        [238.59792818418697, 8.440952697361878e-06, 229.19951724549472],
    ],
    (0.0, 60.0, -60.0, 0.0, 30.0, 0.0): [
        [0.0, 0.0, 123.0],
        [0.0, 0.0, 123.0],
        [-107.7880617233685, 6.614138515900473e-06, 386.86328780245134],
        [162.0119223211412, 8.805531586493906e-06, 432.5449932170076],
        [236.38895258523578, 9.24771961579688e-06, 439.03906505938517],
        [266.5849605235861, -17.499989760581215, 441.67556639573115],
        [389.5258503189905, -88.74998572294896, 452.40988946133865],
    ],
}


@pytest.fixture(scope="module")
def kin():
    return PiperXKinematics()


def bounds(kin, margin=0.0):
    lo = np.array([kin.limits_deg[i][0] for i in range(1, 7)]) + margin
    hi = np.array([kin.limits_deg[i][1] for i in range(1, 7)]) - margin
    return lo, hi


def random_joints(kin, seed, n, margin=0.0):
    lo, hi = bounds(kin, margin)
    return np.random.default_rng(seed).uniform(lo, hi, size=(n, 6))


def is_rigid(t):
    r = t[:3, :3]
    return np.allclose(r.T @ r, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.0) and np.array_equal(t[3], [0, 0, 0, 1])


class TestModelData:
    def test_bundled_urdf_and_license_are_present(self):
        assert URDF_PATH.name == "piper_x_description.urdf" and URDF_PATH.is_file()
        assert "MIT License" in (URDF_PATH.parent / "LICENSE.agilex-urdf").read_text()

    def test_joint_limits_are_the_piper_x_limits(self, kin):
        expected = {1: (-150, 150), 2: (0, 180), 3: (-170, 0), 4: (-89, 89), 5: (-89, 89), 6: (-120, 120)}
        assert kin.limits_deg == JOINT_LIMITS_DEG
        for joint, (low, high) in expected.items():
            assert kin.limits_deg[joint] == pytest.approx((low, high), abs=1e-4)

    def test_chain_shapes(self, kin):
        assert kin.joint_origins.shape == (6, 4, 4) and kin.joint_axes.shape == (6, 3)
        np.testing.assert_allclose(np.linalg.norm(kin.joint_axes, axis=1), 1.0)

    @pytest.mark.parametrize("joints", list(GOLDEN_LINK_ORIGINS_MM))
    def test_link_origins_golden_values(self, kin, joints):
        np.testing.assert_allclose(kin.link_origins(joints), GOLDEN_LINK_ORIGINS_MM[joints], rtol=0, atol=1e-9)


class TestPoseConversion:
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            pose = np.r_[rng.uniform(-500, 500, 3), rng.uniform(-179, 179), rng.uniform(-89, 89), rng.uniform(-179, 179)]
            np.testing.assert_allclose(matrix_to_pose(pose_to_matrix(pose)), pose, atol=1e-9)

    def test_intrinsic_zyx_convention(self):
        def elementary(axis, deg):
            c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
            return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
                    "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
                    "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]
        rx, ry, rz = 12.0, -34.0, 56.0
        expected = elementary("z", rz) @ elementary("y", ry) @ elementary("x", rx)
        t = pose_to_matrix([1.0, 2.0, 3.0, rx, ry, rz])
        np.testing.assert_allclose(t[:3, :3], expected, atol=1e-12)
        np.testing.assert_array_equal(t[:3, 3], [1.0, 2.0, 3.0])

    def test_rejects_bad_shapes(self):
        with pytest.raises(ValueError, match="6 elements"):
            pose_to_matrix([0.0] * 5)
        with pytest.raises(ValueError, match="4x4"):
            matrix_to_pose(np.eye(3))

    def test_unit_conversion_copies_and_scales_translation_only(self):
        t = pose_to_matrix([100.0, -200.0, 300.0, 10.0, 20.0, 30.0])
        original = t.copy()
        m = mm_to_m_matrix(t)
        np.testing.assert_array_equal(t, original)
        np.testing.assert_array_equal(m[:3, :3], t[:3, :3])
        np.testing.assert_allclose(m[:3, 3], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(m_to_mm_matrix(m), t)


class TestForwardKinematics:
    def test_fk_is_a_rigid_transform(self, kin):
        for q in random_joints(kin, 1, 50):
            assert is_rigid(kin.fk(q))

    def test_joint1_rotates_the_whole_arm_about_base_z(self, kin):
        q = np.array([0.0, 50.0, -70.0, 20.0, -30.0, 40.0])
        turned = q.copy()
        turned[0] = 37.0
        rz = np.eye(4)
        rz[:3, :3] = Rotation.from_euler("z", 37.0, degrees=True).as_matrix()
        np.testing.assert_allclose(kin.fk(turned), rz @ kin.fk(q), atol=1e-9)

    def test_link_poses_end_in_fk(self, kin):
        for q in random_joints(kin, 2, 20):
            poses = kin.link_poses(q)
            assert poses.shape == (7, 4, 4)
            np.testing.assert_array_equal(poses[0], np.eye(4))
            np.testing.assert_array_equal(poses[6], kin.fk(q))
            np.testing.assert_array_equal(kin.link_origins(q)[:6], poses[1:, :3, 3])

    def test_fingertip_lies_on_the_tool_axis(self, kin):
        for q in random_joints(kin, 3, 20):
            t = kin.fk(q)
            np.testing.assert_allclose(kin.link_origins(q)[6], t[:3, 3] + GRIPPER_FINGER_TIP_MM * t[:3, 2], atol=1e-9)

    def test_jacobian_pose_is_fk_bit_for_bit(self, kin):
        for q in random_joints(kin, 4, 20):
            pose, _, _ = kin.fk_with_spatial_jacobian(q)
            np.testing.assert_array_equal(pose, kin.fk(q))

    def test_positional_jacobian_matches_finite_differences(self, kin):
        h = 1e-4
        for q in random_joints(kin, 5, 20, margin=1.0):
            _, jac, _ = kin.fk_with_spatial_jacobian(q)
            numeric = np.column_stack([(kin.fk(q + h * e)[:3, 3] - kin.fk(q - h * e)[:3, 3]) / (2 * h) for e in np.eye(6)])
            np.testing.assert_allclose(jac, numeric, atol=1e-5)

    def test_joint_axes_match_rotation_derivative(self, kin):
        h = 1e-5
        for q in random_joints(kin, 6, 20, margin=1.0):
            _, _, axes = kin.fk_with_spatial_jacobian(q)
            r0 = kin.fk(q)[:3, :3]
            for i, e in enumerate(np.eye(6)):
                delta = Rotation.from_matrix(kin.fk(q + h * e)[:3, :3] @ r0.T).as_rotvec() / np.radians(h)
                np.testing.assert_allclose(delta, axes[:, i], atol=1e-6)

    def test_rejects_wrong_joint_count(self, kin):
        for method in (kin.fk, kin.fk_with_spatial_jacobian, kin.link_poses, kin.link_origins):
            with pytest.raises(ValueError, match="6 elements"):
                method([0.0] * 5)


class TestFingerCorners:
    def test_geometry(self, kin):
        for q, opening in zip(random_joints(kin, 7, 20), np.linspace(0.0, 80.0, 20)):
            left_tip, right_tip, left_root, right_root = kin.finger_corners(q, opening)
            t = kin.fk(q)
            assert np.linalg.norm(left_tip - right_tip) == pytest.approx(opening, abs=1e-9)
            assert np.linalg.norm(left_root - right_root) == pytest.approx(opening, abs=1e-9)
            assert np.linalg.norm(left_tip - left_root) == pytest.approx(GRIPPER_FINGER_TIP_MM - GRIPPER_FINGER_ROOT_MM, abs=1e-9)
            np.testing.assert_allclose((left_tip + right_tip) / 2, kin.link_origins(q)[6], atol=1e-9)
            np.testing.assert_allclose(np.cross(left_tip - right_tip, t[:3, 0]), 0.0, atol=1e-9)

    def test_closed_gripper_collapses_onto_the_axis(self, kin):
        corners = kin.finger_corners(np.zeros(6), 0.0)
        np.testing.assert_allclose(corners[0], corners[1], atol=1e-12)
        np.testing.assert_allclose(corners[2], corners[3], atol=1e-12)


class TestInverseKinematics:
    def test_round_trip_from_a_nearby_seed(self, kin):
        rng = np.random.default_rng(12)
        for q in random_joints(kin, 12, 50, margin=2.0):
            target = kin.fk(q)
            sol, pos_err, rot_err = kin.ik_best_effort(target, seed_deg=q + rng.normal(scale=5.0, size=6))
            assert pos_err <= 0.5 and rot_err <= 0.1
            np.testing.assert_allclose(kin.fk(sol)[:3, 3], target[:3, 3], atol=0.5)

    def test_round_trip_from_the_seed_library(self, kin):
        for q in random_joints(kin, 11, 30, margin=2.0):
            _, pos_err, rot_err = kin.ik_best_effort(kin.fk(q))
            assert pos_err <= 0.5 and rot_err <= 0.1

    def test_pose_vector_and_matrix_targets_agree(self, kin):
        q = np.array([20.0, 60.0, -80.0, 10.0, 20.0, 5.0])
        target = kin.fk(q)
        seed = q + 4.0
        a = kin.ik_best_effort(target, seed_deg=seed)
        b = kin.ik_best_effort(matrix_to_pose(target), seed_deg=seed)
        np.testing.assert_allclose(a[0], b[0], atol=1e-6)

    def test_exact_seed_is_returned_without_restarts(self, kin):
        q = np.array([-30.0, 90.0, -60.0, 20.0, -40.0, 60.0])
        sol, pos_err, rot_err = kin.ik_best_effort(kin.fk(q), seed_deg=q, max_restarts=0)
        np.testing.assert_allclose(sol, q, atol=1e-9)
        assert pos_err < 1e-6 and rot_err < 1e-6

    def test_unreachable_target_returns_the_closest_admissible_joints(self, kin):
        target = kin.fk(np.array([0.0, 60.0, -60.0, 0.0, 30.0, 0.0]))
        target[:3, 3] = [1500.0, 300.0, 200.0]
        seed = np.zeros(6)
        sol, pos_err, _ = kin.ik_best_effort(target, seed_deg=seed)
        lo, hi = bounds(kin)
        assert np.all(sol >= lo) and np.all(sol <= hi)
        assert pos_err > 500.0
        reached = kin.fk(sol)[:3, 3]
        assert np.linalg.norm(reached - target[:3, 3]) < np.linalg.norm(kin.fk(seed)[:3, 3] - target[:3, 3])

    def test_rejects_bad_target_shape(self, kin):
        with pytest.raises(ValueError, match="6-element pose or a 4x4"):
            kin.ik_best_effort(np.eye(3))


class TestSeedCandidates:
    def test_count_limits_and_spread(self, kin):
        target = kin.fk([20.0, 60.0, -80.0, 10.0, 20.0, 5.0])
        lo, hi = bounds(kin)
        for count in (1, 4, 8):
            seeds = kin.seed_candidates(target, count)
            assert len(seeds) == count
            assert all(np.all(s >= lo) and np.all(s <= hi) for s in seeds)
        four = kin.seed_candidates(target, 4)
        assert min(np.abs(a[1:5] - b[1:5]).max() for i, a in enumerate(four) for b in four[i + 1:]) >= 30.0

    def test_best_seed_converges_to_the_target(self, kin):
        q = np.array([-45.0, 80.0, -40.0, -20.0, 50.0, 70.0])
        target = kin.fk(q)
        best = kin.seed_candidates(target, 1)[0]
        _, pos_err, rot_err = kin.ik_best_effort(target, seed_deg=best, max_restarts=0)
        assert pos_err <= 0.5 and rot_err <= 0.1


class TestLimitsHelpers:
    def test_within_limits_and_margins(self):
        assert within_limits(np.zeros(6)) == []
        assert within_limits(np.zeros(6), margin_deg=0.5) == [2, 3]
        assert within_limits(np.zeros(6), margin_deg=-0.5) == []
        assert within_limits([151.0, 0.0, 1.0, 0.0, 0.0, -121.0]) == [1, 3, 6]

    def test_clamp_joints(self):
        clamped = clamp_joints([200.0, -10.0, 5.0, 0.0, 95.0, -130.0], margin_deg=1.0)
        expected = [JOINT_LIMITS_DEG[1][1] - 1.0, 1.0, -1.0, 0.0, JOINT_LIMITS_DEG[5][1] - 1.0, JOINT_LIMITS_DEG[6][0] + 1.0]
        np.testing.assert_allclose(clamped, expected)


def test_so3_log_left_jacobian_inverse_matches_finite_differences():
    rng = np.random.default_rng(9)
    for phi in [np.zeros(3), np.array([1e-6, -2e-6, 3e-6])] + list(rng.normal(scale=1.0, size=(10, 3))):
        r = Rotation.from_rotvec(phi)
        j_inv = kinematics._so3_log_left_jacobian_inverse(phi)
        h = 1e-7
        numeric = np.column_stack([
            ((Rotation.from_rotvec(h * e) * r).as_rotvec() - (Rotation.from_rotvec(-h * e) * r).as_rotvec()) / (2 * h)
            for e in np.eye(3)
        ])
        np.testing.assert_allclose(j_inv, numeric, atol=1e-6)
