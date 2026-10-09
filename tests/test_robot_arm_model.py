# SPDX-License-Identifier: Apache-2.0
"""ArmModel geometry: transforms, IK wrapper, capsules, clearances and checks (real computation)."""

import re

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.robot.arm_model import (
    INTER_ARM_MARGIN_MM,
    SELF_COLLISION_BASE_PLATE_TOP_MM,
    SELF_COLLISION_FINGER_OPENING_MM,
    SELF_COLLISION_FINGER_RADIUS_MM,
    SELF_COLLISION_MARGIN_MM,
    SELF_COLLISION_TOOL_RADIUS_MM,
    ArmModel,
    approach_pose_dir,
    capsules_clearance_mm,
    inter_arm_reason,
    joint_limit_reason,
    mm_to_m_matrix,
    segment_distance_mm,
    segments_distance_matrix_mm,
)
from urai.robot.kinematics import GRIPPER_FINGER_TIP_MM, PiperXKinematics

CAPSULE_NAMES = ["base plate", "base column", "upper arm", "forearm", "wrist", "tool body", "left finger", "right finger"]


@pytest.fixture(scope="module")
def kin():
    return PiperXKinematics()


@pytest.fixture(scope="module")
def left(kin):
    return ArmModel("left", kin, np.eye(4))


@pytest.fixture(scope="module")
def right(kin):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("z", 23, degrees=True).as_matrix()
    t[:3, 3] = [0.02, -0.59, 0.01]
    return ArmModel("right", kin, t)


def transform(model, p):
    return model.t_world_base[:3, :3] @ p + model.t_world_base[:3, 3] * 1000.0


class TestTransforms:
    def test_defaults(self, left):
        assert (left.tcp_offset_mm, left.table_z_mm, left.link_clearance_mm, left.fingertip_bias_mm,
                left.fingertip_below_table_mm, left.max_reach_m, left.ik_position_tol_mm) == (
            GRIPPER_FINGER_TIP_MM, 3.5, 60.0, 0.0, 5.0, 0.85, 2.0)

    def test_derived_transforms(self, right):
        np.testing.assert_allclose(right.t_base_world @ right.t_world_base, np.eye(4), atol=1e-12)
        np.testing.assert_allclose(right.t_link6_tcp @ right.t_tcp_link6, np.eye(4), atol=1e-12)
        assert right.t_link6_tcp[2, 3] == GRIPPER_FINGER_TIP_MM
        np.testing.assert_array_equal(right.base_xy, [0.02, -0.59])

    def test_rejects_a_non_4x4_base(self, kin):
        with pytest.raises(ValueError, match="4x4"):
            ArmModel("left", kin, np.eye(3))

    def test_fk_tcp_world_composes_base_link6_and_tcp(self, kin, right):
        q = np.array([10.0, 70.0, -50.0, 15.0, -25.0, 30.0])
        offset = np.eye(4)
        offset[2, 3] = GRIPPER_FINGER_TIP_MM
        np.testing.assert_allclose(right.fk_tcp_world(q), right.t_world_base @ mm_to_m_matrix(kin.fk(q) @ offset), atol=1e-15)
        np.testing.assert_allclose(right.fk_tcp_world(q)[:3, 3] * 1000.0, transform(right, kin.link_origins(q)[6]), atol=1e-9)

    def test_ik_tcp_world_round_trip(self, right):
        rng = np.random.default_rng(3)
        solved = 0
        for q in rng.uniform([-60, 40, -150, -60, -60, -90], [60, 140, -40, 60, 60, 90], size=(40, 6)):
            if right.front_violation(q) is not None:
                continue
            target = right.fk_tcp_world(q)
            sol, pos_err, _ = right.ik_tcp_world(target, q + rng.normal(scale=3.0, size=6))
            assert sol is not None and pos_err <= right.ik_position_tol_mm
            np.testing.assert_allclose(right.fk_tcp_world(sol)[:3, 3], target[:3, 3], atol=2e-3)
            solved += 1
        assert solved >= 20

    def test_ik_tcp_world_rejects_out_of_reach(self, left):
        target = approach_pose_dir(np.array([1.5, 0.0, 0.1]), 0.0, 0.0, np.zeros(2))
        sol, pos_err, _ = left.ik_tcp_world(target, np.zeros(6))
        assert sol is None and pos_err > left.ik_position_tol_mm and left.last_reject_reason is None

    def test_ik_tcp_world_rejects_turning_past_the_side_line(self, left):
        q = np.array([100.0, 60.0, -60.0, 0.0, 30.0, 0.0])
        sol, pos_err, _ = left.ik_tcp_world(left.fk_tcp_world(q), q)
        assert sol is None and pos_err < 1e-6
        assert left.last_reject_reason == ("left j1=100 deg turns the arm past the side line (|j1| > 90); "
                                           "the robot must keep facing forward")


class TestApproachPose:
    def test_untilted_pose_points_straight_down(self):
        t = approach_pose_dir(np.array([0.3, -0.1, 0.05]), 30.0, 0.0, np.zeros(2))
        np.testing.assert_allclose(t[:3, 2], [0, 0, -1])
        np.testing.assert_allclose(t[:3, 0], [np.cos(np.radians(30)), np.sin(np.radians(30)), 0], atol=1e-12)
        np.testing.assert_allclose(np.cross(t[:3, 0], t[:3, 1]), t[:3, 2], atol=1e-12)
        np.testing.assert_array_equal(t[:3, 3], [0.3, -0.1, 0.05])

    @pytest.mark.parametrize("tilt", [10.0, 30.0, -25.0])
    def test_tilt_leans_the_tool_towards_the_direction(self, tilt):
        direction = np.array([0.6, 0.8])
        t = approach_pose_dir(np.zeros(3), 0.0, tilt, direction)
        z = t[:3, 2]
        assert np.degrees(np.arccos(-z[2])) == pytest.approx(abs(tilt))
        horizontal = z[:2] / np.linalg.norm(z[:2])
        np.testing.assert_allclose(horizontal, np.sign(tilt) * direction, atol=1e-12)
        r = t[:3, :3]
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)

    def test_tilt_without_a_direction_is_rejected(self):
        with pytest.raises(ValueError, match="cannot tilt"):
            approach_pose_dir(np.zeros(3), 0.0, 20.0, np.zeros(2))


class TestSegments:
    def test_known_distances(self):
        assert segment_distance_mm([0, 0, 0], [10, 0, 0], [0, 5, 0], [10, 5, 0]) == pytest.approx(5.0)
        assert segment_distance_mm([-5, 0, 0], [5, 0, 0], [0, -5, 3], [0, 5, 3]) == pytest.approx(3.0)
        assert segment_distance_mm([0, 0, 0], [0, 0, 0], [3, 4, 0], [3, 4, 0]) == pytest.approx(5.0)
        assert segment_distance_mm([0, 0, 0], [10, 0, 0], [13, 4, 0], [20, 4, 0]) == pytest.approx(5.0)

    def test_matrix_matches_scalar_including_degenerate_and_parallel(self):
        rng = np.random.default_rng(4)
        a_p = rng.uniform(-100, 100, (30, 3))
        a_q = a_p + rng.normal(scale=40, size=(30, 3))
        a_q[:5] = a_p[:5]
        b_p = rng.uniform(-100, 100, (40, 3))
        b_q = b_p + rng.normal(scale=40, size=(40, 3))
        b_q[:5] = b_p[:5]
        b_p[5:15] = a_p[5:15] + 7.0
        b_q[5:15] = b_p[5:15] + (a_q[5:15] - a_p[5:15]) * 0.5
        matrix = segments_distance_matrix_mm(a_p, a_q, b_p, b_q)
        scalar = np.array([[segment_distance_mm(a_p[i], a_q[i], b_p[j], b_q[j]) for j in range(40)] for i in range(30)])
        np.testing.assert_allclose(matrix, scalar, atol=1e-9)

    def test_capsule_clearance_subtracts_radii_and_names_the_pair(self):
        a = [("a1", np.zeros(3), np.array([10.0, 0, 0]), 2.0), ("a2", np.array([0, 50.0, 0]), np.array([10, 50.0, 0]), 1.0)]
        b = [("b1", np.array([0, 0, 20.0]), np.array([10, 0, 20.0]), 3.0)]
        assert capsules_clearance_mm(a, b) == (pytest.approx(15.0), "a1", "b1")


class TestCapsules:
    @pytest.mark.parametrize("q", [np.zeros(6), [30, 60, -40, 10, 20, 0], [-20, 80, -50, -20, 30, 15]])
    @pytest.mark.parametrize("world", [False, True])
    def test_skeleton_order_frames_and_radii(self, kin, right, q, world):
        caps = right.body_capsules(q, world=world)
        assert [c[0] for c in caps] == CAPSULE_NAMES
        local = {c[0]: c for c in right.body_capsules(q, world=False)}
        pts = kin.link_origins(q)
        corners = kin.finger_corners(q, SELF_COLLISION_FINGER_OPENING_MM)
        expected = {"base plate": (np.zeros(3), np.array([0, 0, SELF_COLLISION_BASE_PLATE_TOP_MM])),
                    "upper arm": (pts[1], pts[2]), "forearm": (pts[2], pts[4]), "wrist": (pts[4], pts[5]),
                    "tool body": (pts[5], (corners[2] + corners[3]) / 2),
                    "left finger": (corners[2], corners[0]), "right finger": (corners[3], corners[1])}
        for name, start, end, radius in caps:
            assert radius == local[name][3]
            if name in expected:
                a, b = expected[name]
                if world:
                    a, b = transform(right, a), transform(right, b)
                np.testing.assert_allclose(start, a, atol=1e-9)
                np.testing.assert_allclose(end, b, atol=1e-9)
        assert local["tool body"][3] == SELF_COLLISION_TOOL_RADIUS_MM
        assert local["left finger"][3] == local["right finger"][3] == SELF_COLLISION_FINGER_RADIUS_MM

    def test_tool_housing_ends_at_the_finger_roots_and_leaves_the_jaw_gap_open(self, kin, left):
        q = np.array([0.0, 90.0, -90.0, 0.0, 0.0, 0.0])
        corners = kin.finger_corners(q, SELF_COLLISION_FINGER_OPENING_MM)
        tool = [c for c in left.body_capsules(q, world=False) if c[0] in ("tool body", "left finger", "right finger")]
        housing_middle = (tool[0][1] + tool[0][2]) / 2
        for point in (corners[0], corners[1], housing_middle):
            assert capsules_clearance_mm(tool, [("probe", point, point, 2.0)])[0] < 0
        tip_centre = (corners[0] + corners[1]) / 2
        assert capsules_clearance_mm(tool, [("gap", tip_centre, tip_centre, 2.0)])[0] > 0


class TestSelfCollision:
    def test_clearance_is_the_closest_gripper_to_base_pair(self, kin, left):
        for q in np.random.default_rng(5).uniform([-150, 0, -170, -89, -89, -120], [150, 180, 0, 89, 89, 120], (20, 6)):
            pts = kin.link_origins(q)
            corners = kin.finger_corners(q, SELF_COLLISION_FINGER_OPENING_MM)
            top = np.array([0, 0, SELF_COLLISION_BASE_PLATE_TOP_MM])
            movers = {"tool body": (pts[5], pts[6], 35.0), "left finger": (corners[2], corners[0], 10.0),
                      "right finger": (corners[3], corners[1], 10.0)}
            targets = {"base plate": (np.zeros(3), top, 57.0), "base column": (top, pts[1], 45.0), "upper arm": (pts[1], pts[2], 40.0)}
            pairs = {(m, t): segment_distance_mm(a, b, c, d) - ra - rb
                     for m, (a, b, ra) in movers.items() for t, (c, d, rb) in targets.items()}
            clearance, mover, target = left.self_collision_clearance(q)
            assert clearance == pytest.approx(min(pairs.values()), abs=1e-9)
            assert pairs[(mover, target)] == pytest.approx(clearance, abs=1e-9)

    def test_home_pose_is_clear(self, left):
        clearance, _, _ = left.self_collision_clearance(np.zeros(6))
        assert clearance > SELF_COLLISION_MARGIN_MM
        assert left.self_collision_check(np.zeros(6)) is None

    def test_gripper_folded_into_the_base(self, left):
        q = np.array([8.0, 121.0, -2.0, 31.0, 8.0, 83.0])
        clearance, mover, target = left.self_collision_clearance(q)
        assert clearance < 0 and (mover, target) == ("tool body", "base plate")
        assert left.self_collision_check(q).startswith(
            f"left self-collision: tool body vs base plate (clearance {clearance:.0f} mm < 15)")


class TestChecks:
    def test_joint_limit_reason(self):
        assert joint_limit_reason(np.zeros(6)) is None
        assert joint_limit_reason([0, 0, 10, 0, 0, 0]) == "joint limit exceeded (j3=10.0 deg)"
        assert joint_limit_reason([160, -1, 0, 0, 0, 125]) == "joint limit exceeded (j1=160.0 deg, j2=-1.0 deg, j6=125.0 deg)"
        assert joint_limit_reason([0, np.nan, 0, 0, 0, 0]) == "joint limit exceeded (j2=nan deg)"

    def test_table_check_thresholds(self, kin, left):
        high = np.full((7, 3), 500.0)
        assert left.table_check(None, origins=high) is None
        low_elbow = high.copy()
        low_elbow[2, 2] = 3.5 + 60.0 - 1.2
        assert left.table_check(None, origins=low_elbow) == "left joint3 z=62mm below table clearance 60mm"
        wrist_near_table = high.copy()
        wrist_near_table[5, 2] = 10.0
        assert left.table_check(None, origins=wrist_near_table) is None
        wrist_near_table[5, 2] = 3.0
        assert left.table_check(None, origins=wrist_near_table) == "left joint6 z=3mm below table clearance 60mm"
        deep_tip = high.copy()
        deep_tip[6, 2] = 3.5 - 5.0 - 0.1
        assert left.table_check(None, origins=deep_tip) == "left fingertip z=-1.6mm (bias-corrected) is below the table"
        biased = ArmModel("left", kin, np.eye(4), fingertip_bias_mm=1.0)
        assert biased.table_check(None, origins=deep_tip) is None
        assert left.table_check(np.zeros(6)) is None
        assert left.table_check([0, 90, -10, 0, 0, 0]) == "left fingertip z=-95.9mm (bias-corrected) is below the table"

    def test_front_and_facing_rules(self, left):
        assert left.front_violation(np.zeros(6)) is None
        assert left.front_violation([120, 60, -60, 0, 30, 0]).startswith("left j1=120 deg turns the arm past the side line")
        assert left.facing_violation(np.zeros(6), np.array([300.0, 10.0, 0.0])) is None
        assert left.facing_violation(np.zeros(6), np.array([10.0, 20.0, 0.0])) is None
        message = left.facing_violation(np.zeros(6), np.array([-300.0, 100.0, 0.0]))
        assert message.startswith("left flipped IK solution: j1=0 deg faces away from the target (azimuth 162 deg, diff 162 > 90)")

    def test_floor_check_reports_the_table_first(self, left):
        q = np.array([8.0, 121.0, -2.0, 31.0, 8.0, 83.0])
        assert left.floor_check(q) == left.table_check(q)
        assert left.floor_check(np.zeros(6)) is None

    def test_inter_arm_reason(self, kin, left):
        t = np.eye(4)
        t[1, 3] = -0.59
        right = ArmModel("right", kin, t)
        home = np.zeros(6)
        assert inter_arm_reason(left, home, right, home) is None
        reaching = np.array([-75.0, 90.0, -60.0, 0.0, 30.0, 0.0])
        clearance, part, other_part = capsules_clearance_mm(left.body_capsules(reaching), right.body_capsules(home))
        assert clearance < INTER_ARM_MARGIN_MM
        reason = inter_arm_reason(left, reaching, right, home)
        assert reason == (f"left would hit the right arm: left {part} vs right {other_part} (clearance {clearance:.0f} mm < 30); "
                          "the right arm is in the way — park('right') or home() it first, or choose a target farther from it")
        assert inter_arm_reason(left, reaching, right, home, margin_mm=clearance - 1.0) is None
        assert inter_arm_reason(left, reaching, right, home, other_caps=right.body_capsules(home)) == reason
        mirrored = inter_arm_reason(right, home, left, reaching)
        assert mirrored is not None and re.search(rf"right {other_part} vs left {part} \(clearance {clearance:.0f} mm < 30\)", mirrored)
