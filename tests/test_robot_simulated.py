"""The kinematic simulator: servo, kinematic grasping and the ray-cast RGB-D scene, all computed for real."""
from __future__ import annotations

import time

import numpy as np
import pytest

from urai.robot import scene as scene_module
from urai.robot.arm_model import approach_pose_dir
from urai.robot.cameras import CameraFrame
from urai.robot.driver import CTRL_MODE_CAN_TEXT, MotionDisabledError
from urai.robot.kinematics import m_to_mm_matrix
from urai.robot.scene import Box, Capsule, Cup, Scene, SceneObject, Table, render_rgbd
from urai.robot.simulated import (
    FIRMWARE_MAX_JOINT_SPEED_DEG_S,
    SimulatedArmDriver,
    build_simulated_runtime,
    default_scene,
    nominal_intrinsics,
)

TABLE_Z = 0.0035
DOWN = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def pose(x, y, z, yaw_deg=0.0):
    t = np.eye(4)
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    t[:2, :2] = [[c, -s], [s, c]]
    t[:3, 3] = [x, y, z]
    return t


def looking_down(x, y, height):
    t = DOWN.copy()
    t[:3, 3] = [x, y, height]
    return t


def table():
    return Table(z=TABLE_Z, x_range=(-0.3, 0.9), y_range=(-1.0, 0.5))


def wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def solve(model, target, seed=None):
    """Joints reaching the world TCP pose ``target`` with the arm model's own IK."""
    t_base_link6 = m_to_mm_matrix(model.t_base_world @ target) @ model.t_tcp_link6
    seeds = ([seed] if seed is not None else []) + list(model.kin.seed_candidates(t_base_link6, count=16))
    for start in seeds:
        q, _, _ = model.ik_tcp_world(target, start)
        if q is not None:
            return q
    raise AssertionError(f"no IK solution for {target[:3, 3]}")


# ---------------------------------------------------------------------------- servo


def test_servo_rate_follows_controller_speed(arm_models):
    scene = Scene(table(), [])
    arm = SimulatedArmDriver("left", arm_models["left"], scene, speed_percent=10)
    rate = FIRMWARE_MAX_JOINT_SPEED_DEG_S * 0.10
    began = time.monotonic()
    arm.command_joints([20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    time.sleep(0.3)
    moved = arm.joints()[0]
    elapsed = time.monotonic() - began
    assert 0.0 < moved <= rate * elapsed + 1e-9
    assert moved >= rate * 0.25
    assert wait_until(lambda: arm.joints()[0] == 20.0, timeout_s=3.0)
    arm.set_joint_mode(100)
    began = time.monotonic()
    arm.command_joints([-20.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert wait_until(lambda: arm.joints()[0] == -20.0, timeout_s=2.0)
    assert time.monotonic() - began >= 40.0 / FIRMWARE_MAX_JOINT_SPEED_DEG_S - 0.02


def test_references_are_quantised_and_clamped_into_limits(arm_models):
    arm = SimulatedArmDriver("left", arm_models["left"], Scene(table(), []), speed_percent=100)
    lower = arm_models["left"].kin.limits_deg[2][0]
    arm.command_joints([12.3456789, lower - 5.0, 0.0, 0.0, 0.0, 0.0])
    assert wait_until(lambda: np.allclose(arm.joints(), [12.346, lower, 0, 0, 0, 0], atol=1e-12))


def test_state_is_can_control_enabled_and_fresh(arm_models):
    arm = SimulatedArmDriver("right", arm_models["right"], Scene(table(), []), speed_percent=20)
    state = arm.state()
    assert state.ctrl_mode == CTRL_MODE_CAN_TEXT and "CAN" in state.ctrl_mode
    assert state.enabled and state.arm_status == 0 and state.err_code == 0
    assert arm.feedback_age_s() == 0.0 and arm.joint_feedback_age_s() == 0.0
    assert arm.clear_motor_faults() == []


def test_motion_gate_refuses_every_command(arm_models):
    arm = SimulatedArmDriver("left", arm_models["left"], Scene(table(), []), speed_percent=20, allow_motion=False)
    for command in (lambda: arm.set_joint_mode(10), lambda: arm.command_joints([5.0, 0, 0, 0, 0, 0]),
                    lambda: arm.set_gripper(30.0, 1000), arm.exit_teaching, arm.take_control,
                    lambda: arm.init_gripper(70)):
        with pytest.raises(MotionDisabledError):
            command()
    time.sleep(0.05)
    assert np.array_equal(arm.joints(), np.zeros(6))


# ---------------------------------------------------------------------------- grasping


@pytest.fixture
def block_scene():
    return Scene(table(), [SceneObject("block", (Box((0.04, 0.04, 0.04), (200, 40, 40)),), pose(0.30, -0.10, TABLE_Z))])


def test_gripper_stops_on_object_carries_and_releases_it(arm_models, block_scene):
    model = arm_models["left"]
    grasp = approach_pose_dir([0.30, -0.10, TABLE_Z + 0.02], 0.0, 0.0, [1.0, 0.0])
    q_grasp = solve(model, grasp)
    arm = SimulatedArmDriver("left", model, block_scene, speed_percent=100, joints_deg=q_grasp, gripper_mm=70.0)
    arm.set_gripper(0.0, 1000)
    assert wait_until(lambda: arm.held_object() == "block", timeout_s=2.0)
    time.sleep(0.2)
    assert arm.state().gripper_mm == pytest.approx(40.0, abs=0.05)

    lifted = grasp.copy()
    lifted[:3, 3] += [0.05, 0.03, 0.10]
    q_lift = solve(model, lifted, seed=q_grasp)
    arm.command_joints(q_lift)
    assert wait_until(lambda: np.abs(arm.joints() - np.round(q_lift, 3)).max() < 1e-9, timeout_s=3.0)
    carried = block_scene.pose("block")
    tcp = model.fk_tcp_world(arm.joints())
    assert carried[:3, 3] == pytest.approx(tcp[:3, 3] - [0.0, 0.0, 0.02], abs=2e-3)

    arm.set_gripper(70.0, 1000)
    assert wait_until(lambda: arm.held_object() is None, timeout_s=2.0)
    settled = block_scene.pose("block")
    assert settled[2, 3] == pytest.approx(TABLE_Z)
    assert settled[:2, 3] == pytest.approx(carried[:2, 3])
    assert settled[:3, 2] == pytest.approx([0.0, 0.0, 1.0])


def test_gripper_closes_fully_on_empty_air(arm_models, block_scene):
    model = arm_models["left"]
    q = solve(model, approach_pose_dir([0.30, 0.10, TABLE_Z + 0.02], 0.0, 0.0, [1.0, 0.0]))
    arm = SimulatedArmDriver("left", model, block_scene, speed_percent=100, joints_deg=q, gripper_mm=70.0)
    arm.set_gripper(0.0, 1000)
    assert wait_until(lambda: arm.state().gripper_mm == 0.0, timeout_s=2.0)
    assert arm.held_object() is None


def test_object_wider_than_the_opening_is_not_grasped(block_scene):
    hand = pose(0.30, -0.10, TABLE_Z + 0.02 + 0.1425) @ np.diag([1.0, -1.0, -1.0, 1.0])
    pads = (0.1405, 0.1325)
    assert block_scene.width_between_fingers(hand, 0.070, pads)[1] == pytest.approx(0.04)
    assert block_scene.width_between_fingers(hand, 0.030, pads) is None


def test_release_settles_on_the_support_below():
    scene = Scene(table(), [SceneObject("bowl", (Cup(0.07, 0.05, 0.004, 0.006, (220, 220, 220)),), pose(0.4, 0.0, TABLE_Z)),
                            SceneObject("block", (Box((0.03, 0.03, 0.03), (0, 0, 200)),), pose(0.1, 0.1, TABLE_Z))])
    hand = [pose(0.1, 0.1, TABLE_Z)]
    scene.attach("block", lambda: hand[0])
    hand[0] = pose(0.4, 0.0, 0.3, yaw_deg=30.0)
    assert scene.pose("block")[:3, 3] == pytest.approx([0.4, 0.0, 0.3])
    settled = scene.release("block")
    assert settled[2, 3] == pytest.approx(TABLE_Z + 0.006)
    assert np.degrees(np.arctan2(settled[1, 0], settled[0, 0])) == pytest.approx(30.0)
    assert scene.support_height([0.4, 0.0]) == pytest.approx(TABLE_Z + 0.006 + 0.03)
    assert scene.support_height([0.4 + 0.068, 0.0]) == pytest.approx(TABLE_Z + 0.05)


# ---------------------------------------------------------------------------- rendering


def test_depth_matches_box_and_table_geometry():
    k = nominal_intrinsics(640, 480)
    block = SceneObject("block", (Box((0.04, 0.04, 0.04), (200, 40, 40)),), pose(0.3, -0.1, TABLE_Z))
    rgb, depth = render_rgbd(k, looking_down(0.3, -0.1, 1.0), 640, 480, table(), [block])
    top = 1.0 - (TABLE_Z + 0.04)
    assert depth[240, 320] == pytest.approx(top, abs=1e-6)
    inside, outside = int(320 + k[0, 0] * 0.019 / top), int(np.ceil(320 + k[0, 0] * 0.0205 / top)) + 1
    assert depth[240, inside] == pytest.approx(top, abs=1e-6)
    assert depth[240, outside] == pytest.approx(1.0 - TABLE_Z, abs=1e-6)
    lambert = scene_module._AMBIENT + (1.0 - scene_module._AMBIENT) * scene_module._LIGHT[2]  # upward face
    assert np.array_equal(rgb[240, 320], np.round(np.array([200, 40, 40]) * lambert))
    assert rgb.dtype == np.uint8 and depth.dtype == np.float32


def test_depth_sees_into_cups_and_onto_rims():
    k = nominal_intrinsics(640, 480)
    cup = SceneObject("cup", (Cup(0.05, 0.08, 0.005, 0.01, (60, 160, 90)),), pose(0.3, 0.0, TABLE_Z))
    _, depth = render_rgbd(k, looking_down(0.3, 0.0, 1.0), 640, 480, table(), [cup])
    assert depth[240, 320] == pytest.approx(1.0 - (TABLE_Z + 0.01), abs=1e-6)
    rim = 1.0 - (TABLE_Z + 0.08)
    assert depth[240, round(320 + k[0, 0] * 0.0475 / rim)] == pytest.approx(rim, abs=1e-6)


def test_capsules_floor_and_sky():
    k = nominal_intrinsics(640, 480)
    cap = Capsule(np.array([0.2, 0.0, 0.3]), np.array([0.4, 0.0, 0.3]), 0.03, (150, 150, 150))
    _, depth = render_rgbd(k, looking_down(0.3, 0.0, 1.0), 640, 480, table(), [], [cap])
    assert depth[240, 320] == pytest.approx(1.0 - 0.33, abs=1e-6)
    off_table = looking_down(2.0, 0.0, 1.0)
    _, depth = render_rgbd(k, off_table, 640, 480, table(), [])
    assert depth[240, 320] == pytest.approx(1.0 - (TABLE_Z - 0.75), abs=1e-6)
    up = off_table @ np.diag([1.0, -1.0, -1.0, 1.0])
    _, depth = render_rgbd(k, up, 640, 480, table(), [])
    assert np.isnan(depth).all()


def test_points_world_lands_on_the_rendered_surfaces():
    k = nominal_intrinsics(640, 480)
    t_world_cam = looking_down(0.3, -0.1, 1.0)
    block = SceneObject("block", (Box((0.04, 0.04, 0.04), (200, 40, 40)),), pose(0.3, -0.1, TABLE_Z, 25.0))
    rgb, depth = render_rgbd(k, t_world_cam, 640, 480, table(), [block])
    points = CameraFrame(rgb, depth, k, t_world_cam).points_world()
    assert points[240, 320] == pytest.approx([0.3, -0.1, TABLE_Z + 0.04], abs=1e-6)
    assert points[100, 100, 2] == pytest.approx(TABLE_Z, abs=1e-6)
    assert points[5, 5, 2] == pytest.approx(TABLE_Z - 0.75, abs=1e-6)  # beyond the table edge: the floor


def test_default_scene_objects_are_on_the_table_and_in_view(arm_models, example_calibration):
    scene = default_scene(arm_models)
    runtime = build_simulated_runtime(example_calibration, scene=scene)
    frame = runtime.capture("head")
    for obj in scene.objects():
        lo, hi = obj.local_bounds()
        assert obj.t_world_obj[2, 3] + lo[2] == pytest.approx(arm_models["left"].table_z_mm / 1000.0)
        centre = obj.t_world_obj[:3, :3] @ ((lo + hi) / 2.0) + obj.t_world_obj[:3, 3]
        cam = np.linalg.inv(frame.t_world_cam) @ np.append(centre, 1.0)
        u, v = (frame.k @ (cam[:3] / cam[2]))[:2]
        assert 0 <= u < frame.rgb.shape[1] and 0 <= v < frame.rgb.shape[0]
        assert np.linalg.norm(frame.points_world()[int(v), int(u)] - centre) < 0.12
