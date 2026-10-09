"""Read-only checks on the real rig: CAN feedback, controller state, cameras and the closed motion gate.

Nothing here commands an arm or a gripper: the runtime is opened with the motion gate closed.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.hardware

ARMS = ("left", "right")
CAMERAS = ("head", "hand_left", "hand_right")


@pytest.fixture(scope="module")
def runtime(open_robot_runtime):
    with open_robot_runtime(allow_motion=False) as rt:
        yield rt


@pytest.mark.parametrize("arm", ARMS)
def test_can_feedback_is_fresh(runtime, arm):
    driver = runtime.arms[arm]
    for _ in range(5):
        assert driver.feedback_age_s() < 0.5, "driver (low-speed) feedback is stale: power, emergency stop, CAN wiring?"
        assert driver.joint_feedback_age_s() < 0.5, "joint feedback is stale"
        time.sleep(0.1)
    runtime.require_fresh_feedback(arm)


@pytest.mark.parametrize("arm", ARMS)
def test_state_fields(runtime, arm):
    state = runtime.arms[arm].state()
    assert state.joints_deg.shape == (6,) and np.isfinite(state.joints_deg).all()
    limits = runtime.models[arm].kin.limits_deg
    for i, q in enumerate(state.joints_deg, start=1):
        assert limits[i][0] - 3.0 <= q <= limits[i][1] + 3.0, f"j{i}={q:.2f} deg is far outside the URDF limits"
    assert state.ctrl_mode.endswith(")") and "(0x" in state.ctrl_mode, state.ctrl_mode
    assert isinstance(state.arm_status, int) and isinstance(state.err_code, int)
    assert -1.0 <= state.gripper_mm <= runtime.arms[arm].gripper_range_mm + 5.0


@pytest.mark.parametrize("arm", ARMS)
def test_closed_gate_refuses_and_nothing_moves(runtime, arm):
    from urai.robot.driver import MotionDisabledError

    before = runtime.joints_deg(arm)
    for command in (lambda: runtime.arms[arm].set_joint_mode(10), lambda: runtime.arms[arm].command_joints(before),
                    lambda: runtime.arms[arm].set_gripper(10.0, 1000), lambda: runtime.clear_motor_faults(arm)):
        with pytest.raises(MotionDisabledError):
            command()
    time.sleep(0.5)
    assert np.abs(runtime.joints_deg(arm) - before).max() < 0.2


@pytest.mark.skipif(os.environ.get("URAI_TEACHING_ARM") not in ARMS,
                    reason="operator-assisted: put one arm in drag-teach mode and set URAI_TEACHING_ARM=left or right")
def test_drag_teach_mode_reported_by_runtime(runtime):
    arm = os.environ["URAI_TEACHING_ARM"]
    other = "right" if arm == "left" else "left"
    assert "TEACHING" in runtime.arms[arm].state().ctrl_mode
    assert "TEACHING" not in runtime.arms[other].state().ctrl_mode
    assert runtime.arms[arm].feedback_age_s() < 0.5


@pytest.mark.parametrize("camera", CAMERAS)
def test_camera_frames_are_fresh_and_consistent(runtime, camera):
    first = runtime.capture(camera)
    second = runtime.capture(camera)
    h, w = first.rgb.shape[:2]
    assert first.rgb.dtype == np.uint8 and first.rgb.shape == (h, w, 3)
    assert first.depth_m.shape == (h, w) and first.depth_m.dtype == np.float32
    finite = np.isfinite(first.depth_m)
    assert finite.mean() > 0.3, "most of the depth image is invalid"
    assert (first.depth_m[finite] > 0.05).all() and (first.depth_m[finite] < 10.0).all()
    assert second.timestamp > first.timestamp
    assert not np.array_equal(first.rgb, second.rgb)
    k = first.k
    assert k.shape == (3, 3) and k[2, 2] == 1.0 and k[0, 1] == 0.0
    assert 0.4 * w < k[0, 0] < 1.2 * w and abs(k[0, 0] - k[1, 1]) < 0.05 * k[0, 0]
    assert abs(k[0, 2] - w / 2) < 0.1 * w and abs(k[1, 2] - h / 2) < 0.1 * h
    assert np.allclose(first.t_world_cam[3], [0, 0, 0, 1])


def test_head_camera_pose_and_table(runtime, hardware_config):
    calibration, rig = hardware_config
    frame = runtime.capture("head")
    assert np.array_equal(frame.t_world_cam, calibration.t_world_head_camera)
    points = frame.points_world()
    h, w = frame.depth_m.shape
    centre = points[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3)
    centre = centre[np.isfinite(centre).all(axis=1)]
    on_table = np.abs(centre[:, 2] - rig.table_z_mm / 1000.0) < 0.015
    assert on_table.mean() > 0.3, "the head camera does not see the table at the calibrated height (check the calibration)"


def test_head_frame_pump(runtime, hardware_config):
    _, rig = hardware_config
    if not rig.head_stream:
        pytest.skip("head_stream is off in the rig configuration")
    assert runtime.cameras.streaming("head")
    first = runtime.cameras.wait_color("head", None, 2.0)
    later = runtime.cameras.wait_color("head", first.seq, 1.0)
    assert later.seq > first.seq and later.arrival_ms > first.arrival_ms
    assert later.bgr.ndim == 3 and later.bgr.dtype == np.uint8
    assert not runtime.cameras.streaming("hand_left")


@pytest.mark.parametrize("arm", ARMS)
def test_wrist_camera_pose_follows_the_arm(runtime, arm):
    frame = runtime.capture(f"hand_{arm}")
    assert np.allclose(frame.t_world_cam, runtime.camera_pose(f"hand_{arm}"), atol=1e-3)
