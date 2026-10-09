"""Motion checks on the real rig: start-up takeover without motion, a small wrist step, gripper open/close.

Before running: clear the workspace around both arms, keep the emergency stop in reach, and make sure
both grippers are empty. Every motion here is a few degrees of joint 6 or 10 mm of gripper opening.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

pytestmark = [pytest.mark.hardware, pytest.mark.motion]

ARMS = ("left", "right")


def wait_until(predicate, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.fixture(scope="module")
def before_startup(hardware_config):
    """Joints and gripper openings read over CAN (read-only) before the motion runtime starts."""
    from urai.robot.driver import PiperArmDriver
    from urai.robot.kinematics import PiperXKinematics

    _, rig = hardware_config
    limits = PiperXKinematics().limits_deg
    readings = {}
    for arm, interface in (("left", rig.left_interface), ("right", rig.right_interface)):
        driver = PiperArmDriver(interface, joint_limits_deg=limits, speed_percent=rig.speed_percent,
                                allow_motion=False, name=f"{arm} arm").connect()
        try:
            assert driver.joint_feedback_age_s() < 0.5
            state = driver.state()
            readings[arm] = (state.joints_deg, state.gripper_mm)
        finally:
            driver.close()
    return readings


@pytest.fixture(scope="module")
def runtime(before_startup, open_robot_runtime):
    with open_robot_runtime(allow_motion=True) as rt:
        yield rt


@pytest.mark.parametrize("arm", ARMS)
def test_startup_takes_control_without_moving(runtime, before_startup, arm):
    joints, _ = before_startup[arm]
    state = runtime.arms[arm].state()
    assert "CAN" in state.ctrl_mode and state.enabled
    assert state.arm_status == 0 and state.err_code == 0
    assert np.abs(state.joints_deg - joints).max() < 1.0, "start-up moved the arm"
    first = runtime.joints_deg(arm)
    time.sleep(1.0)
    assert np.abs(runtime.joints_deg(arm) - first).max() < 0.2, "the arm is not holding its pose"
    assert runtime.clear_motor_faults(arm) == []


@pytest.mark.parametrize("arm", ARMS)
def test_hold_reference_keeps_the_pose(runtime, arm):
    driver = runtime.arms[arm]
    with runtime.execution_lock:
        q = driver.joints()
        driver.set_joint_mode(5)
        driver.command_joints(q)
        time.sleep(1.0)
        assert np.abs(driver.joints() - q).max() < 0.2


@pytest.mark.parametrize("arm", ARMS)
def test_small_wrist_step_and_back(runtime, arm):
    driver = runtime.arms[arm]
    lower, upper = runtime.models[arm].kin.limits_deg[6]
    with runtime.execution_lock:
        q0 = driver.joints()
        step = 3.0 if q0[5] + 3.0 < upper - 1.0 else -3.0
        assert lower + 1.0 < q0[5] + step < upper - 1.0
        target = q0.copy()
        target[5] += step
        driver.set_joint_mode(10)
        driver.command_joints(target)
        assert wait_until(lambda: np.abs(driver.joints() - target).max() < 0.3, 3.0), \
            f"joint 6 did not reach the 3 deg step: {driver.joints() - target}"
        assert np.abs((driver.joints() - q0)[:5]).max() < 0.3
        driver.command_joints(q0)
        assert wait_until(lambda: np.abs(driver.joints() - q0).max() < 0.3, 3.0)


@pytest.mark.parametrize("arm", ARMS)
def test_gripper_opens_and_closes_by_10_mm(runtime, arm):
    driver = runtime.arms[arm]
    with runtime.execution_lock:
        start = driver.state().gripper_mm
        target = start + 10.0 if start < driver.gripper_range_mm - 15.0 else start - 10.0
        effort = runtime.open_effort if target > start else runtime.grip_effort
        driver.set_gripper(target, effort)
        assert wait_until(lambda: abs(driver.state().gripper_mm - target) < 1.5, 2.0), driver.state().gripper_mm
        back = min(max(start, 0.0), driver.gripper_range_mm)
        driver.set_gripper(back, runtime.open_effort if back > target else runtime.grip_effort)
        assert wait_until(lambda: abs(driver.state().gripper_mm - back) < 1.5, 2.0)
