"""RobotRuntime: camera extrinsics chain, motion gate, feedback freshness and the urai-robot entry point."""
from __future__ import annotations

import json
import threading
import time
import uuid

import numpy as np
import pytest

from urai.robot.driver import MotionDisabledError
from urai.robot.kinematics import mm_to_m_matrix
from urai.robot.launch import build_parser, build_runtime, load_rig_config, main
from urai.robot.rig import RigConfig
from urai.robot.runtime import CAMERAS, RobotRuntime
from urai.robot.simulated import build_simulated_runtime


def wait_until(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


# ---------------------------------------------------------------------------- cameras


def test_head_camera_pose_is_the_calibrated_one(sim_runtime, example_calibration):
    assert np.array_equal(sim_runtime.camera_pose("head"), example_calibration.t_world_head_camera)
    frame = sim_runtime.capture("head")
    assert np.array_equal(frame.t_world_cam, example_calibration.t_world_head_camera)
    assert frame.rgb.shape == (720, 1280, 3) and frame.depth_m.shape == (720, 1280)
    assert frame.camera == "head"


def test_wrist_camera_rides_on_link6(sim_runtime, example_calibration):
    """The wrist camera keeps its calibrated offset from link6 and turns with joint 1 about the base axis."""
    arm = sim_runtime.arms["left"]
    before = sim_runtime.camera_pose("hand_left")
    model = sim_runtime.models["left"]
    link6 = model.t_world_base @ mm_to_m_matrix(model.kin.fk(sim_runtime.joints_deg("left")))
    offset = example_calibration.wrist_cameras["left"].t_link6_camera[:3, 3]
    assert np.linalg.norm(before[:3, 3] - link6[:3, 3]) == pytest.approx(np.linalg.norm(offset))
    arm.set_joint_mode(100)
    arm.command_joints([30.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert wait_until(lambda: sim_runtime.joints_deg("left")[0] == 30.0)
    assert np.allclose(sim_runtime.camera_pose("hand_left"), rot_z(30.0) @ before, atol=1e-9)
    frame = sim_runtime.capture("hand_left")
    assert np.allclose(frame.t_world_cam, sim_runtime.camera_pose("hand_left"))
    assert np.array_equal(sim_runtime.camera_pose("hand_right"), example_calibration.t_world_wrist_camera(
        "right", sim_runtime.models["right"].kin.fk(sim_runtime.joints_deg("right"))))


def test_head_depth_reaches_the_table(sim_runtime):
    frame = sim_runtime.capture("head")
    table_point = np.array([0.75, -0.295, sim_runtime.models["left"].table_z_mm / 1000.0, 1.0])
    cam = np.linalg.inv(frame.t_world_cam) @ table_point
    u, v = (frame.k @ (cam[:3] / cam[2]))[:2]
    assert frame.depth_m[round(v), round(u)] == pytest.approx(cam[2], abs=2e-3)
    assert frame.points_world()[round(v), round(u)] == pytest.approx(table_point[:3], abs=2e-3)


def test_unknown_camera_is_rejected(sim_runtime):
    assert CAMERAS == ("head", "hand_left", "hand_right")
    with pytest.raises(ValueError):
        sim_runtime.capture("front")
    with pytest.raises(ValueError):
        sim_runtime.cameras.streaming("front")
    assert not sim_runtime.cameras.streaming("head")
    with pytest.raises(TimeoutError):
        sim_runtime.cameras.wait_color("head", None, 0.1)


# ---------------------------------------------------------------------------- motion gate


def test_closed_gate_refuses_every_writer(example_calibration):
    runtime = build_simulated_runtime(example_calibration, allow_motion=False)
    assert not runtime.motion_allowed
    with pytest.raises(MotionDisabledError):
        runtime.require_motion("test")
    assert issubclass(MotionDisabledError, RuntimeError)
    for refused in (lambda: runtime.start_control(70.0, 20), lambda: runtime.take_control("left"),
                    lambda: runtime.clear_motor_faults("right"), lambda: runtime.arms["left"].command_joints([10.0, 0, 0, 0, 0, 0]),
                    lambda: runtime.arms["right"].set_gripper(30.0, 1000)):
        with pytest.raises(MotionDisabledError):
            refused()
    time.sleep(0.05)
    assert np.array_equal(runtime.joints_deg("left"), np.zeros(6))
    runtime.capture("head")  # observing is always allowed


def test_start_control_holds_the_measured_pose(sim_runtime):
    before = {arm: sim_runtime.joints_deg(arm) for arm in ("left", "right")}
    sim_runtime.start_control(70.0, 20)
    time.sleep(0.1)
    for arm, q in before.items():
        assert np.array_equal(sim_runtime.joints_deg(arm), q)
    assert sim_runtime.last_gripper_command == {"left": None, "right": None}


def test_execution_lock_is_shared_and_reentrant(sim_runtime):
    lock = sim_runtime.execution_lock
    with lock, lock:
        held_elsewhere = []
        worker = threading.Thread(target=lambda: held_elsewhere.append(lock.acquire(blocking=False)))
        worker.start()
        worker.join()
        assert held_elsewhere == [False]
    assert sim_runtime.clear_motor_faults("left") == []


def test_runtime_needs_both_arms(sim_runtime):
    with pytest.raises(ValueError):
        RobotRuntime(name="x", arms={"left": sim_runtime.arms["left"]}, models=sim_runtime.models,
                     cameras=sim_runtime.cameras, calibration=sim_runtime.calibration, motion_allowed=True,
                     open_effort=1000, grip_effort=1000)


def test_silent_can_bus_blocks_control_takeover(sim_runtime):
    """Real drivers on a bus nobody answers: the runtime refuses to command a hold from the zero reading."""
    can = pytest.importorskip("can", reason="install the robot extra: pip install -e .[robot]")
    pytest.importorskip("piper_sdk", reason="install the robot extra: pip install -e .[robot]")
    from urai.robot.driver import PiperArmDriver

    channels = {arm: f"urai-rt-{uuid.uuid4().hex[:8]}" for arm in ("left", "right")}
    wires = {arm: can.Bus(interface="virtual", channel=ch) for arm, ch in channels.items()}
    drivers = {arm: PiperArmDriver(ch, joint_limits_deg=sim_runtime.models[arm].kin.limits_deg, speed_percent=20,
                                   allow_motion=True, bustype="virtual", name=f"{arm} arm").connect(settle_s=0.05)
               for arm, ch in channels.items()}
    runtime = RobotRuntime(name="silent rig", arms=drivers, models=sim_runtime.models, cameras=sim_runtime.cameras,
                           calibration=sim_runtime.calibration, motion_allowed=True, open_effort=1000, grip_effort=1000)
    try:
        for wire in wires.values():
            while wire.recv(timeout=0.2) is not None:
                pass
        with pytest.raises(RuntimeError, match="joint feedback inf s old"):
            runtime.require_fresh_feedback("left")
        for refused in (lambda: runtime.start_control(70.0, 20), lambda: runtime.take_control("right"),
                        lambda: runtime.clear_motor_faults("left")):
            with pytest.raises(RuntimeError, match="feedback"):
                refused()
        assert all(wire.recv(timeout=0.3) is None for wire in wires.values())
    finally:
        for driver in drivers.values():
            driver.close()
        for wire in wires.values():
            wire.shutdown()


# ---------------------------------------------------------------------------- entry point


def test_rig_config_file(tmp_path):
    path = tmp_path / "rig.json"
    path.write_text(json.dumps({"left_interface": "can_l", "speed_percent": 30, "table_z_mm": 10, "head_stream": False}))
    rig = load_rig_config(path)
    assert rig == RigConfig(left_interface="can_l", speed_percent=30, table_z_mm=10.0, head_stream=False)
    for bad, error in (({"speed": 30}, ValueError), ({"speed_percent": 30.5}, TypeError),
                       ({"head_stream": 1}, TypeError), ({"grip_effort": True}, TypeError), ([1, 2], TypeError)):
        path.write_text(json.dumps(bad))
        with pytest.raises(error):
            load_rig_config(path)


def test_sim_entry_builds_the_simulated_runtime(tmp_path):
    rig = tmp_path / "rig.json"
    rig.write_text(json.dumps({"speed_percent": 35, "head_width": 640, "head_height": 360}))
    runtime = build_runtime(build_parser().parse_args(["--sim", "--rig", str(rig)]))
    try:
        assert runtime.name == "Simulated PiPER-X dual arm" and runtime.motion_allowed
        assert runtime.capture("head").rgb.shape == (360, 640, 3)
    finally:
        runtime.close()


@pytest.mark.parametrize("argv", [["--sim", "--host", "0.0.0.0"], ["--host", "127.0.0.1"]])
def test_entry_refuses_before_touching_anything(argv, monkeypatch):
    monkeypatch.delenv("URAI_TOKEN", raising=False)
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2
