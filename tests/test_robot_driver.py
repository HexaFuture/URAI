"""PiperArmDriver on python-can's in-process ``virtual`` bus.

The real ``piper_sdk`` encodes and sends every frame; a second bus on the same virtual channel records
what goes out. Expected payloads are written from the PiPER CAN protocol (big-endian fields). Nothing
answers on the bus, so these tests also pin down what the driver does when the arm is silent.
"""
from __future__ import annotations

import struct
import uuid

import numpy as np
import pytest

can = pytest.importorskip("can", reason="install the robot extra: pip install -e .[robot]")
pytest.importorskip("piper_sdk", reason="install the robot extra: pip install -e .[robot]")

from urai.robot.driver import (
    CTRL_MODE_CAN_TEXT,
    MotionDisabledError,
    PiperArmDriver,
)
from urai.robot.kinematics import PiperXKinematics

LIMITS = PiperXKinematics().limits_deg


def frame(can_id, payload):
    return can_id, bytes(payload).ljust(8, b"\0")


class Wire:
    """A listener on the driver's virtual channel."""

    def __init__(self, channel):
        self.bus = can.Bus(interface="virtual", channel=channel)

    def frames(self, wait_s=0.3):
        out = []
        while (msg := self.bus.recv(timeout=wait_s)) is not None:
            out.append((msg.arbitration_id, bytes(msg.data)))
        return out

    def close(self):
        self.bus.shutdown()


@pytest.fixture
def wire_driver(request):
    allow = getattr(request, "param", True)
    channel = f"urai-test-{uuid.uuid4().hex[:8]}"
    wire = Wire(channel)
    driver = PiperArmDriver(channel, joint_limits_deg=LIMITS, speed_percent=20, allow_motion=allow, bustype="virtual",
                            name="test arm").connect(settle_s=0.05)
    wire.frames()  # parameter queries piper_sdk sends on ConnectPort
    yield wire, driver
    driver.close()
    wire.close()


def test_joint_mode_frame(wire_driver):
    wire, driver = wire_driver
    driver.set_joint_mode(40)
    # 0x151: ctrl mode CAN (0x01), MOVE J (0x01), speed 40 %, normal follow (0x00)
    assert wire.frames() == [frame(0x151, [0x01, 0x01, 40, 0x00])]


def test_joint_reference_frames(wire_driver):
    wire, driver = wire_driver
    driver.command_joints([10.5, 20.25, -30.125, 5.0, -15.0, 45.0004])
    assert wire.frames() == [
        (0x155, struct.pack(">ii", 10500, 20250)),
        (0x156, struct.pack(">ii", -30125, 5000)),
        (0x157, struct.pack(">ii", -15000, 45000)),
    ]


def test_gripper_frame(wire_driver):
    wire, driver = wire_driver
    driver.set_gripper(35.5, effort=1500)
    # 0x159: opening 0.001 mm (int32), effort 0.001 N·m (uint16), enable (0x01), no zeroing
    assert wire.frames() == [(0x159, struct.pack(">iHBB", 35500, 1500, 0x01, 0x00))]


def test_exit_teaching_frame(wire_driver):
    wire, driver = wire_driver
    driver.exit_teaching()
    assert wire.frames() == [frame(0x150, [0x00, 0x00, 0x02])]


def test_invalid_commands_send_nothing(wire_driver):
    wire, driver = wire_driver
    for bad in (lambda: driver.set_gripper(70.5, 1000), lambda: driver.set_gripper(-1.0, 1000),
                lambda: driver.set_gripper(10.0, 5001), lambda: driver.set_joint_mode(0),
                lambda: driver.set_joint_mode(101), lambda: driver.command_joints([0, 0, 0, 0, 0, np.nan]),
                lambda: driver.command_joints([0, 0, 0])):
        with pytest.raises(ValueError):
            bad()
    assert wire.frames() == []


@pytest.mark.parametrize("wire_driver", [False], indirect=True)
def test_read_only_driver_never_commands(wire_driver):
    wire, driver = wire_driver
    commands = (lambda: driver.set_joint_mode(10), lambda: driver.command_joints([0.0] * 6),
                lambda: driver.set_gripper(10.0, 1000), driver.exit_teaching, driver.take_control,
                driver.clear_motor_faults, lambda: driver.init_gripper(70))
    for command in commands:
        with pytest.raises(MotionDisabledError):
            command()
    assert wire.frames() == []
    driver.state()
    assert wire.frames() == []


def test_silent_bus_reads_as_no_data(wire_driver):
    _, driver = wire_driver
    assert driver.feedback_age_s() == np.inf
    assert driver.joint_feedback_age_s() == np.inf
    assert driver.gripper_feedback_age_s() == np.inf
    state = driver.state()
    assert np.array_equal(state.joints_deg, np.zeros(6))
    assert not state.enabled
    assert "CAN" not in state.ctrl_mode and "TEACHING" not in state.ctrl_mode


def test_ctrl_mode_text_matches_sdk():
    from piper_sdk.piper_msgs.msg_v2.feedback.arm_feedback_status import (
        ArmMsgFeedbackStatusEnum,
    )

    modes = ArmMsgFeedbackStatusEnum.CtrlMode
    assert str(modes.CAN_CTRL) == CTRL_MODE_CAN_TEXT
    assert "TEACHING" in str(modes.TEACHING_MODE)


def test_take_control_sequence(wire_driver):
    """Motion-output role, end drag-teach, enable all motors, joint mode, then a hold at the measured joints.

    On a silent bus the controller mode reads STANDBY (not teaching), so a single mode frame suffices and
    the hold target is the all-zero reading: exactly why the runtime refuses to take control without fresh
    feedback (see test_robot_runtime.py).
    """
    wire, driver = wire_driver
    driver.take_control(settle_s=0.05)
    assert wire.frames() == [
        frame(0x470, [0xFC, 0x00, 0x00, 0x00]),
        frame(0x150, [0x00, 0x00, 0x02]),
        frame(0x471, [0x07, 0x02]),
        frame(0x151, [0x01, 0x01, 20, 0x00]),
        (0x155, struct.pack(">ii", 0, 0)),
        (0x156, struct.pack(">ii", 0, 0)),
        (0x157, struct.pack(">ii", 0, 0)),
    ]


def test_clear_motor_faults_sequence(wire_driver):
    """Clear each disabled driver's error, hold, enable, hold again; still disabled afterwards -> error."""
    wire, driver = wire_driver
    with pytest.raises(RuntimeError, match="remain disabled"):
        driver.clear_motor_faults(settle_s=0.05)
    hold = [(0x155, struct.pack(">ii", 0, 0)), (0x156, struct.pack(">ii", 0, 0)), (0x157, struct.pack(">ii", 0, 0))]
    # 0x475: joint, no zeroing, acceleration setting not applied (default 500 sent), clear error 0xAE
    clears = [frame(0x475, [j, 0x00, 0x00, 0x01, 0xF4, 0xAE]) for j in range(1, 7)]
    assert wire.frames() == [*clears, *hold, frame(0x471, [0x07, 0x02]), *hold]


def test_init_gripper_needs_gripper_feedback(wire_driver):
    """Parameters are written and queried; without gripper feedback the gripper is not enabled."""
    wire, driver = wire_driver
    with pytest.raises(RuntimeError, match="no gripper feedback"):
        driver.init_gripper(70, settle_s=0.05)
    assert wire.frames() == [frame(0x47D, [100, 70, 1]), frame(0x477, [0x04, 0x00, 0x00, 0x00, 0x03])]
