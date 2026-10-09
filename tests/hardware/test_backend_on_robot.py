"""Backend behaviour that only a real PiPER-X arm can show: readiness refusals and automatic recovery.

The kinematic simulator's arms are always enabled, under CAN control and reporting fresh feedback, so these
states have to be produced on the rig. Each test waits up to a minute for its precondition and prints what the
operator has to do, so run them one at a time with output enabled, for example::

    URAI_HARDWARE_TESTS=1 URAI_HARDWARE_MOTION=1 URAI_CALIBRATION=rig.json \\
        pytest -s tests/hardware/test_backend_on_robot.py -k drag_teach

Before the motion tests: clear the workspace, keep the emergency stop in reach, rest both arms at their home
pose (all joints at zero) unless a test says otherwise, and keep the grippers empty. The left arm is used.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from urai.backend import Cancelled, PiperBackend, RobotStateError
from urai.backend.retiming import joint_curve
from urai.service import DispatchCancellation, Service
from urai.settings import motion_limits
from urai.trajectory import compile_arms

pytestmark = pytest.mark.hardware

PRECONDITION_TIMEOUT_S = 60.


def wait_for(predicate, instruction, timeout_s=PRECONDITION_TIMEOUT_S):
    """Print ``instruction`` for the operator and wait until ``predicate()`` holds; fail when it never does."""
    print(f'\n[operator] {instruction} (waiting up to {timeout_s:.0f} s)', flush=True)
    deadline = time.monotonic()+timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.05)
    pytest.fail(f'precondition not met: {instruction}')


def mode(runtime, arm='left'):
    return str(runtime.arms[arm].state().ctrl_mode)


def short_draft(backend):
    """A 2 cm forward stroke of the left arm from where it stands, keeping its orientation."""
    start = backend.state()
    xyz = np.asarray(start['left']['xyz'])
    spec = {'left': {'path': {'mode': 'waypoints', 'points': [xyz.tolist(), (xyz+[.02, 0., 0.]).tolist()]}}}
    return compile_arms(spec, start), start


def test_drag_teach_mode_is_reported_and_blocks_planning_before_any_ik(open_robot_runtime):
    """Operator: press the left arm's teach button (drag-teach mode) and leave the arm alone.

    Read-only: the runtime starts without motion permission, so nothing takes the arm out of teaching.
    """
    with open_robot_runtime(allow_motion=False) as runtime:
        backend = PiperBackend(runtime)
        wait_for(lambda: 'TEACHING' in mode(runtime), 'switch the LEFT arm into drag-teach mode')
        with pytest.raises(RobotStateError, match='示教模式') as refused:
            backend._fresh()
        assert refused.value.reason == 'teaching'
        compiled, start = short_draft(backend)
        stages = []
        with pytest.raises(RobotStateError, match='示教模式'):
            backend.plan(compiled, start, progress=lambda **detail: stages.append(detail['stage']))
        assert 'ik' not in stages
        assert 'TEACHING' in mode(runtime)


def test_stale_can_feedback_is_refused_and_never_used_for_a_hold(open_robot_runtime):
    """Operator: once the test is waiting, unplug the left arm's CAN adapter (or switch the left arm off).

    Read-only. Afterwards reconnect the adapter and restart anything that uses the bus.
    """
    with open_robot_runtime(allow_motion=False) as runtime:
        backend = PiperBackend(runtime)
        backend._fresh()
        driver = runtime.arms['left']
        wait_for(lambda: not driver.feedback_age_s() <= .5, 'disconnect the LEFT arm from the CAN bus')
        with pytest.raises(RobotStateError, match='CAN 反馈已过期|反馈时间无效'):
            backend._fresh()
        assert any('stale feedback' in error for error in backend._hold(['left']))


def test_an_arm_outside_can_control_is_reported(open_robot_runtime):
    """Operator: switch the left arm off and on again (it comes up in standby, not under CAN control), then run
    this test before anything else takes control of it. Read-only."""
    with open_robot_runtime(allow_motion=False) as runtime:
        backend = PiperBackend(runtime)
        wait_for(lambda: runtime.arms['left'].feedback_age_s() <= .5 and 'CAN' not in mode(runtime)
                 and 'TEACHING' not in mode(runtime), 'power-cycle the LEFT arm and leave it in standby')
        with pytest.raises(RobotStateError, match='当前控制模式为'):
            backend._fresh()


def test_a_controller_error_is_reported(open_robot_runtime):
    """Operator: bring the left arm into a controller-reported fault, for example by pressing its emergency stop
    while it rests at home; release and reset it after the test."""
    with open_robot_runtime(allow_motion=False) as runtime:
        backend = PiperBackend(runtime)
        driver = runtime.arms['left']
        wait_for(lambda: driver.state().arm_status != 0 or driver.state().err_code != 0,
                 'make the LEFT arm report a controller fault (arm_status or err_code != 0)')
        with pytest.raises(RobotStateError, match='控制器报错|未使能'):
            backend._fresh()


@pytest.mark.motion
def test_prepare_restores_can_control_after_drag_teaching_without_moving_the_arm(open_robot_runtime):
    """Operator: after start-up, switch the left arm into drag-teach mode, move it a little by hand, let go and
    keep clear. The backend waits for one second of stillness, leaves teaching mode and holds the measured pose."""
    with open_robot_runtime(allow_motion=True) as runtime:
        backend = PiperBackend(runtime)
        wait_for(lambda: 'TEACHING' in mode(runtime), 'switch the LEFT arm into drag-teach mode, move it, let go')
        time.sleep(2.)
        taught = runtime.joints_deg('left')
        stages = []
        result = backend.prepare(DispatchCancellation(), lambda **detail: stages.append(detail['stage']),
                                 restore_can=True)
        assert result == {'restored_arms': ['left']}
        assert 'settle' in stages and 'restore' in stages
        assert 'CAN' in mode(runtime) and np.abs(runtime.joints_deg('left')-taught).max() < .5
        backend._fresh()


@pytest.mark.motion
def test_a_preview_that_restored_control_refreshes_the_observation(open_robot_runtime):
    """Operator: once the test waits, switch the left arm into drag-teach mode, let go and keep clear.

    The arm does not move, yet the preview takes a new observation because control was restored before planning.
    Nothing is executed.
    """
    with open_robot_runtime(allow_motion=True) as runtime:
        backend = PiperBackend(runtime)
        service = Service(backend)
        service.observe()
        xyz = np.asarray(backend.state()['left']['xyz'])
        stroke = {'mode': 'waypoints', 'points': [xyz.tolist(), (xyz+[.02, 0., 0.]).tolist()]}
        service.set_draft({'observation_id': service.frame.id, 'arms': {'left': {'path': stroke}}})
        observed = service.frame.id
        wait_for(lambda: 'TEACHING' in mode(runtime), 'switch the LEFT arm into drag-teach mode and let go')
        plan = service.preview()
        assert plan['preparation']['restored_arms'] == ['left'] and plan['preparation']['observation_refreshed']
        assert service.frame.id != observed and 'CAN' in mode(runtime)


@pytest.mark.motion
def test_an_arm_still_being_dragged_is_never_taken_over(open_robot_runtime):
    """Operator: switch the left arm into drag-teach mode and keep moving it slowly for 15 seconds."""
    with open_robot_runtime(allow_motion=True) as runtime:
        backend = PiperBackend(runtime)
        wait_for(lambda: 'TEACHING' in mode(runtime), 'switch the LEFT arm into drag-teach mode and keep moving it')
        with pytest.raises(RobotStateError, match='仍在移动'):
            backend.prepare(DispatchCancellation(), lambda **detail: None, restore_can=True)
        assert 'TEACHING' in mode(runtime)


@pytest.mark.motion
def test_cancel_or_restore_can_off_leaves_drag_teach_mode_alone(open_robot_runtime):
    """Operator: switch the left arm into drag-teach mode, let go and keep clear."""
    with open_robot_runtime(allow_motion=True) as runtime:
        backend = PiperBackend(runtime)
        wait_for(lambda: 'TEACHING' in mode(runtime), 'switch the LEFT arm into drag-teach mode and let go')
        with pytest.raises(RobotStateError, match='示教'):
            backend.prepare(DispatchCancellation(), lambda **detail: None, restore_can=False)
        cancel = DispatchCancellation()
        threading.Timer(.3, cancel.set).start()
        with pytest.raises(Cancelled):
            backend.prepare(cancel, lambda **detail: None, restore_can=True)
        assert 'TEACHING' in mode(runtime)


@pytest.mark.motion
def test_home_preparation_re_enables_a_disabled_arm_but_a_preview_does_not(open_robot_runtime):
    """Operator: rest the left arm at its home pose (all joints at zero) and keep a hand near it.

    The test disables the left arm's motors itself (piper_sdk ``DisableArm``), the state a protection stop
    leaves behind. A preview's preparation refuses the disabled arm; a home's preparation re-enables it holding
    the measured joints.
    """
    with open_robot_runtime(allow_motion=True) as runtime:
        backend = PiperBackend(runtime)
        driver = runtime.arms['left']
        assert np.abs(runtime.joints_deg('left')).max() < 2., 'rest the left arm at home first'
        driver.piper.DisableArm(7, 0x01)
        wait_for(lambda: not driver.state().enabled, 'nothing to do: the left motors are being disabled', 5.)
        rested = runtime.joints_deg('left')
        with pytest.raises(RobotStateError, match='未使能'):
            backend.prepare(DispatchCancellation(), lambda **detail: None)
        result = backend.prepare(DispatchCancellation(), lambda **detail: None, restore_disabled=True)
        assert result == {'restored_arms': ['left']}
        assert driver.state().enabled and np.abs(runtime.joints_deg('left')-rested).max() < .5


@pytest.mark.motion
def test_a_disabled_arm_with_a_controller_fault_is_never_enabled(open_robot_runtime):
    """Operator: after start-up, press the left arm's emergency stop while it rests at home, so that it is disabled
    and reports a fault; release and reset it after the test."""
    with open_robot_runtime(allow_motion=True) as runtime:
        backend = PiperBackend(runtime)
        driver = runtime.arms['left']

        def faulted():
            state = driver.state()
            return not state.enabled and (state.arm_status != 0 or state.err_code != 0)

        wait_for(faulted, 'make the LEFT arm disabled with a controller fault (press its emergency stop)')
        with pytest.raises(RobotStateError, match='控制器报错'):
            backend.prepare(DispatchCancellation(), lambda **detail: None, restore_disabled=True)
        assert not driver.state().enabled


@pytest.mark.motion
def test_late_control_feedback_is_held_and_never_dispatched_from_stale_geometry(open_robot_runtime):
    """No operator action: the test loads the CPU core the control loop runs on, then moves joint 6 of the left
    arm 5 degrees out and back.

    Feedback validated later than 80 ms is held and checked again; a second late sample, or a late sample in a
    release phase, stops the motion. Either the motion completes, or it stops with a deadline or stall error and
    the arm is held where it stopped.
    """
    with open_robot_runtime(allow_motion=True) as runtime:
        backend = PiperBackend(runtime)
        model = runtime.models['left']
        start = backend.state()
        q0 = np.asarray(start['left']['joints_deg'])
        step = 5. if q0[5]+5. < model.kin.limits_deg[6][1]-1. else -5.
        knots = np.tile(q0, (3, 1))
        knots[1, 5] += step
        times = np.array([0., 2., 4.])
        bounds = np.array([model.kin.limits_deg[i] for i in range(1, 7)])
        plan = {'duration_s': 4., 'approach_duration_s': 0., 'motion_limits': motion_limits('fine'),
                'arms': {'left': {'curve': joint_curve(times, knots, (1,), bounds), 'time_s': times,
                                  'joints_deg': knots, 'gripper_events': []}}}
        cores = os.sched_getaffinity(0)
        core = min(cores)
        os.sched_setaffinity(0, {core})
        hogs = [subprocess.Popen([sys.executable, '-c', f'import os; os.sched_setaffinity(0, {{{core}}})\nwhile True: pass'])
                for _ in range(4)]
        try:
            result = backend.execute(plan, start, DispatchCancellation(), lambda *args: None)
            assert result['completed'] and result['feedback_retries'] <= 3
        except RuntimeError as exc:
            assert 'control deadline' in str(exc) or 'scheduler stalled' in str(exc), exc
            stopped = runtime.joints_deg('left')
            time.sleep(.5)
            assert np.abs(runtime.joints_deg('left')-stopped).max() < .2
        finally:
            for hog in hogs:
                hog.kill()
                hog.wait()
            os.sched_setaffinity(0, cores)
