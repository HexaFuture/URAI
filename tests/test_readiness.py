"""Robot readiness before and during planning, on the real PiPER-X backend and the kinematic simulator.

The refusals for drag-teach mode, a disabled arm, stale CAN feedback and controller errors need those states on
a real arm; they are covered by tests/hardware/test_backend_on_robot.py.
"""
from __future__ import annotations

import time

import pytest
from test_dispatch import joint_plan, run, single_joint

from urai.backend import RobotStateError
from urai.trajectory import compile_arms

#: A task start beyond the reach of the left arm: every approach route fails, none of them on the task itself.
OUT_OF_REACH = {'left': {'path': {'mode': 'waypoints', 'points': [[1., 0., .1], [1.02, 0., .1]]}, 'speed': .03}}
#: A task start in front of the left arm that needs an approach.
IN_FRONT = {'left': {'path': {'mode': 'waypoints', 'points': [[.30, 0., .10], [.30, .03, .10]]}, 'speed': .03,
                     'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, -90], [1, 180, 0, -90]]}}}


def test_simulated_arms_are_ready(piper_backend):
    piper_backend._fresh()
    for state in piper_backend.state().values():
        assert state['enabled'] and 'CAN' in state['ctrl_mode'] and state['feedback_age_s'] < .5


def test_a_preview_start_is_refused_once_the_arm_or_its_gripper_moved(piper_backend):
    start = piper_backend.state()
    assert run(piper_backend, joint_plan(piper_backend, [0., .3], single_joint(piper_backend, [0., 1.])))[0]['completed']
    with pytest.raises(RobotStateError, match='left moved since preview') as moved:
        piper_backend.validate_start(start)
    assert moved.value.reason == 'moved'
    start = piper_backend.state()
    events = [{'time_s': 0., 'opening_mm': 10.}]
    assert run(piper_backend, joint_plan(piper_backend, [0., .3], single_joint(piper_backend, [0., 0.]), events))[0]['completed']
    time.sleep(.1)
    with pytest.raises(RobotStateError, match='left gripper moved since preview') as moved:
        piper_backend.validate_start(start)
    assert moved.value.reason == 'moved'


def nudge_left_arm(backend, degrees=3.):
    """Command the left arm's first joint ``degrees`` further through its driver, the way a second program on the
    bus would. The simulated arm starts at 20% controller speed: it has moved 0.5 degrees after about 15 ms."""
    driver = backend.runtime.arms['left']
    target = driver.joints()
    target[0] += degrees
    driver.command_joints(target)


def test_an_arm_moving_during_planning_ends_the_route_search_at_once(piper_backend):
    start = piper_backend.state()
    candidates, nudged = [], []

    def progress(**detail):
        if detail['stage'] == 'approach':
            candidates.append(detail['routes'])
        elif detail['stage'] == 'ik' and not nudged:
            nudged.append(True)
            nudge_left_arm(piper_backend)
            time.sleep(.3)

    with pytest.raises(RobotStateError, match='moved since preview') as moved:
        piper_backend.plan(compile_arms(IN_FRONT, start), start, progress=progress)
    assert moved.value.reason == 'moved'
    assert candidates == [{'left': 'direct'}]               # lift and joint routes were never tried


@pytest.mark.parametrize('moves', [False, True])
def test_a_state_change_during_the_last_route_outranks_the_route_failures(piper_backend, moves):
    """The arm starts moving as the joint route begins; its IK (about 0.2 s) fails, and the state check that follows
    the failure reports the moved arm instead of the list of failed routes."""
    start = piper_backend.state()
    candidates = []

    def progress(**detail):
        if detail['stage'] == 'approach':
            candidates.append(detail['routes']['left'])
            if moves and detail['routes']['left'] == 'joint':
                nudge_left_arm(piper_backend)

    if moves:
        with pytest.raises(RobotStateError, match='moved since preview') as failure:
            piper_backend.plan(compile_arms(OUT_OF_REACH, start), start, progress=progress)
        assert 'No feasible' not in str(failure.value)
    else:
        with pytest.raises(ValueError, match='No feasible complete approach') as failure:
            piper_backend.plan(compile_arms(OUT_OF_REACH, start), start, progress=progress)
        assert not isinstance(failure.value, RobotStateError)
    assert candidates == ['direct', 'lift', 'joint']
