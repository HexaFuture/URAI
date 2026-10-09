"""Planned motions executed end to end by the real PiPER-X backend on the kinematic simulator.

Each test plans with the real planner (IK, approach search, retiming, collision sweep) on the example rig and
runs the plan through the real 50 Hz loop. The simulated arms follow the streamed reference; the simulated
grippers stop on an object between the fingers and carry it.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from test_dispatch import Recorder, drift_deg, joint_plan, run

from urai.backend import HOME_HOLD_S, HOME_OPENING_MM
from urai.service import DispatchCancellation, Service
from urai.trajectory import compile_arms

#: A left-arm pose away from home that every check accepts.
AWAY = np.array([20., 40., -40., 10., 20., 10.])
#: The red block of the default scene and a free spot on the table, with top-down grasp yaws the arm reaches.
RED_BLOCK_XY, RED_BLOCK_YAW = (.32, -.215), -75.
FREE_SPOT_XY, FREE_SPOT_YAW = (.30, 0.), -90.


def top_down(xy, yaw_deg, heights, events=(), speed=.05):
    """Draft of one arm through ``heights`` (m) above ``xy`` with the tool pointing straight down."""
    return {'path': {'mode': 'waypoints', 'points': [[xy[0], xy[1], z] for z in heights]}, 'speed': speed,
            'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, yaw_deg], [1, 180, 0, yaw_deg]]},
            'gripper_events': list(events)}


def pick(xy, yaw_deg):
    """Open above ``xy``, descend until the fingertips are 10 mm above the table, grip with a holding check, lift."""
    events = [{'s': 0., 'opening_mm': 60., 'hold_s': .6, 'wait_for_arrival': True},
              {'s': .5, 'opening_mm': 0., 'hold_s': .8, 'wait_for_arrival': True, 'verify': 'holding'}]
    return top_down(xy, yaw_deg, [.10, .0135, .10], events)


def plan_drafts(backend, specs):
    start = backend.state()
    return backend.plan(compile_arms(specs, start), start), start


def test_home_opens_the_gripper_at_the_start_pose_then_returns_to_zero(piper_backend):
    rt = piper_backend.runtime
    assert run(piper_backend, joint_plan(piper_backend, [0., 2.], np.array([np.zeros(6), AWAY])))[0]['completed']
    start = piper_backend.state()
    plan = piper_backend.plan_home(start, ['left'])
    result, recorder = run(piper_backend, plan)
    opened = recorder.first(lambda row: row.commanded == HOME_OPENING_MM)
    departed = recorder.first(lambda row: np.abs(row.joints-AWAY).max() > .5)
    assert result['completed'] and opened is not None
    assert departed.wall >= opened.wall+HOME_HOLD_S       # the jaws open at the start pose before the arm moves
    np.testing.assert_allclose(rt.joints_deg('left'), 0., atol=.7)
    assert rt.gripper_mm('left') == HOME_OPENING_MM
    np.testing.assert_array_equal(rt.joints_deg('right'), np.zeros(6))   # the unselected arm never moved


def test_both_task_segments_of_a_dual_arm_draft_start_together(piper_backend):
    """The arms need different approaches; each holds its task start until both are there, then both set off."""
    specs = {'left': top_down((.30, 0.), -90., [.10, .10]), 'right': top_down((.27, -.59), -90., [.14, .14])}
    specs['left']['path']['points'][1][1] += .06
    specs['right']['path']['points'][1][1] += .06
    plan, start = plan_drafts(piper_backend, specs)
    offsets = {arm: p['task_offset_s'] for arm, p in plan['arms'].items()}
    assert offsets['left'] == offsets['right'] == plan['approach_duration_s'] > 0.
    own = {arm: p['approach']['time_s'][-1] for arm, p in plan['arms'].items()}
    assert abs(own['left']-own['right']) > .1               # one arm reaches its start early and waits there
    result = piper_backend.execute(plan, start, DispatchCancellation(), lambda *args: None)
    assert result['completed']
    entered = next(row['wall_t'] for row in result['trace'] if row['phase'] == 'user')
    departures = {}
    for arm, p in plan['arms'].items():
        task_start = p['curve'](p['task_offset_s'])
        departures[arm] = next(row['wall_t'] for row in result['trace'] if row['phase'] == 'user'
                               and np.abs(np.asarray(row['joints_deg'][arm])-task_start).max() > .3)
        assert departures[arm] >= entered
    assert abs(departures['left']-departures['right']) < .1


def test_a_verified_grip_carries_the_block(piper_backend):
    rt = piper_backend.runtime
    scene = rt.cameras.scene
    resting = scene.pose('red_block')[2, 3]
    plan, _ = plan_drafts(piper_backend, {'left': pick(RED_BLOCK_XY, RED_BLOCK_YAW)})
    result, _ = run(piper_backend, plan)
    assert result['completed']
    assert rt.arms['left'].held_object() == 'red_block'
    assert 1.5 < rt.gripper_mm('left') < 67. and rt.gripper_mm('left') == pytest.approx(40.)
    assert scene.pose('red_block')[2, 3] > resting+.05


def test_an_empty_grip_is_reported_within_three_seconds_and_the_arm_held(piper_backend):
    rt = piper_backend.runtime
    plan, _ = plan_drafts(piper_backend, {'left': pick(FREE_SPOT_XY, FREE_SPOT_YAW)})
    check_at = plan['arms']['left']['gripper_events'][1]['hold_end_time_s']
    recorder = Recorder(rt)
    with pytest.raises(RuntimeError, match='夹爪疑似空抓'):
        run(piper_backend, plan, recorder=recorder)
    failed = time.monotonic()-recorder.began
    checking = recorder.first(lambda row: row.t >= check_at-1e-9)
    assert 3. <= failed-checking.wall <= 3.3
    assert rt.gripper_mm('left') == 0. and rt.arms['left'].held_object() is None
    tcp = piper_backend.models['left'].fk_tcp_world(rt.joints_deg('left'))
    assert tcp[2, 3] < .02                              # still at the grasp height: the lift never started
    assert drift_deg(rt) < .05


def test_the_tracking_guard_stops_and_holds_an_arm_that_falls_behind_its_reference(piper_backend):
    """A controller held at 1% speed leaves the arm more than the default 3 degrees and half a second behind.

    The simulated servo follows any reachable reference exactly one cycle late, which the lag window of the guard
    absorbs even with a 0.01 degree limit; an arm that cannot keep up is what the guard is for.
    """
    rt = piper_backend.runtime
    assert piper_backend.tracking_limit_deg == 3.
    plan, _ = plan_drafts(piper_backend, {'left': top_down((.30, 0.), -90., [.10, .10])})
    plan['motion_limits']['controller_speed_percent'] = 1
    end = plan['arms']['left']['joints_deg'][-1]
    with pytest.raises(RuntimeError, match=r'joint tracking error \d+\.\d deg on j\d exceeded 3 deg'):
        run(piper_backend, plan)
    assert np.abs(rt.joints_deg('left')-end).max() > 1.
    assert drift_deg(rt) < .05


def test_a_cancelled_execution_stops_the_arm_where_it_is(piper_backend):
    rt = piper_backend.runtime
    service = Service(piper_backend)
    service.observe()
    service.set_draft({'observation_id': service.frame.id, 'arms': {'left': top_down((.30, 0.), -90., [.10, .10])}})
    plan = service.preview()
    service.execute(plan['id'])
    deadline = time.monotonic()+30.
    while service.status()['execution']['progress'] < .3:
        assert time.monotonic() < deadline
        time.sleep(.02)
    service.cancel()
    service.worker.join(5)
    assert service.execution['state'] == 'cancelled'
    assert np.abs(rt.joints_deg('left')).max() > 5.     # neither finished nor sent home
    assert drift_deg(rt) < .05
