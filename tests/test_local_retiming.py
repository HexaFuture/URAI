"""Local retiming: only the intervals that exceed a joint limit are stretched, on one clock shared by both arms."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.backend.model_checks import solve_ik
from urai.backend.retiming import joint_curve, retiming_scale
from urai.settings import motion_limits
from urai.trajectory import compile_arms


def stop_and_go():
    """j1 keeps turning at 5 deg/s while j5 swings 15 degrees in 1.5 s and then stops."""
    t = np.linspace(0, 4, 201); q = np.zeros((len(t), 6))
    q[:, 0] = 5*t; q[:, 1] = 40.; q[:, 2] = -40.; q[:, 4] = -np.minimum(10*t, 15)
    return t, q


def test_a_local_joint_stop_does_not_slow_the_entire_path(planner):
    t, q = stop_and_go()
    p = {'time_s': t, 'joints_deg': q, 'orientation_mode': 'hold', 'stop_indices': [], 'task_offset_s': 0.}
    limits = motion_limits('normal')
    uniform = t[-1]*retiming_scale(joint_curve(t, q), limits)
    raw, timed = planner._local_retime({'left': p}, limits)
    new_t = np.interp(t, raw, timed)
    assert new_t[-1] < uniform*.7
    assert retiming_scale(joint_curve(new_t, q), limits) <= 1.020001
    assert np.all(np.diff(new_t) >= np.diff(t)-1e-10)


def test_shared_local_clock_keeps_all_arm_knots_and_stop_boundary(planner):
    t, q = stop_and_go()
    arms = {'left': {'time_s': t, 'joints_deg': q, 'stop_indices': [50], 'task_offset_s': 1.},
            'right': {'time_s': t[::2], 'joints_deg': q[::2]*.5, 'stop_indices': [25], 'task_offset_s': 1.}}
    raw, timed = planner._local_retime(arms, motion_limits())
    assert 1. in raw
    assert np.all(np.diff(timed) > 0)
    for arm in arms.values():
        times = np.interp(arm['time_s'], raw, timed)
        c = joint_curve(times, arm['joints_deg'], arm['stop_indices'])
        np.testing.assert_allclose(c(np.interp(1., raw, timed), 1), 0, atol=1e-7)
        assert retiming_scale(c, motion_limits()) <= 1.020001


def arm_state(model, xyz):
    target = np.eye(4); target[:3, :3] = Rotation.from_euler('xyz', [180., 0., 0.], degrees=True).as_matrix(); target[:3, 3] = xyz
    q, pe, _ = solve_ik(model, target, np.array([0., 80., -80., 0., 0., 0.]))
    assert pe < 1e-6
    pose = model.fk_tcp_world(q)
    return {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            'joints_deg': q.tolist(), 'gripper_mm': 0.}


@pytest.mark.parametrize('offset', [0., 1.])
def test_off_knot_events_follow_their_own_arm_progress_after_local_retiming(planner, offset):
    """Both arms share one warped clock, but an event between two knots keeps its place on its own arm's path.

    The left path creeps for its first half and then speeds up; the fine profile stretches some intervals of the
    shared clock far more than others. With ``offset`` both arms first approach their task starts.
    """
    fraction = (99+.5)/200
    left, right = [.35, .05, .10], [.35, -.64, .10]
    start = {'left': arm_state(planner.models['left'], [left[0]-.03*offset, left[1], left[2]]),
             'right': arm_state(planner.models['right'], [right[0]-.03*offset, right[1], right[2]])}
    specs = {'left': {'path': {'mode': 'waypoints', 'points': [left, [.45, .05, .10], [.45, .15, .10]]},
                      'speed': [[0, .02], [.45, .02], [.5, .15], [1, .15]],
                      'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 180, 0, 0]]},
                      'gripper_events': [{'s': fraction, 'opening_mm': 40.}]},
             'right': {'path': {'mode': 'waypoints', 'points': [right, [.42, -.60, .08]]}, 'speed': .08,
                       'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 180, 0, -40]]},
                       'gripper_events': [{'s': fraction, 'opening_mm': 40.}]}}
    plan = planner.plan(compile_arms(specs, start), start, motion_profile='fine')
    assert plan['retiming_factor'] > 1.5
    assert plan['approach_duration_s'] == (0. if offset == 0. else pytest.approx(plan['arms']['left']['task_offset_s']))
    for arm in plan['arms'].values():
        task = arm.get('user_trajectory', arm)
        assert np.min(np.abs(task['s']-fraction)) > 1e-3   # the event lies strictly between two knots
        expected = float(np.interp(fraction, task['s'], task['time_s']))
        assert arm['gripper_events'][0]['time_s'] == pytest.approx(expected, abs=1e-8, rel=0)
        assert task['gripper_events'][0]['time_s'] == pytest.approx(expected, abs=1e-8, rel=0)
