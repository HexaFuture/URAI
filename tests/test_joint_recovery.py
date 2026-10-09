"""Recovery from a taught start just outside a nominal joint bound: inward only, and only before the task."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.backend.model_checks import (
    model_joint_limit_reason,
    recovery_joint_bounds,
    solve_ik,
)
from urai.backend.retiming import joint_curve, retiming_scale
from urai.settings import motion_limits
from urai.trajectory import compile_arms

#: A hand-guided pose with j3 1.26 degrees above its 0 degree upper bound, as drag teaching leaves it.
TAUGHT = np.array([0., 10., 1.26, 0., 20., 0.])


def test_recovery_bounds_admit_only_the_initial_inward_prefix(arm_models):
    model = arm_models['left']
    qs = np.zeros((4, 6)); qs[:, 1] = 30.; qs[:, 2] = [1.26, .6, 0., -1.]
    limits = recovery_joint_bounds(model, qs, task_index=2)
    assert limits[2, 1] == 1.26 and limits[2, 0] == model.kin.limits_deg[3][0]
    for values, task_index in [([1.26, 1.3, 0., -1.], 2),   # moves outward first
                               ([1.26, 0., .5, -1.], 3),    # leaves the bounds again after entering
                               ([1.26, .6, .2, -1.], 2)]:   # still outside at the task start
        qs[:, 2] = values
        with pytest.raises(ValueError, match='inward'):
            recovery_joint_bounds(model, qs, task_index=task_index)


def test_recovery_interpolation_and_retiming_stay_within_initial_pose_envelope(planner):
    model = planner.models['left']
    qs = np.zeros((5, 6)); qs[:, 1] = 30.; qs[:, 2] = [1.26, 1., .4, 0., -1.]
    bounds = recovery_joint_bounds(model, qs, task_index=3)
    times = np.array([0., .1, .2, .3, .7])
    p = {'time_s': times, 'joints_deg': qs, 'task_offset_s': .3, 'stop_indices': [3], 'recovery_joint_bounds': bounds}
    raw, timed = planner._local_retime({'left': p}, motion_limits())
    curve = joint_curve(np.interp(times, raw, timed), qs, [3], bounds)
    for i in range(4):
        sampled = curve(np.linspace(timed[i], timed[i+1], 101))[:, 2]
        assert sampled.min() >= qs[i+1, 2]-1e-8 and sampled.max() <= qs[i, 2]+1e-8
    assert retiming_scale(curve, motion_limits()) <= 1.020001


def arm_state(model, q):
    pose = model.fk_tcp_world(np.asarray(q, dtype=float))
    return {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            'joints_deg': list(map(float, q)), 'gripper_mm': 70.}


def tool_down_joints(model, xyz):
    target = np.eye(4); target[:3, :3] = Rotation.from_euler('xyz', [180., 0., 0.], degrees=True).as_matrix(); target[:3, 3] = xyz
    q, pe, _ = solve_ik(model, target, np.array([0., 80., -80., 0., 0., 0.]))
    assert pe < 1e-6
    return q


@pytest.mark.parametrize('taught_arm', ['left', 'right'])
def test_full_plan_accepts_active_or_inactive_taught_start(planner, taught_arm):
    models = planner.models
    assert model_joint_limit_reason(models[taught_arm], TAUGHT) is not None
    start = {'left': arm_state(models['left'], TAUGHT if taught_arm == 'left' else tool_down_joints(models['left'], [.35, .05, .10])),
             'right': arm_state(models['right'], TAUGHT if taught_arm == 'right' else tool_down_joints(models['right'], [.35, -.64, .10]))}
    task = {'left': {'path': {'mode': 'waypoints', 'points': [[.38, .05, .10], [.42, .05, .08]]}, 'speed': .03,
                     'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 180, 0, 0]]}}}
    plan = planner.plan(compile_arms(task, start), start, motion_profile='normal')
    arm = plan['arms']['left']
    # Only the taught arm itself is forced onto the joint-space route; the other arm approaches directly.
    assert plan['approach_routes'] == {'left': 'joint' if taught_arm == 'left' else 'direct'}
    np.testing.assert_array_equal(arm['joints_deg'][0], start['left']['joints_deg'])
    assert set(plan['initial_joint_bounds']) == {taught_arm}
    assert plan['initial_joint_bounds'][taught_arm][2, 1] == 1.26
    grid = np.linspace(arm['task_offset_s'], plan['duration_s'], 100)
    assert arm['curve'](grid)[:, 2].max() <= 1e-8
    if taught_arm == 'left':
        outside = np.maximum(arm['joints_deg'][:, 2], 0.)
        assert np.all(np.diff(outside) <= 1e-9)
        assert arm['recovery_joint_bounds'][2, 1] == 1.26
        assert any('从实测关节位置平滑返回' in note for note in plan['notes'])
