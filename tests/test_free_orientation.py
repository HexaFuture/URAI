"""Free orientation: IK on the TCP position only, the solved RPY shown in the preview and used for retiming."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.approach import with_approaches
from urai.backend.model_checks import solve_ik
from urai.backend.retiming import joint_axes, retiming_scale
from urai.trajectory import MAX_ANGULAR_SPEED, compile_arm

SEED = np.array([0., 80., -80., 0., 0., 0.])


def pose(xyz, rpy=(180., 0., 0.)):
    target = np.eye(4); target[:3, :3] = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix(); target[:3, 3] = xyz
    return target


def arm_state(model, q):
    p = model.fk_tcp_world(np.asarray(q, dtype=float))
    return {'xyz': p[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(p[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            'joints_deg': list(map(float, q)), 'gripper_mm': 0.}


def tool_down_state(model, xyz):
    q, pe, _ = solve_ik(model, pose(xyz), SEED)
    assert pe < 1e-6
    return arm_state(model, q)


def angle_deg(a, b):
    return float(np.rad2deg((Rotation.from_matrix(a).inv()*Rotation.from_matrix(b)).magnitude()))


def test_position_ik_changes_rpy_and_solves_actual_tcp_with_tool_offset(arm_models):
    model = arm_models['left']
    target = model.fk_tcp_world(np.array([20., 90., -70., 10., 10., 30.]))
    target[:3, :3] = np.eye(3)   # an orientation the free solve is not asked to reach
    q, pe, re = solve_ik(model, target, np.array([0., 80., -60., 0., 30., 0.]), {'position_mm': 1, 'orientation_deg': 0},
                         orientation_free=True)
    assert pe < .1 and re > 10
    # The TCP, 142.5 mm out along the tool axis, is what lands on the target, not the link6 flange.
    assert np.linalg.norm(model.fk_tcp_world(q)[:3, 3]-target[:3, 3])*1000 < .1
    assert np.linalg.norm(model.t_world_base[:3, :3] @ model.kin.fk(q)[:3, 3]/1000.+model.t_world_base[:3, 3]-target[:3, 3]) > .1
    lower = np.array([model.kin.limits_deg[i][0] for i in range(1, 7)])
    upper = np.array([model.kin.limits_deg[i][1] for i in range(1, 7)])
    assert np.all(q >= lower) and np.all(q <= upper)
    q2, _, _ = solve_ik(model, target, q, orientation_free=True)
    assert np.max(np.abs(q2-q)) < .05


def test_free_mode_survives_compilation_and_approach_and_ignores_rotation_tolerance(planner):
    models = planner.models
    start = {'left': tool_down_state(models['left'], [.35, .05, .10]), 'right': tool_down_state(models['right'], [.35, -.64, .10])}
    goal = np.array([.45, .15, .15])
    spec = {'path': {'mode': 'waypoints', 'points': [goal*.9+np.asarray(start['left']['xyz'])*.1, goal]},
            'speed': '.03', 'orientation': {'mode': 'free'}}
    task = compile_arm(spec, start['left'])
    assert task['orientation_mode'] == 'free'
    compiled = with_approaches({'left': task}, start, {'left': 'direct'})
    assert compiled['left']['orientation_mode'] == 'free'
    assert compiled['left']['approach']['orientation_mode'] == 'free'
    plan = planner._plan(compiled, start, {'position_mm': 1, 'orientation_deg': 0})
    arm = plan['arms']['left']
    # The tool turned away from the placeholder (start) orientation although the rotation tolerance is 0 degrees.
    assert angle_deg(task['rotation_matrices'][0], arm['rotation_matrices'][-1]) > 10
    np.testing.assert_allclose(arm['user_trajectory']['rotation_matrices'][-1], arm['rotation_matrices'][-1])
    for q, rotation in zip(arm['joints_deg'], arm['rotation_matrices']):
        np.testing.assert_allclose(models['left'].fk_tcp_world(q)[:3, :3], rotation, atol=1e-12)
    assert retiming_scale(arm['curve'], plan['motion_limits']) < 1.021
    assert any('姿态自由' in note for note in plan['notes'])


def test_free_joint_approach_does_not_require_the_placeholder_orientation(planner):
    start = tool_down_state(planner.models['left'], [.35, .05, .10])
    target = [.25, 0., .20]   # tool down is out of reach here: j4 would leave its bound
    path = {'mode': 'waypoints', 'points': [target, [.251, 0., .20]]}
    free = compile_arm({'path': path, 'orientation': {'mode': 'free'}}, start)
    approach = planner._joint_approach('left', free, start, {'position_mm': 1, 'orientation_deg': 0})
    assert angle_deg(free['rotation_matrices'][0], approach['rotation_matrices'][-1]) > 5
    assert np.linalg.norm(approach['xyz'][-1]-target)*1000 < 1
    held = compile_arm({'path': path}, start)
    with pytest.raises(ValueError, match='approach: pose error'):
        planner._joint_approach('left', held, start, {'position_mm': 1, 'orientation_deg': 0})


def test_free_rotation_is_retimed_using_the_solved_orientation(planner):
    """An arc around the left base turns j1, and with it the tool, faster than the 30 deg/s TCP limit.

    The requested RPY of a free path is a constant placeholder, so the draft's own timing sees no rotation; the
    throw profile's joint limits are far above this motion. Only the solved orientation can slow it down.
    """
    model = planner.models['left']
    xyz = [.3*math.cos(-.6), .3*math.sin(-.6), .12]
    q, pe, _ = solve_ik(model, pose(xyz), SEED, orientation_free=True)
    assert pe < .01
    start = {'left': arm_state(model, q), 'right': tool_down_state(planner.models['right'], [.35, -.64, .10])}
    task = compile_arm({'path': {'mode': 'function', 'x': '.3*cos(1.2*s-.6)', 'y': '.3*sin(1.2*s-.6)', 'z': '.12'},
                        'speed': '.2', 'orientation': {'mode': 'free'}, 'gripper_events': [{'s': 1, 'opening_mm': 30}]},
                       start['left'])
    plan = planner._plan({'left': task}, start, motion_profile='throw')
    assert plan['retiming_factor'] > 1.02
    curve = plan['arms']['left']['curve']
    times = np.linspace(0, plan['duration_s'], 20001)
    speeds = np.linalg.norm((joint_axes(model.kin, curve(times)) @ curve(times, 1)[:, :, None])[:, :, 0], axis=1)
    assert .9*MAX_ANGULAR_SPEED < speeds.max() <= MAX_ANGULAR_SPEED
    assert plan['arms']['left']['gripper_events'][0]['time_s'] == plan['duration_s']
