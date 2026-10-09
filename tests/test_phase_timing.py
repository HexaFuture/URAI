"""Approach and task phases are retimed separately; events keep their phase and both tasks start together."""
import numpy as np
from scipy.spatial.transform import Rotation

from urai.backend.model_checks import solve_ik
from urai.backend.retiming import retiming_scale
from urai.trajectory import compile_arms


def tool_down_state(model, xyz):
    target = np.eye(4); target[:3, :3] = Rotation.from_euler('xyz', [180., 0., 0.], degrees=True).as_matrix(); target[:3, 3] = xyz
    q, pe, _ = solve_ik(model, target, np.array([0., 80., -80., 0., 0., 0.]))
    assert pe < 1e-6
    pose = model.fk_tcp_world(q)
    return {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            'joints_deg': q.tolist(), 'gripper_mm': 0.}


def test_phase_retiming_keeps_dual_start_events_and_derivative_bounds(planner):
    start = {'left': tool_down_state(planner.models['left'], [.35, .05, .10]),
             'right': tool_down_state(planner.models['right'], [.35, -.64, .10])}
    specs = {a: {'path': {'mode': 'function', 'x': f'x0+{d}+.05*s', 'y': 'y0+.01*sin(4*pi*s)', 'z': 'z0'}, 'speed': '.1',
                 'gripper_events': [{'s': 0, 'opening_mm': 30}, {'s': .3473, 'opening_mm': 20}, {'s': 1, 'opening_mm': 0}]}
             for a, d in [('left', .02), ('right', .08)]}
    plan = planner.plan(compile_arms(specs, start), start, motion_profile='normal')
    assert plan['approach_routes'] == {'left': 'direct', 'right': 'direct'}
    assert plan['retiming_factor'] > 1.03   # the joint limits did stretch the clock
    offsets = []
    for arm in plan['arms'].values():
        offsets.append(arm['task_offset_s'])
        assert arm['gripper_events'][0]['time_s'] == arm['task_offset_s']
        assert arm['gripper_events'][-1]['time_s'] == arm['time_s'][-1]
        user = arm.get('user_trajectory', arm)
        for event in arm['gripper_events']:
            assert event['time_s'] == float(np.interp(event['s'], user['s'], user['time_s']))
        assert user['gripper_events'] == arm['gripper_events']
        np.testing.assert_allclose(arm['curve'](arm['time_s'][arm['stop_indices']], 1), 0, atol=1e-8)
        assert retiming_scale(arm['curve'], plan['motion_limits']) <= 1.021
    assert offsets[0] == offsets[1] == plan['approach_duration_s']
