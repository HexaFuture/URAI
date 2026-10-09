"""Automatic approach segments: slow connectors before the user's trajectory, and the joint-space route."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.approach import with_approaches
from urai.backend import SimulationBackend
from urai.trajectory import compile_arm


def test_custom_start_gets_slow_connector_and_delayed_events():
    backend = SimulationBackend(); start = backend.state()
    spec = {'path': {'mode': 'function', 'x': 'x0+.08+.02*s', 'y': 'y0', 'z': 'z0'}, 'speed': '.05',
            'gripper_events': [{'s': 0, 'opening_mm': 50}], 'approach': {'speed_m_s': .02}}
    task = compile_arm(spec, start['left'])
    combined = with_approaches({'left': task}, start, {'left': 'direct'})
    p = combined['left']
    np.testing.assert_allclose(p['xyz'][0], start['left']['xyz'])
    np.testing.assert_allclose(p['user_trajectory']['xyz'][0], task['xyz'][0])
    assert p['task_offset_s'] >= 4
    assert p['gripper_events'][0]['time_s'] == pytest.approx(p['task_offset_s'])
    assert np.all(np.diff(p['time_s']) > 0)
    # The default connector speed (0.8 m/s) is bounded by the 0.1 m/s^2 ramp: 8 cm take 2*sqrt(0.08/0.1) s.
    del spec['approach']
    p = with_approaches({'left': compile_arm(spec, start['left'])}, start, {'left': 'direct'})['left']
    assert p['task_offset_s'] == pytest.approx(2*np.sqrt(.08/.1), rel=1e-6)


def test_dual_tasks_start_together_after_different_approaches():
    b = SimulationBackend(); start = b.state()
    tasks = {a: compile_arm({'path': {'mode': 'function', 'x': f'x0+{offset}+.01*s', 'y': 'y0', 'z': 'z0'}, 'speed': '.03'}, start[a])
             for a, offset in [('left', .02), ('right', .08)]}
    p = with_approaches(tasks, start, {'left': 'direct', 'right': 'direct'})
    assert p['left']['task_offset_s'] == pytest.approx(p['right']['task_offset_s'])
    assert len(p['left']['stop_indices']) >= 2
    for a in p:
        assert np.all(np.diff(p[a]['time_s']) > 0)


def test_lift_candidate_rises_above_both_endpoints():
    b = SimulationBackend(); start = b.state()
    task = compile_arm({'path': {'mode': 'function', 'x': 'x0+.04+.01*s', 'y': 'y0', 'z': 'z0-.02'}, 'speed': '.03'}, start['left'])
    p = with_approaches({'left': task}, start, {'left': 'lift'})['left']
    assert p['approach']['xyz'][:, 2].max() >= start['left']['xyz'][2]+.12-1e-9


def test_simulation_custom_start_does_not_jump_to_first_user_point():
    b = SimulationBackend(); start = b.state()
    task = compile_arm({'path': {'mode': 'function', 'x': 'x0+.05+.01*s', 'y': 'y0', 'z': 'z0'}, 'speed': '.03'}, start['left'])
    p = b.plan({'left': task}, start)
    np.testing.assert_allclose(p['arms']['left']['xyz'][0], start['left']['xyz'])


def test_joint_approach_keeps_current_boundary_pose_and_projects_actual_curve(planner):
    # The all-zero rest pose has j2 and j3 exactly on their bounds.
    model = planner.models['left']
    q0 = np.zeros(6); pose = model.fk_tcp_world(q0)
    state = {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
             'joints_deg': q0.tolist(), 'gripper_mm': 0.}
    task = compile_arm({'path': {'mode': 'function', 'x': 'x0+.15+.01*s', 'y': 'y0', 'z': 'z0-.05'}, 'speed': '.03',
                        'approach': {'speed_m_s': .05}}, state)
    p = planner._joint_approach('left', task, state)
    np.testing.assert_array_equal(p['source_joints_deg'][0], q0)
    np.testing.assert_allclose(p['xyz'][0], state['xyz'])
    np.testing.assert_allclose(p['xyz'][-1], task['xyz'][0], atol=1e-6)
    # Every sample is the TCP of the interpolated joints: a curved path, not the straight endpoint chord.
    for k in (50, 100, 150):
        np.testing.assert_allclose(p['xyz'][k], model.fk_tcp_world(p['source_joints_deg'][k])[:3, 3])
    chord = p['xyz'][0]+np.linspace(0, 1, len(p['xyz']))[:, None]*(p['xyz'][-1]-p['xyz'][0])
    assert np.linalg.norm(p['xyz']-chord, axis=1).max() > .02
    assert p['route'] == 'joint' and np.all(np.diff(p['time_s']) > 0)
    assert np.max(np.linalg.norm(np.diff(p['xyz'], axis=0), axis=1)/np.diff(p['time_s'])) <= .05+1e-9
