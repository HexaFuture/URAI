"""Return to home: the joint-space plan to the all-zero pose and the service's home operation.

The service tests run on the synthetic simulation backend, which follows any plan ideally and returns to its own
initial pose; the plan tests use the real PiPER-X planner on the example rig.
"""
import json
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_dispatch import state_from_joints

from urai.backend import HOME_HOLD_S, HOME_OPENING_MM, SimulationBackend
from urai.service import DispatchCancellation, Service
from urai.trajectory import SAMPLES, compile_arms

INITIAL_LEFT = [.25, .12, .24]
DISPLACED_LEFT = [.30, .05, .18]
#: A left-arm pose away from home that every check accepts.
AWAY = np.array([20., 40., -40., 10., 20., 10.])


def displaced_service(log_dir=None):
    """A service whose simulated left arm has executed a move away from its initial pose, tool tilted."""
    service = Service(SimulationBackend(), log_dir)
    backend = service.backend
    start = backend.state()
    spec = {'path': {'mode': 'waypoints', 'points': [start['left']['xyz'], DISPLACED_LEFT]}, 'speed': .2,
            'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 170, 5, 10]]}}
    plan = backend.plan(compile_arms({'left': spec}, start), start)
    assert backend.execute(plan, start, DispatchCancellation(), lambda *args: None)['completed']
    np.testing.assert_allclose(backend.positions['left'], DISPLACED_LEFT, atol=1e-9)
    service.observe()
    return service


def test_simulation_home_moves_the_selected_arm_back_to_its_initial_pose(tmp_path):
    service = displaced_service(tmp_path)
    right_before = service.backend.positions['right'].copy()
    execution = service.home(['left'])
    assert execution['state'] == 'running' and execution['kind'] == 'home' and execution['arms'] == ['left']
    service.worker.join(30)
    assert service.status()['execution']['state'] == 'completed'
    np.testing.assert_allclose(service.backend.positions['left'], INITIAL_LEFT, atol=1e-6)
    expected = Rotation.from_euler('xyz', [180., 0., 0.], degrees=True).as_matrix()
    actual = Rotation.from_euler('xyz', service.backend.rotations['left'], degrees=True).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(service.backend.positions['right'], right_before)
    assert service.backend.openings['left'] == HOME_OPENING_MM
    record = json.loads((tmp_path / (execution['id'] + '.json')).read_text())
    assert record['preview']['kind'] == 'home' and record['execution']['state'] == 'completed'
    assert 'verify' not in record['preview']['arms']['left']['gripper_events'][0]


def test_home_api_defaults_to_both_arms_and_rejects_bad_requests(client_for):
    service = displaced_service()
    client = client_for(service)
    assert client.post('/api/home', json={'arms': ['up']}).status_code == 409
    assert client.post('/api/home', json={'arms': []}).status_code == 409
    assert client.post('/api/home', json={'arms': 'left'}).status_code == 409
    assert client.post('/api/home', json={'cancel_epoch': service.cancel_epoch + 1}).status_code == 409
    assert service.execution['state'] == 'idle'
    response = client.post('/api/home', json={})
    assert response.status_code == 200, response.text
    assert response.json()['state'] == 'running'
    assert response.json()['arms'] == ['left', 'right']
    service.worker.join(30)
    assert service.execution['state'] == 'completed'
    np.testing.assert_allclose(service.backend.positions['left'], INITIAL_LEFT, atol=1e-6)
    np.testing.assert_allclose(service.backend.positions['right'], [.25, -.45, .24], atol=1e-6)


def test_home_can_be_stopped_and_holds_position():
    service = displaced_service()
    service.home()
    time.sleep(.2)
    service.cancel()
    service.worker.join(5)
    assert service.execution['state'] == 'cancelled'
    assert service.backend.hold_count == 1
    assert np.linalg.norm(service.backend.positions['left'] - INITIAL_LEFT) > .01


def test_home_refuses_while_running_or_after_cancellation():
    service = displaced_service()
    service.home(['left'])
    running = service.worker
    with pytest.raises(ValueError, match='running'):
        service.home()
    running.join(30)
    assert service.execution['state'] == 'completed'
    with pytest.raises(ValueError, match='取消'):
        service.home(cancel_epoch=service.cancel_epoch + 1)
    assert service.worker is running and service.execution['state'] == 'completed'


def test_home_records_its_automatic_preparation(tmp_path):
    service = displaced_service(tmp_path)
    service.set_settings({'auto_prepare': {'restore_can': False}})
    execution = service.home(['left'])
    service.worker.join(30)
    record = json.loads((tmp_path / (execution['id'] + '.json')).read_text())
    assert record['preview']['preparation'] == {'restored_arms': []}
    assert record['preview']['auto_prepare'] == {'refresh_observation': True, 'restore_can': False}
    assert not any('恢复 CAN' in note for note in record['preview']['notes'])


def test_plan_home_dwells_open_at_the_start_pose_then_follows_the_joint_line(planner, arm_models):
    start = state_from_joints(arm_models, {'left': AWAY, 'right': np.zeros(6)})
    plan = planner.plan_home(start, ['left'])
    assert set(plan['arms']) == {'left'}
    p = plan['arms']['left']
    s = np.linspace(0, 1, SAMPLES)
    line = AWAY[None, :]*(1-s)[:, None]
    line[-1] = 0.
    np.testing.assert_array_equal(p['joints_deg'], np.vstack([line[:1], line]))   # exactly the line: no IK ran
    [event] = p['gripper_events']
    assert event['s'] == 0 and event['opening_mm'] == HOME_OPENING_MM and event['wait_for_arrival']
    assert 'verify' not in event and event['hold_s'] == pytest.approx(HOME_HOLD_S)
    assert event['time_s'] == pytest.approx(p['time_s'][0])
    assert event['hold_end_time_s'] - event['time_s'] >= HOME_HOLD_S - 1e-9
    for t in np.linspace(event['time_s'], event['hold_end_time_s'], 5):
        np.testing.assert_allclose(p['curve'](t), AWAY, atol=1e-8)
    np.testing.assert_allclose(p['xyz'][0], start['left']['xyz'])
    np.testing.assert_allclose(p['xyz'][-1], arm_models['left'].fk_tcp_world(np.zeros(6))[:3, 3], atol=1e-12)
    assert plan['approach_duration_s'] == 0
    assert plan['duration_s'] > event['hold_end_time_s']
    assert any('回原位' in note for note in plan['notes'])
    assert set(plan) >= {'duration_s', 'arms', 'retiming_factor', 'notes', 'pose_tolerance', 'motion_limits'}


def test_plan_home_without_opening_keeps_the_gripper_on_the_same_joint_line(planner, arm_models):
    """The recovery action after a stopped run homes with whatever the gripper holds still in it."""
    start = state_from_joints(arm_models, {'left': AWAY, 'right': np.zeros(6)}, gripper_mm=40.)
    opened = planner.plan_home(start, ['left'])
    kept = planner.plan_home(start, ['left'], open_grippers=False)
    assert kept['arms']['left']['gripper_events'] == []
    np.testing.assert_array_equal(kept['arms']['left']['joints_deg'], opened['arms']['left']['joints_deg'])
    np.testing.assert_array_equal(kept['arms']['left']['time_s'], opened['arms']['left']['time_s'])
