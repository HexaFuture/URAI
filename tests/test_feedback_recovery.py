"""Recovering from a stopped operation: acknowledging it without motion, and homing with the gripper kept.

Everything runs through the real HTTP app on the real PiPER-X backend and the kinematic simulator. The guard
against late control-loop feedback (held, re-checked, never dispatched from stale geometry) needs a loaded
control host and is covered by tests/hardware/test_backend_on_robot.py.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from urai.service import Service


def near_home(service):
    """A 3 cm stroke forward from the left arm's home pose, keeping its orientation: every plan here is short."""
    xyz = np.asarray(service.backend.state()['left']['xyz'])
    return {'mode': 'waypoints', 'points': [xyz.tolist(), (xyz+[.03, 0., 0.]).tolist()]}


def failed_preview(backend):
    """A service whose last preview failed: the drawn path leaves the arm's workspace."""
    service = Service(backend)
    service.observe()
    service.set_draft({'observation_id': service.frame.id,
                       'arms': {'left': {'path': {'mode': 'waypoints', 'points': [[.9, 0., .1], [.95, 0., .1]]}}}})
    with pytest.raises(ValueError):
        service.preview()
    assert service.execution['state'] == 'error'
    return service


def test_a_failed_preview_offers_recovery_and_acknowledging_it_never_moves(piper_backend, client_for):
    rt = piper_backend.runtime
    service = failed_preview(piper_backend)
    error = service.execution['error']
    client = client_for(service)
    actions = {action['name']: action for action in client.get('/api/state').json()['recovery_actions']}
    assert set(actions) == {'clear_error', 'home'} and actions['home']['body']['open_grippers'] is False
    response = client.post('/api/recover')
    assert response.status_code == 200
    assert response.json()['previous_execution']['error'] == error
    assert service.execution['state'] == 'idle' and service.preview_data is None
    assert client.get('/api/state').json()['recovery_actions'] == []
    for arm in ('left', 'right'):
        np.testing.assert_array_equal(rt.joints_deg(arm), np.zeros(6))
        assert rt.last_gripper_command[arm] is None


def test_clearing_the_draft_clears_an_error(piper_backend, client_for):
    service = failed_preview(piper_backend)
    assert client_for(service).delete('/api/draft').status_code == 200
    assert service.execution['state'] == 'idle' and service.draft == {}


def test_recovery_cannot_interrupt_a_running_motion(piper_backend, client_for):
    service = Service(piper_backend)
    service.observe()
    service.set_draft({'observation_id': service.frame.id,
                       'arms': {'left': {'path': near_home(service), 'speed': .02}}})
    service.execute(service.preview()['id'])
    try:
        assert service.execution['state'] == 'running'
        assert client_for(service).post('/api/recover').status_code == 409
        assert service.execution['state'] == 'running'
    finally:
        service.cancel()
        service.worker.join(10)


def test_after_an_empty_grip_home_returns_the_arm_without_opening_the_gripper(piper_backend, tmp_path, client_for):
    rt = piper_backend.runtime
    service = Service(piper_backend, tmp_path)
    service.observe()
    grip = [{'s': 0., 'opening_mm': 30., 'hold_s': .6, 'wait_for_arrival': True},
            {'s': 1., 'opening_mm': 0., 'hold_s': .3, 'wait_for_arrival': True, 'verify': 'holding'}]
    service.set_draft({'observation_id': service.frame.id,
                       'arms': {'left': {'path': near_home(service), 'gripper_events': grip}}})
    service.execute(service.preview()['id'])
    service.worker.join(30)
    assert service.execution['state'] == 'error' and '疑似空抓' in service.execution['error']
    assert np.abs(rt.joints_deg('left')).max() > 1.
    closed = rt.gripper_mm('left')
    response = client_for(service).post('/api/home', json={'open_grippers': False})
    assert response.status_code == 200, response.text
    service.worker.join(30)
    assert service.execution['state'] == 'completed'
    np.testing.assert_allclose(rt.joints_deg('left'), 0., atol=.7)
    time.sleep(.2)
    assert rt.gripper_mm('left') == closed and rt.last_gripper_command['left'] == 0.
    record = json.loads((tmp_path / f"{response.json()['id']}.json").read_text())
    assert record['preview']['kind'] == 'home' and record['preview']['arms']['left']['gripper_events'] == []
