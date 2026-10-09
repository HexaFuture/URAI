"""Motion profiles: snapshotted into each preview, guarded while planning, and sent to the arm controllers."""
import threading

import pytest
from conftest import line_draft, make_client, wait_until

from urai.robot.simulated import FIRMWARE_MAX_JOINT_SPEED_DEG_S
from urai.settings import MOTION_PROFILES


def top_down_draft(service):
    """A left-arm move that needs an approach and a long IK chain: its preview plans for about two seconds."""
    return {'observation_id': service.frame.id, 'arms': {'left': {
        'path': {'mode': 'waypoints', 'points': [[.28, 0., .12], [.31, 0., .12]]}, 'speed': .05,
        'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 180, 0, 0]]}}}}


def test_speed_profile_is_snapshotted_and_invalidates_preview_without_changing_path(sim_service):
    sim_service.observe()
    c = make_client(sim_service)
    draft = {'observation_id': sim_service.frame.id,
             'arms': {'left': {'path': {'mode': 'function', 'x': 'x0+.01*s', 'y': 'y0', 'z': 'z0'}}}}
    c.put('/api/draft', json=draft)
    old = c.post('/api/preview').json()
    assert c.put('/api/settings', json={'motion_profile': 'fast'}).status_code == 200
    assert c.get('/api/draft').json() == draft
    assert c.post('/api/execute', json={'preview_id': old['id']}).status_code == 409
    new = c.post('/api/preview').json()
    assert old['motion_limits'] == {'profile': 'normal', **MOTION_PROFILES['normal']}
    assert new['motion_limits'] == {'profile': 'fast', **MOTION_PROFILES['fast']}
    assert new['motion_limits']['controller_speed_percent'] == 60 and new['motion_limits']['joint_speed_deg_s'] == 45


def test_invalid_combined_settings_are_atomic(sim_service):
    c = make_client(sim_service)
    before = c.get('/api/settings').json()
    data = {'motion_profile': 'unknown', 'pose_tolerance': {'position_mm': 20, 'orientation_deg': 5}}
    assert c.put('/api/settings', json=data).status_code == 409
    assert c.get('/api/settings').json() == before


def test_the_profile_cannot_change_while_a_preview_is_planning(piper_service):
    piper_service.observe()
    c = make_client(piper_service)
    piper_service.set_draft(top_down_draft(piper_service))
    results = []
    worker = threading.Thread(target=lambda: results.append(piper_service.preview()))
    worker.start()
    wait_until(lambda: piper_service.execution.get('stage') == 'ik')
    refused = c.put('/api/settings', json={'motion_profile': 'fast'})
    worker.join(30)
    assert refused.status_code == 409 and 'running' in refused.json()['detail']
    assert results and results[0]['motion_limits']['profile'] == 'normal'
    assert piper_service.motion_profile == 'normal'
    assert c.put('/api/settings', json={'motion_profile': 'fast'}).status_code == 200


def test_execution_sets_the_controller_speed_of_its_preview(piper_service):
    """The preview carries the profile's controller speed and execution puts each arm's controller in it.

    The simulated controller then moves its joints at that percentage of the firmware joint speed limit; it starts
    at the rig's own setting, so the change is visible.
    """
    service, profile = piper_service, 'throw'
    service.observe()
    driver = service.backend.runtime.arms['left']
    assert driver._rate_deg_s != pytest.approx(FIRMWARE_MAX_JOINT_SPEED_DEG_S*MOTION_PROFILES[profile]['controller_speed_percent']/100)
    service.set_settings({'motion_profile': profile})
    service.set_draft(line_draft(service, ['left'], [.05, 0., 0.], speed=.05))
    plan = service.preview()
    assert plan['motion_limits'] == {'profile': profile, **MOTION_PROFILES[profile]}
    service.execute(plan['id'])
    service.worker.join(60)
    assert service.execution['state'] == 'completed'
    percent = plan['motion_limits']['controller_speed_percent']
    assert driver._rate_deg_s == pytest.approx(FIRMWARE_MAX_JOINT_SPEED_DEG_S*percent/100)
