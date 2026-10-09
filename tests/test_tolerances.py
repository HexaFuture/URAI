"""Pose tolerances: shared settings, validation, and the PiPER-X planner applying exactly the configured limits."""
import numpy as np
import pytest
from conftest import line_draft, make_client
from scipy.spatial.transform import Rotation

from urai.trajectory import compile_arms

TOP_DOWN = {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 180, 0, 0]]}


def test_shared_settings_default_and_invalidate_previous_preview(sim_service):
    s = sim_service
    s.observe()
    client = make_client(s)
    assert client.get('/api/settings').json()['pose_tolerance'] == {'position_mm': 10., 'orientation_deg': 3.}
    d = {'observation_id': s.frame.id, 'arms': {'left': {'path': {'mode': 'function', 'x': 'x0+.01*s', 'y': 'y0', 'z': 'z0'}}}}
    client.put('/api/draft', json=d)
    old = client.post('/api/preview').json()
    changed = client.put('/api/settings', json={'pose_tolerance': {'position_mm': 12., 'orientation_deg': 4.}})
    assert changed.status_code == 200
    assert client.get('/api/draft').json() == d
    assert client.post('/api/execute', json={'preview_id': old['id']}).status_code == 409
    new = client.post('/api/preview').json()
    assert new['pose_tolerance'] == {'position_mm': 12., 'orientation_deg': 4.}
    assert old['pose_tolerance'] == {'position_mm': 10., 'orientation_deg': 3.}
    s.observe()
    assert client.get('/api/settings').json()['pose_tolerance'] == new['pose_tolerance']


@pytest.mark.parametrize('bad', [
    {'position_mm': -1, 'orientation_deg': 3},
    {'position_mm': 10, 'orientation_deg': -1},
    {'position_mm': True, 'orientation_deg': 3},
    {'position_mm': 10, 'orientation_deg': 181},
])
def test_invalid_tolerance_does_not_change_settings(sim_service, bad):
    client = make_client(sim_service)
    before = client.get('/api/settings').json()
    assert client.put('/api/settings', json={'pose_tolerance': bad}).status_code == 409
    assert client.get('/api/settings').json() == before


def test_settings_cannot_change_while_a_motion_runs(sim_service):
    sim_service.observe()
    client = make_client(sim_service)
    sim_service.set_draft(line_draft(sim_service, ['left'], [.05, 0., 0.]))
    sim_service.execute(sim_service.preview()['id'])
    response = client.put('/api/settings', json={'pose_tolerance': {'position_mm': 20, 'orientation_deg': 5}})
    sim_service.cancel()
    sim_service.worker.join(10)
    assert response.status_code == 409 and 'running' in response.json()['detail']
    assert sim_service.pose_tolerance == {'position_mm': 10., 'orientation_deg': 3.}


def planned_errors(backend, plan):
    """Largest position (mm) and orientation (deg) error between the planned joints' forward kinematics and the
    requested samples of the left arm."""
    arm, model = plan['arms']['left'], backend.models['left']
    poses = [model.fk_tcp_world(q) for q in arm['joints_deg']]
    position = max(np.linalg.norm(p[:3, 3]-x)*1000 for p, x in zip(poses, arm['xyz']))
    rotation = max(np.rad2deg((Rotation.from_matrix(p[:3, :3]).inv()*Rotation.from_matrix(r)).magnitude())
                   for p, r in zip(poses, arm['rotation_matrices']))
    return position, rotation


def test_the_planner_refuses_and_accepts_with_exactly_the_configured_tolerance(piper_backend):
    """A tool pointing straight down 0.20 m above the table at x = 0.30 m is just out of the left arm's exact reach:
    the best IK solutions miss it by about 8-12 mm and 3-5 degrees."""
    start = piper_backend.state()
    spec = {'left': {'path': {'mode': 'waypoints', 'points': [[.30, 0., .20], [.31, 0., .20]]}, 'speed': .05,
                     'orientation': TOP_DOWN}}
    for tolerance in ({'position_mm': 10, 'orientation_deg': 3}, {'position_mm': 20, 'orientation_deg': 3},
                      {'position_mm': 10, 'orientation_deg': 10}):
        allowed = f"allowed {tolerance['position_mm']:g} mm / {tolerance['orientation_deg']:g} deg"
        with pytest.raises(ValueError, match=allowed):
            piper_backend.plan(compile_arms(spec, start), start, pose_tolerance=tolerance)
    relaxed = {'position_mm': 50, 'orientation_deg': 40}
    plan = piper_backend.plan(compile_arms(spec, start), start, pose_tolerance=relaxed)
    assert plan['pose_tolerance'] == {'position_mm': 50., 'orientation_deg': 40.}
    position, rotation = planned_errors(piper_backend, plan)
    assert 10. < position <= 50. and rotation <= 40.


def test_small_increments_still_respect_a_tight_position_tolerance(piper_backend):
    start = piper_backend.state()
    spec = {'left': {'path': {'mode': 'waypoints', 'points': [[.28, 0., .12], [.2802, 0., .12]]}, 'speed': .05,
                     'orientation': TOP_DOWN}}
    plan = piper_backend.plan(compile_arms(spec, start), start, pose_tolerance={'position_mm': .01, 'orientation_deg': 3})
    position, rotation = planned_errors(piper_backend, plan)
    assert position <= .01 and rotation <= 3.
