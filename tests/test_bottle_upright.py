"""Upright-bottle geometry and API stages using the real synthetic backend."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from conftest import make_client
from test_bottle_cap import context, observe_points
from urai.skills.bottle_upright import cap_route


@pytest.mark.parametrize('lift', [0., .005])
def test_cap_rotation_keeps_the_contact_on_the_world_vertical_axis(lift):
    center = np.array([.35, -.2, .3])
    compensation = .01
    points, rotations, events = cap_route(center, [1., 0.], compensation, 60, 2, .07, 1200,
                                         pitch=15, lift=lift)
    base = rotations[0]
    # The cap contact is the TCP with the fingertip compensation removed.
    for p, r in zip(points[1:-1], rotations[1:-1]):
        np.testing.assert_allclose(p-r[:, 1]*compensation, center, atol=1e-12)
        relative = r @ base.T
        np.testing.assert_allclose(relative @ [0., 0., 1.], [0., 0., 1.], atol=1e-12)
    if lift:
        np.testing.assert_allclose(points[-1]-rotations[-1][:, 1]*compensation, center+[0., 0., lift])
        assert events[-1]['opening_mm'] == 0
    else:
        np.testing.assert_allclose(points[-1], points[0])
        assert events[-1]['opening_mm'] == 70
    assert sum(e.get('opening_mm') == 0 for e in events) == 2
    assert all(0 <= e['s'] <= 1 for e in events)


def draft(client, service, name, pixel, arm='left', **numbers):
    response = client.post(f'/api/skills/{name}', json={
        'observation_id': service.frame.id, 'arm': arm, 'pixels': [pixel], 'commit': True, **numbers})
    assert response.status_code == 200, response.text
    result = response.json()
    preview = client.post('/api/preview', json={'expected_revision': result['revision']})
    assert preview.status_code == 200, preview.text
    return result, preview.json()


def execute(client, service, preview):
    response = client.post('/api/execute', json={'preview_id': preview['id']})
    assert response.status_code == 200, response.text
    service.worker.join(timeout=120)
    assert service.execution['state'] == 'completed', service.execution
    assert 'return_home' not in service.execution['result']


def test_upright_stages_preserve_the_holder_and_release_clears_the_session():
    service, _, pixel = context()
    client = make_client(service)
    _, preview = draft(client, service, 'bottle_upright_prepare', pixel, speed_m_s=.15)
    execute(client, service, preview)
    session = service.bottle_cap
    assert session['mode'] == 'upright' and session['state'] == 'waiting_cap'
    holder = service.backend.state()['left']
    r = Rotation.from_euler('xyz', holder['rpy_deg'], degrees=True).as_matrix()
    np.testing.assert_allclose(r @ session['axis_local'], [0., 0., 1.], atol=1e-9)
    cap = np.asarray(holder['xyz']) + r @ session['cap_local']
    [pixel] = observe_points(service, cap)
    _, preview = draft(client, service, 'bottle_upright_twist', pixel, arm='right', turn_deg=10, speed_m_s=.15)
    execute(client, service, preview)
    assert service.bottle_cap['state'] == 'completed'
    np.testing.assert_allclose(service.backend.state()['left']['xyz'], holder['xyz'])
    [pixel] = observe_points(service, cap)
    _, preview = draft(client, service, 'bottle_upright_retract', pixel, arm='right', retreat_mm=30)
    execute(client, service, preview)
    assert service.bottle_cap['state'] == 'needs_relocalization'
    [pixel] = observe_points(service, cap)
    _, preview = draft(client, service, 'bottle_upright_release', pixel)
    execute(client, service, preview)
    assert service.bottle_cap is None
    np.testing.assert_allclose(service.backend.state()['left']['xyz'], holder['xyz'])
    assert service.backend.state()['left']['gripper_mm'] == pytest.approx(70.)


def test_body_recenter_uses_the_body_surface_and_creates_an_upright_draft():
    service, _, pixel = context(cap=(.4, 0., .1))
    result, _ = draft(make_client(service), service, 'bottle_body_recenter', pixel)
    metadata = result['arm']['bottle_cap']
    assert metadata['mode'] == 'upright'
    assert metadata['bottle_height_m'] == pytest.approx(.28, abs=.001)


def test_upright_twist_requires_an_upright_session():
    service, _, pixel = context()
    client = make_client(service)
    response = client.post('/api/skills/bottle_upright_twist', json={
        'observation_id': service.frame.id, 'arm': 'right', 'pixels': [pixel]})
    assert response.status_code == 409
