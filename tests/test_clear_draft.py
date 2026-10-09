"""``DELETE /api/draft`` drops the draft and its preview without commanding the robot."""
from conftest import line_draft, make_client


def test_clear_removes_both_arms_and_the_preview_without_motion(sim_service):
    sim_service.observe()
    client = make_client(sim_service)
    assert client.put('/api/draft', json=line_draft(sim_service, ['left', 'right'], [.02, 0., 0.])).status_code == 200
    preview = client.post('/api/preview', json={})
    assert preview.status_code == 200
    before = client.get('/api/state').json()
    response = client.delete('/api/draft')
    assert response.status_code == 200 and response.json() == {'revision': before['revision']+1, 'draft': {}}
    after = client.get('/api/state').json()
    assert sim_service.draft == {} and sim_service.preview_data is None and after['preview_id'] is None
    assert after['arms'] == before['arms'] and sim_service.worker is None
    assert client.post('/api/execute', json={'preview_id': preview.json()['id']}).status_code == 409


def test_clearing_after_an_error_also_acknowledges_it(sim_service):
    sim_service.observe()
    client = make_client(sim_service)
    broken = {'observation_id': sim_service.frame.id,
              'arms': {'left': {'path': {'mode': 'function', 'x': 'sqrt(-1)', 'y': 'y0', 'z': 'z0'}, 'speed': '.02'}}}
    assert client.put('/api/draft', json=broken).status_code == 200
    assert client.post('/api/preview', json={}).status_code == 409
    assert sim_service.execution['state'] == 'error'
    epoch = sim_service.cancel_epoch
    assert client.delete('/api/draft').status_code == 200
    assert sim_service.draft == {} and sim_service.execution['state'] == 'idle'
    assert sim_service.cancel_epoch == epoch+1


def test_clear_refuses_to_edit_a_running_trajectory(sim_service):
    sim_service.observe()
    client = make_client(sim_service)
    sim_service.set_draft(line_draft(sim_service, ['left'], [.05, 0., 0.]))
    sim_service.execute(sim_service.preview()['id'])
    kept = dict(sim_service.draft)
    response = client.delete('/api/draft')
    assert response.status_code == 409 and sim_service.draft == kept
    sim_service.cancel()
    sim_service.worker.join(10)
