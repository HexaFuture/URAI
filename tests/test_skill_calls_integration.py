"""Exercise real URAI skill planning and HTTP wiring on SimulationBackend only."""
from conftest import make_client
from urai.backend import SimulationBackend
from urai.service import Service


def setup(tmp_path, monkeypatch):
    service = Service(SimulationBackend(), log_dir=tmp_path)
    service.observe()
    service.frame.depth[:] = 1.
    service.frame.depth[210:242, 380:412] = .97
    executions = []
    def execute(plan, *args):
        executions.append(plan.get('kind', 'skill'))
        return {'completed': True, 'elapsed_s': 0.}
    monkeypatch.setattr(service.backend, 'execute', execute)
    return service, make_client(service), executions


def body(service):
    return {'request_id': 'block-run', 'observation_id': service.frame.id, 'arm': 'right',
            'pixels': [[396, 226], [560, 290]], 'speed_m_s': .1, 'approach_speed_m_s': .1, 'wait_s': 10}


def test_pick_place_one_http_post_uses_existing_skill_and_home(tmp_path, monkeypatch):
    service, client, executions = setup(tmp_path, monkeypatch)
    request = body(service)
    response = client.post('/api/skills/pick_place/run', json=request)
    assert response.status_code == 200, response.text
    assert response.json()['state'] == 'completed', response.text
    assert len(executions) == 2 and executions[-1] == 'home'
    assert client.post('/api/skills/pick_place/run', json=request).json()['state'] == 'completed'
    assert len(executions) == 2
    assert client.get('/api/skills').status_code == 200
    assert client.get('/api/state').status_code == 200


def test_real_preview_failure_never_dispatches(tmp_path, monkeypatch):
    service, client, executions = setup(tmp_path, monkeypatch)
    def failed_plan(*args, **kwargs):
        raise ValueError('test collision')
    monkeypatch.setattr(service.backend, 'plan', failed_plan)
    response = client.post('/api/skills/pick_place/run', json=body(service))
    assert response.json()['state'] == 'error'
    assert 'test collision' in response.json()['error']
    assert executions == []
