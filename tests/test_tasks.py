"""Server-side task queue on the PiPER-X backend driving the kinematic simulator.

The simulated scene holds two blocks, a bowl, a cup and a capped bottle on the table. Each queue item observes
again, relocates its object in the fresh picture, plans with the real planner and moves the simulated arms,
whose grippers really pick the blocks up and set them down.
"""
import json

import numpy as np
import pytest
from conftest import line_draft, make_client, wait_until

from urai.app import apply_settings_file
from urai.tasks import MAX_ITEMS, arm_body_reason

TABLE_Z = .0035
#: Landing points on free table, one in reach of each arm.
LANDING = {'left': [.24, -.12, TABLE_Z], 'right': [.25, -.33, TABLE_Z]}
#: Fastest allowed queue options; with the paper's ``throw`` profile they keep one simulated item near 20 s.
FAST = {'speed_m_s': .15, 'approach_speed_m_s': .2, 'home_speed_m_s': 1.}
PAPER_SETTINGS = 'configs/paper-settings.json'


def observe(client):
    return client.post('/api/observe', json={}).json()['id']


def graspable(client, observation_id):
    """The graspable objects of the whole picture, keyed by the arm that will take them."""
    found = client.post('/api/objects', json={'observation_id': observation_id, 'all': True})
    assert found.status_code == 200, found.text
    return {item['arm']: item for item in found.json()['objects'] if item['graspable']}


def block_xy(service, name):
    return service.backend.runtime.cameras.scene.pose(name)[:2, 3].copy()


def joints(service):
    return np.r_[service.backend.runtime.joints_deg('left'), service.backend.runtime.joints_deg('right')]


def test_queue_runs_items_in_order_with_fresh_observations_and_moves_the_objects(piper_service, tmp_path):
    service = piper_service
    service.log_dir = tmp_path
    client = make_client(service)
    submitted = observe(client)
    objects = graspable(client, submitted)
    assert set(objects) == {'left', 'right'}
    start = {name: block_xy(service, name) for name in ('red_block', 'blue_block')}
    assert client.put('/api/settings', json={'motion_profile': 'throw'}).status_code == 200
    response = client.post('/api/tasks', json={'observation_id': submitted, **FAST, 'items': [
        {'object_pixel': objects['right']['pixel'], 'landing_xyz': LANDING['right']},
        {'object_pixel': objects['left']['pixel'], 'landing_xyz': LANDING['left'], 'label': 'red'}]})
    assert response.status_code == 200, response.text
    status = response.json()
    assert status['active'] and [item['arm'] for item in status['items']] == ['right', 'left']
    assert [item['landing_source'] for item in status['items']] == ['item', 'item']
    assert [item['placement'] for item in status['items']] == ['place', 'place']
    assert [item['label'] for item in status['items']] == ['物体 1', 'red']
    assert client.get('/api/state').json()['task_queue']['active']
    with pytest.raises(ValueError, match='队列'):
        service.set_draft(line_draft(service, ['left'], [.02, 0., 0.]))
    again = client.post('/api/tasks', json={'observation_id': service.frame.id, 'items': [
        {'object_pixel': objects['left']['pixel'], 'landing_xyz': LANDING['left']}]})
    assert again.status_code == 409 and '队列' in again.json()['detail']
    service.tasks.thread.join(180)
    items = client.get('/api/tasks').json()['items']
    assert [item['status'] for item in items] == ['completed', 'completed'], items
    assert len({submitted, *[item['observation_id'] for item in items]}) == 3
    assert service.frame.id == items[-1]['observation_id']
    assert all(item['preview_id'] and item['error'] is None for item in items)
    # The blocks were carried to their landing points (the grasp need not be centred on the block).
    for name, arm in (('blue_block', 'right'), ('red_block', 'left')):
        assert np.linalg.norm(block_xy(service, name)-LANDING[arm][:2]) < .04, name
        assert np.linalg.norm(block_xy(service, name)-start[name]) > .08, name
    records = [json.loads(path.read_text()) for path in tmp_path.glob('*.json')]
    assert sorted((record['task']['index'], record['task']['arm']) for record in records) == [(0, 'right'), (1, 'left')]
    assert all(record['execution']['state'] == 'completed' for record in records)
    assert not service.tasks.active and not client.get('/api/state').json()['task_queue']['active']
    assert service.draft == {} and service.preview_data is None
    assert np.abs(joints(service)).max() < .7                  # every item ended with its arm back home
    assert client.put('/api/draft', json=line_draft(service, ['left'], [.02, 0., 0.])).status_code == 200


def test_a_failed_item_stops_the_queue_and_skips_the_rest(piper_service):
    service = piper_service
    client = make_client(service)
    observation_id = observe(client)
    objects = graspable(client, observation_id)
    assert client.put('/api/drop-point', json={'xyz': LANDING['left']}).status_code == 200
    service.tasks.submit({'observation_id': observation_id, 'items': [
        {'object_pixel': objects['left']['pixel'], 'landing_xyz': [.95, -.2, TABLE_Z]},    # beyond the arm's reach
        {'object_pixel': objects['right']['pixel']}]})
    service.tasks.thread.join(60)
    items = service.tasks.status()['items']
    assert [item['status'] for item in items] == ['failed', 'skipped']
    assert [item['placement'] for item in items] == ['place', 'drop']
    assert [item['landing_source'] for item in items] == ['item', 'drop_point']
    assert items[0]['error'] and items[1]['error'] is None
    assert items[0]['observation_id'] and items[1]['observation_id'] is None
    assert not service.tasks.active and service.draft == {}
    assert np.abs(joints(service)).max() < 1e-9               # planning failed before anything moved


def test_delete_tasks_cancels_the_running_item_and_skips_the_rest(piper_service):
    service = piper_service
    client = make_client(service)
    observation_id = observe(client)
    objects = graspable(client, observation_id)
    response = client.post('/api/tasks', json={'observation_id': observation_id, **FAST, 'items': [
        {'object_pixel': objects['left']['pixel'], 'landing_xyz': LANDING['left']},
        {'object_pixel': objects['right']['pixel'], 'landing_xyz': LANDING['right']}]})
    assert response.status_code == 200, response.text
    wait_until(lambda: service.tasks.status()['items'][0]['status'] == 'running', timeout=60)
    wait_until(lambda: np.abs(service.backend.runtime.joints_deg('left')).max() > 2., timeout=30)
    assert client.get('/api/state').json()['task_queue']['active']
    assert client.put('/api/draft', json=line_draft(service, ['left'], [.02, 0., 0.])).status_code == 409
    assert client.delete('/api/tasks').status_code == 200
    service.tasks.thread.join(30)
    items = client.get('/api/tasks').json()['items']
    assert [item['status'] for item in items] == ['cancelled', 'skipped']
    assert not service.tasks.active and service.execution['state'] == 'cancelled'
    assert np.abs(service.backend.runtime.joints_deg('left')).max() > 2.        # held where it was stopped


def test_the_drop_point_is_validated_and_reported(sim_service):
    sim_service.observe()
    client = make_client(sim_service)
    assert client.get('/api/drop-point').json() == {'xyz': None}
    refused = client.post('/api/tasks', json={'observation_id': sim_service.frame.id, 'items': [{'object_pixel': [395, 220]}]})
    assert refused.status_code == 409 and '投放点' in refused.json()['detail']
    for bad in ([.4, 'x', 0.], [3., 0., 0.], [.4, 0.], [.4, True, 0.], None):
        assert client.put('/api/drop-point', json={'xyz': bad}).status_code == 409, bad
    assert client.put('/api/drop-point', json={'xyz': [.48, .07, 0]}).json() == {'xyz': [.48, .07, 0.]}
    assert client.get('/api/drop-point').json() == {'xyz': [.48, .07, 0.]}
    assert client.get('/api/state').json()['drop_point'] == [.48, .07, 0.]


def test_task_and_object_routes_validate_their_input_without_moving(piper_service):
    service = piper_service
    client = make_client(service)
    observation_id = observe(client)
    red = graspable(client, observation_id)['left']
    u, v = red['pixel']
    polygon = [[u-40, v-40], [u+40, v-40], [u+40, v+40], [u-40, v+40]]
    assert client.post('/api/objects', json={'observation_id': 'old', 'polygon': polygon}).status_code == 409
    assert client.post('/api/objects', json={'observation_id': observation_id, 'polygon': polygon[:2]}).status_code == 409
    response = client.post('/api/objects', json={'observation_id': observation_id, 'polygon': polygon})
    assert response.status_code == 200, response.text
    objects = response.json()['objects']
    assert len(objects) == 1 and objects[0]['graspable'] and objects[0]['arm'] == 'left'
    assert service.draft == {} and service.execution['state'] == 'idle'
    item = {'object_pixel': objects[0]['pixel'], 'landing_xyz': LANDING['left'], 'arm': 'left'}
    body = {'observation_id': observation_id, 'items': [item]}
    for bad in ({**body, 'observation_id': 'old'}, {**body, 'items': []}, {**body, 'items': [item]*(MAX_ITEMS+1)},
                {**body, 'items': [{**item, 'arm': 'middle'}]},
                {**body, 'items': [{**item, 'object_xy': [0, 0]}]},
                {**body, 'items': [{**item, 'colour': 'red'}]},
                {**body, 'items': [{**item, 'label': 'x'*65}]},
                {**body, 'items': [{**item, 'placement': 'throw'}]},
                {**body, 'items': [{**item, 'grasp_pixels': [[u-10, v], [u+10, v]]}]},
                {**body, 'items': [{**item, 'object_pixel': [-5, 10]}]},
                {**body, 'speed_m_s': .5}, {**body, 'toss_style': 'underhand'}, {**body, 'release_tilt_deg': 90}):
        assert client.post('/api/tasks', json=bad).status_code == 409, bad
    assert client.get('/api/tasks').json() == {'active': False, 'items': []}
    assert client.delete('/api/tasks').status_code == 200
    by_world = {**body, 'items': [{'object_xy': objects[0]['center_xy'], 'landing_xyz': LANDING['left'], 'arm': 'left'}]}
    response = client.post('/api/tasks', json=by_world)
    assert response.status_code == 200, response.text
    queued = response.json()['items'][0]
    assert queued['arm'] == 'left' and queued['landing_source'] == 'item' and queued['placement'] == 'place'
    # Relocated from the world point it is the same block; the queue's own segmentation puts its centre about
    # 1 cm closer to the oblique head camera than the objects route does.
    assert queued['object_xyz'][:2] == pytest.approx(objects[0]['center_xy'], abs=.015)
    assert client.delete('/api/tasks').status_code == 200
    service.tasks.thread.join(30)
    assert service.tasks.status()['items'][0]['status'] == 'cancelled'
    assert np.abs(joints(service)).max() < 1e-9


def test_tasks_are_refused_while_a_motion_runs(sim_service):
    sim_service.observe()
    client = make_client(sim_service)
    sim_service.set_drop_point([.48, .07, 0.])
    sim_service.set_draft(line_draft(sim_service, ['left'], [.05, 0., 0.]))
    sim_service.execute(sim_service.preview()['id'])
    response = client.post('/api/tasks', json={'observation_id': sim_service.frame.id, 'items': [{'object_pixel': [395, 220]}]})
    sim_service.cancel()
    sim_service.worker.join(10)
    assert response.status_code == 409 and 'running' in response.json()['detail']
    assert sim_service.tasks.status() == {'active': False, 'items': []}


def test_the_objects_route_detects_the_whole_frame_and_marks_arm_bodies(piper_service):
    service = piper_service
    client = make_client(service)
    observation_id = observe(client)
    objects = client.post('/api/objects', json={'observation_id': observation_id, 'all': True}).json()['objects']
    scene = service.backend.runtime.cameras.scene

    def nearest(name):
        xy = scene.pose(name)[:2, 3]
        return min(objects, key=lambda item: np.linalg.norm(np.asarray(item['center_xy'])-xy))

    taken = [item for item in objects if item['graspable']]
    assert sorted(item['arm'] for item in taken) == ['left', 'right']
    for name in ('red_block', 'blue_block'):
        assert nearest(name)['graspable'] and np.linalg.norm(np.asarray(nearest(name)['center_xy'])-scene.pose(name)[:2, 3]) < .02
    bottle = nearest('bottle')
    assert not bottle['graspable'] and bottle['height_mm'] > 80 and '上限' in bottle['reason']
    # The arms' own base columns rise from the table too; the arm models claim them.
    assert any(not item['graspable'] and '左臂本体' in item['reason'] for item in objects)
    assert any(not item['graspable'] and '右臂本体' in item['reason'] for item in objects)
    models, current = service.backend.models, service.backend.state()
    _, a, b, _ = next(iter(models['right'].body_capsules(np.zeros(6), world=True)))
    on_link = (np.asarray(a)+np.asarray(b))/2000.
    assert '右臂本体' in arm_body_reason(models, current, on_link)
    assert arm_body_reason(models, current, [*scene.pose('red_block')[:2, 3], .03]) is None
    assert arm_body_reason({}, current, on_link) is None         # the synthetic backend has no arm models


def test_toss_items_need_the_throw_profile(piper_service):
    """A toss swing retimed to any slower profile would release the object at the right pose but barely move it."""
    service = piper_service
    client = make_client(service)
    observation_id = observe(client)
    red = graspable(client, observation_id)['left']
    body = {'observation_id': observation_id,
            'items': [{'object_pixel': red['pixel'], 'landing_xyz': [.30, .30, TABLE_Z], 'placement': 'toss'}]}
    refused = client.post('/api/tasks', json=body)
    assert refused.status_code == 409 and 'throw' in refused.json()['detail']
    assert service.tasks.status() == {'active': False, 'items': []}
    apply_settings_file(service, PAPER_SETTINGS)               # the paper trials ran with the throw profile
    accepted = client.post('/api/tasks', json=body)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()['items'][0]['placement'] == 'toss'
    assert client.delete('/api/tasks').status_code == 200        # stop before anything is thrown
    service.tasks.thread.join(60)
    assert service.tasks.status()['items'][0]['status'] == 'cancelled'
    assert np.abs(joints(service)).max() < 1e-9
