"""Service state machine on the real backends: drafts, previews, execution, cancellation and run records."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from conftest import line_draft, make_client, stop_midway, wait_until

from urai.service import PREVIEW_TTL_S


def observed(service):
    service.observe()
    return service


def draft(s):
    return {'observation_id': s.frame.id, 'arms': {'left': {'path': {
        'mode': 'function', 'x': 'x0+.005*s', 'y': 'y0', 'z': 'z0'}, 'speed': '.02'}}}


def test_preview_cannot_be_reused_or_survive_edit(sim_service):
    s = observed(sim_service)
    s.set_draft(draft(s))
    p = s.preview()
    s.set_draft(draft(s))
    with pytest.raises(ValueError, match='preview'):
        s.execute(p['id'])


def test_observation_invalidates_draft_and_preview(sim_service):
    s = observed(sim_service)
    s.set_draft(draft(s))
    p = s.preview()
    s.observe()
    assert s.draft == {}
    with pytest.raises(ValueError):
        s.execute(p['id'])


def test_a_refreshed_observation_keeps_the_world_path_and_replans_from_the_current_pose(sim_service):
    """A motion stopped half-way leaves its draft; observing again with preserve_draft keeps the drafted world path,
    and the next preview approaches it from wherever the arm stopped."""
    s = observed(sim_service)
    stop_midway(s)
    arms = copy.deepcopy(s.draft['arms'])
    here = s.backend.state()['left']['xyz']
    frame = s.observe(preserve_draft=True)
    assert s.draft['arms'] == arms and s.draft['observation_id'] == frame['id']
    assert not s.status()['observation_stale']
    p = s.preview()
    assert p['approach_duration_s'] > 0
    assert p['start']['left']['xyz'] == pytest.approx(here)


def test_observing_is_refused_mid_execution_and_allowed_after_an_error(sim_service):
    s = observed(sim_service)
    client = make_client(s)
    s.set_draft(line_draft(s, ['left'], [.05, 0., 0.]))
    s.execute(s.preview()['id'])
    kept = copy.deepcopy(s.draft)
    frame = s.frame.id
    assert client.post('/api/observe', json={'preserve_draft': True}).status_code == 409
    assert s.frame.id == frame and s.draft == kept
    s.cancel()
    s.worker.join(10)
    broken = draft(s)
    broken['arms']['left']['path']['x'] = 'sqrt(-1)'
    s.set_draft(broken)
    with pytest.raises(ValueError):
        s.preview()
    assert s.execution['state'] == 'error'
    assert client.post('/api/observe', json={'preserve_draft': True}).status_code == 200
    assert s.draft['arms'] == broken['arms'] and s.execution['state'] == 'idle'


def test_execute_is_atomic_and_cancel_holds(sim_service):
    s = observed(sim_service)
    s.set_draft(draft(s))
    p = s.preview()
    s.execute(p['id'])
    with pytest.raises(ValueError):
        s.execute(p['id'])
    with pytest.raises(ValueError, match='running'):
        s.set_draft(draft(s))
    s.cancel()
    s.worker.join(3)
    assert s.status()['execution']['state'] == 'cancelled'
    assert s.backend.hold_count == 1


def test_an_arm_moved_after_the_preview_is_not_executed(piper_service):
    """Someone moves the arm between preview and execute: the measured start no longer matches the plan."""
    s = observed(piper_service)
    s.set_draft(line_draft(s, ['left'], [.05, 0., 0.], speed=.05))
    p = s.preview()
    driver = s.backend.runtime.arms['left']
    moved = driver.joints()+[2., 0., 0., 0., 0., 0.]
    driver.command_joints(moved)
    wait_until(lambda: np.allclose(driver.joints(), moved, atol=.01))
    with pytest.raises(ValueError, match='moved since preview'):
        s.execute(p['id'])
    assert s.worker is None and s.execution['state'] == 'ready'


def test_arm_movement_since_observation_requires_new_frame(sim_service):
    s = observed(sim_service)
    stop_midway(s)
    s.set_settings({'auto_prepare': {'refresh_observation': False}})
    s.set_draft(draft(s))
    with pytest.raises(ValueError, match='observation'):
        s.preview()


def test_without_motion_permission_execution_and_homing_are_refused(locked_piper_service):
    """Started without --allow-motion, planning still works but every motion fails before the arms move."""
    s = observed(locked_piper_service)
    start = {arm: s.backend.state()[arm]['joints_deg'] for arm in ('left', 'right')}
    s.set_draft(line_draft(s, ['left'], [.05, 0., 0.], speed=.05))
    s.execute(s.preview()['id'])
    s.worker.join(10)
    assert s.execution['state'] == 'error' and 'not allowed' in s.execution['error']
    s.home(['left', 'right'])
    s.worker.join(10)
    assert s.execution['state'] == 'error' and 'not allowed' in s.execution['error']
    assert {arm: s.backend.state()[arm]['joints_deg'] for arm in ('left', 'right')} == start


def test_function_draft_is_copied_and_nonfinite_input_rejected(sim_service):
    s = observed(sim_service)
    d = draft(s)
    s.set_draft(d)
    d['arms']['left']['path']['x'] = '100'
    p = s.preview()
    assert p['arms']['left']['xyz'][-1][0] < 1
    bad = draft(s)
    bad['arms']['left']['path']['x'] = 'sqrt(-1)'
    s.set_draft(bad)
    with pytest.raises(ValueError):
        s.preview()


def test_api_routes_share_service_state(sim_service):
    s = observed(sim_service)
    client = make_client(s)
    assert client.get('/api/state').json()['backend']['mode'] == 'simulation'
    assert client.put('/api/draft', json=draft(s)).status_code == 200
    c = client.post('/api/compile')
    assert c.status_code == 200
    compiled = c.json()['arms']['left']
    # Samples are thinned to the 50 Hz control period, so a short fast draft compiles to far fewer than 201.
    assert 2 < len(compiled['xyz']) <= 201 and len(compiled['time_s']) == len(compiled['xyz'])
    p = client.post('/api/preview').json()
    assert 'id' in p
    assert client.get('/api/draft').json()['arms']['left']['path']['mode'] == 'function'
    assert client.post('/api/execute', json={'preview_id': 'missing'}).status_code == 409


def test_each_run_record_keeps_its_own_terminal_state(sim_service, tmp_path):
    """One JSON record per executed plan, written with that run's own final state, whatever happens next."""
    s = observed(sim_service)
    s.log_dir = tmp_path
    s.set_draft(draft(s))
    completed = s.preview()
    s.execute(completed['id'])
    s.worker.join(20)
    stop_midway(s)
    cancelled = s.execution['id']
    s.set_draft(draft(s))
    s.preview()                                     # the service moves on to a new plan
    assert s.execution['state'] == 'ready'
    records = {json.loads(path.read_text())['preview']['id']: json.loads(path.read_text()) for path in tmp_path.glob('*.json')}
    assert set(records) == {completed['id'], cancelled}
    assert records[completed['id']]['execution']['state'] == 'completed'
    assert records[completed['id']]['execution']['id'] == completed['id']
    assert records[completed['id']]['result']['completed'] is True
    assert records[cancelled]['execution']['state'] == 'cancelled'
    assert records[cancelled]['execution']['id'] == cancelled and records[cancelled]['result'] is None


@pytest.mark.parametrize('endpoint', ['compile', 'preview'])
def test_incomplete_second_arm_identifies_the_arm_and_missing_endpoint(sim_service, endpoint):
    s = observed(sim_service)
    client = make_client(s)
    d = draft(s)
    d['arms']['right'] = {'path': {'mode': 'waypoints', 'points': [s.backend.state()['right']['xyz']]}}
    client.put('/api/draft', json=d)
    response = client.post('/api/'+endpoint)
    assert response.status_code == 409
    assert '右臂' in response.json()['detail'] and '终点' in response.json()['detail']
    assert client.get('/api/draft').json() == d


def test_an_expired_preview_is_remade_instead_of_reported(sim_service):
    """Executing a preview that outlived PREVIEW_TTL_S plans again from the same draft, through the same checks,
    and runs the new plan. The preview is aged by moving its creation time back; no clock is altered."""
    s = observed(sim_service)
    s.set_draft(draft(s))
    stale = s.preview()
    revision = s.revision
    s.preview_data['created_at'] -= PREVIEW_TTL_S+10
    execution = s.execute(stale['id'])
    assert execution['state'] == 'running'
    assert execution['id'] != stale['id']          # a fresh plan, not the expired one
    assert s.revision == revision                  # from the same draft, untouched
    s.worker.join(20)
    assert s.execution['state'] == 'completed'


def test_a_preview_that_was_superseded_is_still_the_callers_error(sim_service):
    """Only expiry is remade. A draft that changed under the caller means they are acting on a stale idea."""
    s = observed(sim_service)
    s.set_draft(draft(s))
    first = s.preview()
    s.set_draft(draft(s))                          # a new revision drops the preview
    with pytest.raises(ValueError, match='preview'):
        s.execute(first['id'])


def test_an_execution_records_whether_the_console_or_a_script_asked_for_it(sim_service, tmp_path):
    """The console, a person's script and an agent all reach one route; each run record says which.

    The console is same-origin and sends an Origin header; a command line does not.
    """
    s = observed(sim_service)
    s.log_dir = tmp_path
    client = make_client(s)
    for headers in ({'Origin': 'http://127.0.0.1'}, {}):
        s.set_draft(draft(s))
        plan = s.preview()
        assert client.post('/api/execute', json={'preview_id': plan['id']}, headers=headers).status_code == 200
        s.worker.join(20)
        assert s.execution['state'] == 'completed'
    records = sorted(tmp_path.glob('*.json'), key=lambda p: p.stat().st_mtime)
    callers = [json.loads(path.read_text())['caller'] for path in records]
    assert len(callers) == 2
    assert callers[0]['kind'] == 'console' and callers[0]['origin'] == 'http://127.0.0.1'
    assert callers[1]['kind'] == 'script' and callers[1]['origin'] == ''
    assert all(caller['agent'] for caller in callers)      # the user agent is recorded either way
    assert all(caller['address'] for caller in callers)


def one_arm_draft(s, arm, dx='.005'):
    """A draft naming exactly one arm: what one stroke of a two-armed move produces."""
    return {'observation_id': s.frame.id,
            'arms': {arm: {'path': {'mode': 'function', 'x': f'x0+{dx}*s', 'y': 'y0', 'z': 'z0'}, 'speed': '.02'}}}


def test_merge_keeps_the_arm_the_new_draft_does_not_name(sim_service):
    """Two-armed moves are drawn one arm at a time; the second stroke must not drop the first."""
    s = observed(sim_service)
    s.set_draft(one_arm_draft(s, 'left'))
    saved = s.set_draft(one_arm_draft(s, 'right'), merge=True)
    assert set(saved['draft']['arms']) == {'left', 'right'}


def test_merge_replaces_an_arm_drawn_twice(sim_service):
    """A second stroke for the same arm edits that arm instead of adding one."""
    s = observed(sim_service)
    s.set_draft(one_arm_draft(s, 'left', '.005'))
    saved = s.set_draft(one_arm_draft(s, 'left', '.009'), merge=True)
    assert set(saved['draft']['arms']) == {'left'}
    assert '.009' in saved['draft']['arms']['left']['path']['x']


def test_without_merge_the_whole_draft_is_replaced(sim_service):
    """Callers that do not pass merge keep the old behaviour: the draft is replaced as a whole."""
    s = observed(sim_service)
    s.set_draft(one_arm_draft(s, 'left'))
    saved = s.set_draft(one_arm_draft(s, 'right'))
    assert set(saved['draft']['arms']) == {'right'}


def test_merge_drops_an_arm_planned_against_an_older_picture(sim_service):
    """The first stroke was drawn on an older frame; what it pointed at may have moved, so it is not carried over."""
    s = observed(sim_service)
    s.set_draft(one_arm_draft(s, 'left'))
    s.observe()
    saved = s.set_draft(one_arm_draft(s, 'right'), merge=True)
    assert set(saved['draft']['arms']) == {'right'}


def test_merged_draft_plans_both_arms_from_one_instant(sim_service):
    """A merged draft runs both arms together: both are planned and their task segments share one time origin.

    The two approaches differ in length (the right arm starts far from where it stands), so the arm that
    arrives first waits and both hands start their task at the same moment.
    """
    s = observed(sim_service)
    here = s.backend.state()

    def waypoints(arm, start, end):
        return {'observation_id': s.frame.id,
                'arms': {arm: {'path': {'mode': 'waypoints', 'points': [start, end]}, 'speed': '.02'}}}

    lx, ly, lz = here['left']['xyz']
    rx, ry, rz = here['right']['xyz']
    s.set_draft(waypoints('left', [lx, ly, lz], [lx+.02, ly, lz]))
    s.set_draft(waypoints('right', [rx-.10, ry, rz+.05], [rx-.08, ry, rz+.05]), merge=True)
    plan = s.preview()
    assert set(plan['arms']) == {'left', 'right'}
    offsets = [plan['arms'][a].get('task_offset_s', 0.) for a in ('left', 'right')]
    assert offsets[0] > 0 and abs(offsets[0]-offsets[1]) < 1e-9
