"""The draft of a motion that ran to completion is cleared; anything else keeps it."""
import pytest
from conftest import line_draft, stop_midway


def observed(service):
    service.observe()
    return service


def straight_draft(service):
    return {'observation_id': service.frame.id,
            'arms': {'left': {'path': {'mode': 'waypoints', 'points': [[.25, 0, .2], [.28, 0, .2]]},
                              'speed': '.05', 'orientation': {'mode': 'hold'}, 'gripper_events': []}}}


def run_once(service):
    """Draft, preview and execute one short trajectory; returns once the worker thread is done."""
    service.set_draft(straight_draft(service))
    plan = service.preview()
    service.execute(plan['id'])
    service.worker.join(30)
    assert not service.worker.is_alive()
    return service.execution


def test_a_finished_trajectory_is_cleared_so_the_console_stops_drawing_it(sim_service):
    service = observed(sim_service)
    before = service.revision
    execution = run_once(service)
    assert execution['state'] == 'completed' and execution['draft_cleared'] is True
    assert service.draft == {} and service.revision > before
    assert service.preview_data is None
    # The run cannot be repeated from the leftover draft: there is nothing left to preview.
    with pytest.raises(ValueError, match='draft'):
        service.preview()


def test_the_draft_survives_a_run_that_failed(locked_piper_service):
    """A failed run keeps its path so the operator can see what was attempted (here: no motion permission)."""
    service = observed(locked_piper_service)
    service.set_draft(line_draft(service, ['left'], [.05, 0., 0.], speed=.05))
    arms = {'left': service.draft['arms']['left']}
    service.execute(service.preview()['id'])
    service.worker.join(30)
    assert service.execution['state'] == 'error' and 'draft_cleared' not in service.execution
    assert service.draft['arms'] == arms


def test_the_draft_survives_a_run_that_was_cancelled(sim_service):
    service = observed(sim_service)
    stop_midway(service)
    assert 'draft_cleared' not in service.execution
    assert set(service.draft['arms']) == {'left'} and service.draft['observation_id'] == service.frame.id


def test_homing_keeps_a_prepared_draft(sim_service):
    """Home is the step before executing a prepared path, not a user trajectory."""
    service = observed(sim_service)
    service.set_draft(straight_draft(service))
    arms = {'left': service.draft['arms']['left']}
    revision = service.revision
    service.home(['left'])
    service.worker.join(60)
    assert service.execution['state'] == 'completed' and 'draft_cleared' not in service.execution
    assert service.draft['arms'] == arms and service.revision == revision
