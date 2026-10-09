"""Preview progress reporting and cancellation, on the PiPER-X planner driving the kinematic simulator."""
import threading
import time

import numpy as np
from conftest import wait_until


def ready_service(service):
    """An observed service with a left-arm draft whose preview plans for about two seconds (approach + IK chain)."""
    service.observe()
    service.set_draft({'observation_id': service.frame.id, 'arms': {'left': {
        'path': {'mode': 'waypoints', 'points': [[.28, 0., .12], [.31, 0., .12]]}, 'speed': .05,
        'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 180, 0, 0]]}}}})
    return service


def start_preview(service):
    """Run ``service.preview()`` in a thread; returns the thread and the list that receives its result or error."""
    outcome = []

    def work():
        try:
            outcome.append(service.preview())
        except ValueError as exc:
            outcome.append(exc)
    thread = threading.Thread(target=work)
    thread.start()
    return thread, outcome


def joints(service):
    return {arm: service.backend.runtime.joints_deg(arm).tolist() for arm in ('left', 'right')}


def test_preview_reports_stage_and_can_be_cancelled_without_motion(piper_service):
    s = ready_service(piper_service)
    before = joints(s)
    thread, outcome = start_preview(s)
    try:
        wait_until(lambda: s.execution.get('stage') == 'ik')
        status = s.status()['execution']
        assert status['state'] == 'planning' and status['kind'] == 'preview'
        assert status['arm'] == 'left' and status['total'] > 0 and 0 <= status['done'] <= status['total']
        assert s.cancel()['state'] == 'cancelling'
    finally:
        thread.join(30)
    assert not thread.is_alive()
    assert isinstance(outcome[0], ValueError) and '取消' in str(outcome[0])
    assert s.preview_data is None
    assert s.execution['state'] == 'cancelled' and s.execution['kind'] == 'preview'
    assert joints(s) == before


def test_a_cancel_at_any_planning_stage_leaves_no_preview_to_execute(piper_service):
    """Whether the cancel lands early or after the last planning checkpoint, nothing can be executed afterwards."""
    s = ready_service(piper_service)
    draft = s.draft
    landed = set()
    for stage in ('approach', 'ik', 'retiming', 'collision'):
        s.set_draft(draft)
        thread, outcome = start_preview(s)
        while s.execution['state'] == 'planning' and s.execution.get('stage') != stage:
            time.sleep(.0002)                           # some stages last only milliseconds
        answer = s.cancel()
        thread.join(30)
        assert s.preview_data is None, stage
        if answer['state'] == 'cancelling':
            landed.add(stage)
            assert isinstance(outcome[0], ValueError) and '取消' in str(outcome[0])
            assert s.execution['state'] == 'cancelled'
        s.recover()
    # The two long stages are always caught while they run.
    assert {'ik', 'retiming'} <= landed
    assert np.allclose(np.r_[s.backend.runtime.joints_deg('left'), s.backend.runtime.joints_deg('right')], 0.)
