"""Automatic preparation before planning: restoring control, refreshing a stale observation, retrying once.

Restoring CAN control from drag-teach mode and re-enabling a protection-stopped arm need those states on a real
arm; they are covered by tests/hardware/test_backend_on_robot.py. Here the real PiPER-X backend runs on the
kinematic simulator, whose arms are always enabled and under CAN control, and the service runs on top of it.
"""
from __future__ import annotations

import copy
import threading
import time

import numpy as np
import pytest
from test_dispatch import joint_plan, run, single_joint

from urai.backend import Cancelled, RobotStateError
from urai.service import DispatchCancellation, Service


def test_prepare_with_ready_arms_restores_nothing_and_sends_nothing(piper_backend):
    rt = piper_backend.runtime
    progress = []
    assert piper_backend.prepare(DispatchCancellation(), lambda **detail: progress.append(detail)) == {'restored_arms': []}
    assert progress == []
    for arm in ('left', 'right'):
        np.testing.assert_array_equal(rt.joints_deg(arm), np.zeros(6))
        assert rt.last_gripper_command[arm] is None


def test_prepare_after_cancellation_refuses_before_reading_the_arms(piper_backend):
    cancel = DispatchCancellation()
    cancel.set()
    with pytest.raises(Cancelled):
        piper_backend.prepare(cancel, lambda **detail: None)


def drafted_service(backend):
    """A service with a fresh observation and a short left-arm draft in front of the arm."""
    service = Service(backend)
    service.observe()
    service.set_draft({'observation_id': service.frame.id, 'arms': {'left': {
        'path': {'mode': 'waypoints', 'points': [[.30, 0., .10], [.30, .03, .10]]}, 'speed': .03,
        'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, -90], [1, 180, 0, -90]]},
        'gripper_events': [{'s': 1, 'opening_mm': 70}]}}})
    return service


def turn_left_arm(backend, degrees=3.):
    """Move the left arm's first joint through the backend, as a previous operation would have."""
    assert run(backend, joint_plan(backend, [0., .5], single_joint(backend, [0., degrees])))[0]['completed']


def test_preview_refreshes_a_stale_observation_and_keeps_the_world_draft(piper_backend):
    service = drafted_service(piper_backend)
    old_id, old_revision, arms = service.frame.id, service.revision, copy.deepcopy(service.draft['arms'])
    turn_left_arm(piper_backend)
    plan = service.preview(expected_revision=old_revision, expected_cancel_epoch=0)
    assert service.frame.id != old_id and plan['observation_id'] == service.frame.id
    assert plan['revision'] == service.revision > old_revision and service.draft['arms'] == arms
    assert plan['preparation']['observation_refreshed'] and plan['observation']['id'] == service.frame.id
    assert plan['start']['left']['joints_deg'][0] == pytest.approx(3., abs=.01)
    assert service.cancel_epoch == 0


def test_automatic_preparation_settings_are_shared_and_strict_when_disabled(piper_backend):
    service = drafted_service(piper_backend)
    service.set_settings({'auto_prepare': {'refresh_observation': False, 'restore_can': False}})
    assert service.settings()['auto_prepare'] == {'refresh_observation': False, 'restore_can': False}
    turn_left_arm(piper_backend)
    with pytest.raises(ValueError, match='observation'):
        service.preview()
    with pytest.raises(ValueError):
        service.set_settings({'auto_prepare': {'restore_can': 'true'}})


def test_cancel_during_the_refresh_publishes_neither_an_observation_nor_a_plan(piper_backend):
    service = drafted_service(piper_backend)
    turn_left_arm(piper_backend)
    old_id, old_draft = service.frame.id, copy.deepcopy(service.draft)
    outcome = []

    def preview():
        try:
            outcome.append(service.preview())
        except ValueError as exc:
            outcome.append(exc)

    thread = threading.Thread(target=preview)
    thread.start()
    deadline = time.monotonic()+10.
    while service.execution.get('stage') != 'refresh':   # the synthetic head camera renders for about 0.3 s
        assert time.monotonic() < deadline and thread.is_alive()
        time.sleep(.001)
    service.cancel()
    thread.join(10)
    assert isinstance(outcome[0], ValueError) and '取消' in str(outcome[0])
    assert service.frame.id == old_id and service.draft == old_draft and service.preview_data is None
    assert service.execution['state'] == 'cancelled'


class PlanningWatch(threading.Thread):
    """Counts planning attempts of a running preview and moves the left arm during the IK of the chosen ones.

    The arm is moved by commanding its driver directly, the way a second program on the bus would.
    """

    def __init__(self, service, move_on):
        super().__init__(daemon=True)
        self.service, self.move_on = service, move_on
        self.attempts = 0
        self.done = threading.Event()

    def run(self):
        driver = self.service.backend.runtime.arms['left']
        solving = False
        while not self.done.is_set():
            stage = self.service.execution.get('stage')
            if stage == 'ik' and not solving:
                solving = True
                self.attempts += 1
                if self.attempts in self.move_on:
                    target = driver.joints()
                    target[0] += 3.
                    driver.command_joints(target)
            elif stage in ('prepare', 'refresh', 'compile'):
                solving = False
            time.sleep(.002)


@pytest.mark.parametrize('persistent', [False, True])
def test_preparation_retries_once_when_the_arm_moves_during_planning(piper_backend, persistent):
    service = drafted_service(piper_backend)
    watch = PlanningWatch(service, move_on={1, 2} if persistent else {1})
    watch.start()
    try:
        if persistent:
            with pytest.raises(RobotStateError, match='moved since preview'):
                service.preview()
            assert service.execution['state'] == 'error' and service.preview_data is None
        else:
            plan = service.preview()
            assert plan['id'] == service.preview_data['id']
            assert plan['preparation']['observation_refreshed']
            assert plan['start']['left']['joints_deg'][0] == pytest.approx(3., abs=.01)
    finally:
        watch.done.set()
        watch.join(2)
    assert watch.attempts == 2
