"""Automatic return home after a drafted motion, on the synthetic backend and on the PiPER-X backend over the
kinematic simulator. Disturbances are applied the way a person would: by moving an arm through its own driver."""
import threading

import numpy as np
import pytest
from conftest import line_draft, wait_until

GRIP = [{'s': 1., 'opening_mm': 20., 'hold_s': .5}]


def observed(service):
    service.observe()
    return service


def run(service, draft):
    service.set_draft(draft)
    service.execute(service.preview()['id'])


@pytest.mark.parametrize('arms', [['left'], ['left', 'right']])
def test_a_completed_motion_returns_the_moved_arms_home_keeping_the_grippers(sim_service, arms):
    s = observed(sim_service)
    backend = s.backend
    untouched = {arm: backend.state()[arm] for arm in ('left', 'right') if arm not in arms}
    run(s, line_draft(s, arms, [.02, 0., 0.], gripper_events=GRIP))
    s.worker.join(30)
    assert s.execution['state'] == 'completed'
    result = s.execution['result']
    assert result['return_home']['completed'] is True
    assert set(result['return_home_plan']['arms']) == set(arms)
    for arm in arms:
        assert result['return_home_plan']['arms'][arm]['gripper_events'] == []
        np.testing.assert_allclose(backend.positions[arm], backend.home_positions[arm], atol=1e-9)
        assert backend.openings[arm] == 20.
    for arm, state in untouched.items():
        assert backend.state()[arm] == state


def test_a_cancelled_motion_does_not_return_home(sim_service):
    s = observed(sim_service)
    home = s.backend.home_positions['left'].copy()
    run(s, line_draft(s, ['left'], [.05, 0., 0.]))
    wait_until(lambda: s.backend.positions['left'][0]-home[0] > .02)
    s.cancel()
    s.worker.join(10)
    assert s.execution['state'] == 'cancelled' and 'result' not in s.execution
    assert s.backend.positions['left'][0]-home[0] > .02


def test_a_failed_motion_does_not_return_home(piper_service):
    """The other arm is pushed while the left one moves: the run stops with an error, and the left arm stays
    where it was stopped instead of travelling home."""
    s = observed(piper_service)
    runtime = s.backend.runtime
    run(s, line_draft(s, ['left'], [.05, 0., 0.], speed=.02))
    wait_until(lambda: np.abs(runtime.joints_deg('left')).max() > 1.)
    runtime.arms['right'].command_joints(runtime.joints_deg('right')+[5., 0., 0., 0., 0., 0.])
    s.worker.join(30)
    assert s.execution['state'] == 'error' and 'right' in s.execution['error']
    assert 'result' not in s.execution and s.execution.get('stage') != 'return_home'
    assert np.abs(runtime.joints_deg('left')).max() > 1.


def test_homing_does_not_return_home_again(sim_service):
    s = observed(sim_service)
    run(s, line_draft(s, ['left'], [.05, 0., 0.]))
    wait_until(lambda: s.backend.positions['left'][0]-s.backend.home_positions['left'][0] > .01)
    s.cancel()
    s.worker.join(10)
    s.home(['left'])
    s.worker.join(30)
    assert s.execution['state'] == 'completed' and s.execution['kind'] == 'home'
    assert 'return_home' not in s.execution['result']
    np.testing.assert_allclose(s.backend.positions['left'], s.backend.home_positions['left'], atol=1e-9)


def test_a_stop_during_the_return_home_leaves_the_arm_where_it_stopped(sim_service):
    s = observed(sim_service)
    s.set_settings({'home_speed_m_s': .05})
    run(s, line_draft(s, ['left'], [.05, 0., 0.], speed=.05))
    wait_until(lambda: s.execution.get('stage') == 'return_home')
    s.cancel()
    s.worker.join(10)
    assert s.execution['state'] == 'cancelled' and 'result' not in s.execution
    assert np.linalg.norm(s.backend.positions['left']-s.backend.home_positions['left']) > .02


def test_a_failed_return_home_is_reported_as_an_error(piper_service):
    """The right arm keeps being pushed while the left one returns home: both attempts of the return see a
    moving robot, and the run ends in an error instead of claiming success."""
    s = observed(piper_service)
    s.set_settings({'motion_profile': 'throw', 'home_speed_m_s': .05})
    right = s.backend.runtime.arms['right']
    rest = right.joints()
    stop = threading.Event()

    def push():
        step = 0
        while not stop.is_set():
            step += 1
            right.command_joints(rest+[3. if step % 2 else 0., 0., 0., 0., 0., 0.])
            stop.wait(.2)

    pusher = threading.Thread(target=push)
    run(s, line_draft(s, ['left'], [.05, 0., 0.], speed=.05))
    wait_until(lambda: s.execution.get('stage') == 'return_home')
    pusher.start()
    try:
        s.worker.join(60)
    finally:
        stop.set()
        pusher.join(5)
    assert s.execution['state'] == 'error' and 'right' in s.execution['error']


def test_home_plans_keep_the_grippers_when_asked(sim_service, piper_backend):
    for backend in (sim_service.backend, piper_backend):
        start = backend.state()
        kept = backend.plan_home(start, ['left'], open_grippers=False)
        opened = backend.plan_home(start, ['left'], open_grippers=True)
        assert kept['arms']['left']['gripper_events'] == []
        assert [event['opening_mm'] for event in opened['arms']['left']['gripper_events']] == [70.]


def test_a_disturbance_as_the_return_starts_is_absorbed(piper_service):
    """An arm still settling (here: nudged by 1 degree) when the return home starts is planned for again,
    not treated as a failure."""
    s = observed(piper_service)
    s.set_settings({'motion_profile': 'throw'})
    driver = s.backend.runtime.arms['left']
    run(s, line_draft(s, ['left'], [.05, 0., 0.], speed=.05))
    wait_until(lambda: s.execution.get('stage') == 'return_home')
    driver.command_joints(driver.joints()+[1., 0., 0., 0., 0., 0.])
    s.worker.join(60)
    assert s.execution['state'] == 'completed' and s.execution['result']['return_home']['completed'] is True
    assert np.abs(driver.joints()).max() < .7
