"""Pure pause gripper events and orientation-step sample refinement."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.backend import SimulationBackend
from urai.trajectory import compile_arm


def rotating_spec(pause):
    """A straight path whose orientation turns 100 degrees within 5 % of its progress."""
    return {'path': {'mode': 'waypoints', 'points': [[.2, 0., .2], [.3, 0., .2]]},
            'speed': .05,
            'orientation': {'mode': 'keyframes',
                            'points': [[0, 180, 0, 0], [.5, 180, 0, 0], [.55, 180, 0, 100], [1, 180, 0, 100]]},
            'gripper_events': [pause]}


def test_pause_event_dwells_in_place_without_an_opening():
    pause = {'s': .55, 'hold_s': 1.5, 'wait_for_arrival': True}
    compiled = compile_arm(rotating_spec(pause), SimulationBackend().state()['left'])
    event, = compiled['gripper_events']
    assert 'opening_mm' not in event
    assert event['wait_for_arrival'] is True
    assert event['hold_end_time_s'] - event['time_s'] >= 1.5 - 1e-9
    index = int(np.where(compiled['time_s'] == event['time_s'])[0][0])
    np.testing.assert_array_equal(compiled['xyz'][index], compiled['xyz'][index + 1])
    np.testing.assert_array_equal(compiled['rotation_matrices'][index], compiled['rotation_matrices'][index + 1])
    assert index in compiled['stop_indices'] and index + 1 in compiled['stop_indices']
    assert compiled['requested_speed_m_s'][index] == 0 and compiled['requested_speed_m_s'][index + 1] == 0


def test_in_place_rotation_is_refined_below_the_orientation_step():
    compiled = compile_arm(rotating_spec({'s': .55, 'hold_s': 1.}), SimulationBackend().state()['left'])
    rotations = Rotation.from_matrix(compiled['rotation_matrices'])
    steps = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
    assert steps.max() <= 2.5 + 1e-6
    total = np.rad2deg((rotations[0].inv() * rotations[-1]).magnitude())
    assert total == pytest.approx(100, abs=1e-6)
    assert len(compiled['s']) <= 2001
    assert compiled['time_s'][-1] >= 100 / 30


def test_pause_event_requires_a_hold_and_rejects_verification():
    state = SimulationBackend().state()['left']
    with pytest.raises(ValueError, match='hold_s'):
        compile_arm(rotating_spec({'s': .55}), state)
    with pytest.raises(ValueError, match='hold_s'):
        compile_arm(rotating_spec({'s': .55, 'hold_s': 0}), state)
    with pytest.raises(ValueError, match='verify'):
        compile_arm(rotating_spec({'s': .55, 'hold_s': 1., 'wait_for_arrival': True, 'verify': 'holding'}), state)


def test_simulation_execution_keeps_the_gripper_during_a_pause():
    from urai.service import DispatchCancellation
    from urai.trajectory import compile_arms
    backend = SimulationBackend()
    backend.openings['left'] = 20.
    start = backend.state()
    spec = {'path': {'mode': 'waypoints', 'points': [[.25, .12, .24], [.27, .12, .24]]}, 'speed': .1,
            'orientation': {'mode': 'keyframes',
                            'points': [[0, 180, 0, 0], [.5, 180, 0, 0], [.6, 180, 0, 8], [1, 180, 0, 8]]},
            'gripper_events': [{'s': .6, 'hold_s': .2, 'wait_for_arrival': True}]}
    plan = backend.plan(compile_arms({'left': spec}, start), start)
    result = backend.execute(plan, start, DispatchCancellation(), lambda *args: None)
    assert result['completed']
    assert backend.openings['left'] == 20.
    np.testing.assert_allclose(backend.positions['left'], [.27, .12, .24], atol=1e-9)
