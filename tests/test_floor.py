"""The planner lifts samples whose solved fingertip would fall below the table floor of the arm model."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.backend import FloorError
from urai.backend.model_checks import floor_world_mm, solve_ik
from urai.trajectory import compile_arms

#: Where the left arm's descending path runs, tool down.
FLOOR_XY = (.30, 0.)


def tool_down_state(model, xyz):
    target = np.eye(4); target[:3, :3] = Rotation.from_euler('xyz', [180., 0., 0.], degrees=True).as_matrix(); target[:3, 3] = xyz
    q, pe, _ = solve_ik(model, target, np.array([0., 80., -80., 0., 0., 0.]))
    assert pe < 1e-6
    pose = model.fk_tcp_world(q)
    return {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            'joints_deg': q.tolist(), 'gripper_mm': 0.}


def start_state(models, xy=FLOOR_XY, z=.05):
    """Left arm tool down above ``xy``; the right arm parked tool down in front of its own base."""
    return {'left': tool_down_state(models['left'], [*xy, z]), 'right': tool_down_state(models['right'], [.35, -.64, .10])}


def descending_spec(bottom_z, xy=FLOOR_XY):
    return {'path': {'mode': 'waypoints', 'points': [[*xy, .05], [*xy, bottom_z]]}, 'speed': .05,
            'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, 0], [1, 180, 0, 0]]}}


def tip_heights_mm(model, joints):
    return np.array([model.fk_tcp_world(q)[2, 3]*1000.+model.fingertip_bias_mm for q in joints])


def test_solved_fingertip_is_lifted_above_the_table_floor(planner):
    # Floor: bias-corrected tip >= 3.5 - 5 = -1.5 mm, planned with a 0.5 mm margin. The drawn bottom is at -5.5 mm.
    model = planner.models['left']
    floor = floor_world_mm(model)+planner.FLOOR_MARGIN_MM
    start = start_state(planner.models)
    compiled = compile_arms({'left': descending_spec(-.0055)}, start)
    drawn = compiled['left']['xyz'].copy()
    plan = planner.plan(compiled, start)
    left = plan['arms']['left']
    assert tip_heights_mm(model, left['joints_deg']).min() >= floor-1e-9
    # Only the offending samples move, up to the floor, and the preview path shows it.
    offending = drawn[:, 2]*1000. < floor
    assert offending.sum() > 3
    np.testing.assert_array_equal(left['xyz'][~offending], drawn[~offending])
    np.testing.assert_allclose(left['xyz'][offending, 2]*1000., floor, atol=1e-6)
    np.testing.assert_array_equal(left['xyz'][:, :2], drawn[:, :2])
    note = next(note for note in plan['notes'] if '贴近桌面' in note)
    assert f'{offending.sum()} 个采样点' in note and '4.5 mm' in note


def test_lift_accepts_a_fingertip_exactly_on_the_floor(planner):
    xy = (.30, .10)
    start = start_state(planner.models, xy)
    plan = planner.plan(compile_arms({'left': descending_spec(-.0055, xy)}, start), start)
    floor = floor_world_mm(planner.models['left'])+planner.FLOOR_MARGIN_MM
    assert tip_heights_mm(planner.models['left'], plan['arms']['left']['joints_deg']).min() >= floor-1e-6


def test_samples_above_the_floor_keep_their_drawn_height(planner):
    start = start_state(planner.models)
    compiled = compile_arms({'left': descending_spec(.02)}, start)
    drawn = compiled['left']['xyz'].copy()
    plan = planner.plan(compiled, start)
    np.testing.assert_array_equal(plan['arms']['left']['xyz'], drawn)
    np.testing.assert_allclose(plan['arms']['left']['xyz'][-1], [*FLOOR_XY, .02])
    assert not any('贴近桌面' in note for note in plan['notes'])


def test_floor_failure_ends_the_approach_search_at_once(planner):
    """A planner allowed no lift rounds refuses the dip with FloorError on the first route; no other route is tried."""
    planner.FLOOR_LIFT_ROUNDS = 0
    start = start_state(planner.models, z=.10)   # above the task start, so an approach is needed
    candidates = []

    def progress(**detail):
        if detail.get('stage') == 'approach':
            candidates.append(detail['candidate'])
    with pytest.raises(FloorError, match='低于桌面'):
        planner.plan(compile_arms({'left': descending_spec(-.0055)}, start), start, progress=progress)
    assert candidates == [1]
