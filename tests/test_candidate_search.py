"""Approach candidates on the real arm models: per-arm pruning, the IK memo and immediate task failures."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.approach import with_approaches
from urai.backend import ApproachError, TaskError
from urai.backend.model_checks import solve_ik
from urai.trajectory import compile_arms

#: Both arms start tool down in front of their own bases.
START = {'left': [.35, .05, .10], 'right': [.35, -.64, .10]}


def tool_down(yaw_deg=0.):
    return Rotation.from_euler('xyz', [180., 0., yaw_deg], degrees=True).as_matrix()


def arm_state(model, xyz):
    target = np.eye(4); target[:3, :3] = tool_down(); target[:3, 3] = xyz
    q, pe, _ = solve_ik(model, target, np.array([0., 80., -80., 0., 0., 0.]))
    assert pe < 1e-6
    pose = model.fk_tcp_world(q)
    return {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            'joints_deg': q.tolist(), 'gripper_mm': 0.}


def start_state(models):
    return {arm: arm_state(models[arm], xyz) for arm, xyz in START.items()}


def segment(a, b, yaw_deg=0.):
    """A tool-down straight line at 5 cm/s with a fixed jaw yaw."""
    return {'path': {'mode': 'waypoints', 'points': [a, b]}, 'speed': .05,
            'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, yaw_deg], [1, 180, 0, yaw_deg]]}}


def search(planner, specs, start):
    """Plan ``specs`` and return (result or exception, the approach candidates tried in order)."""
    tried = []

    def progress(**detail):
        if detail.get('stage') == 'approach':
            tried.append((detail['candidate'], detail.get('routes')))
    try:
        return planner.plan(compile_arms(specs, start), start, progress=progress), tried
    except ValueError as exc:
        return exc, tried


# The left task turns the jaws to yaw 170 deg. A Cartesian connector slerps through yaw 85 deg, which this
# wrist cannot reach tool down (j6 would leave its +-120 deg range), on the straight line and on the lifted
# one alike. Only the joint-space route, which turns j6 the other way round, gets there.
TURNED_LEFT = segment([.35, .10, .10], [.40, .10, .10], 170.)
SHIFTED_RIGHT = segment([.38, -.64, .10], [.42, -.64, .10])


def test_an_approach_ik_failure_prunes_every_combination_with_that_route(planner):
    start = start_state(planner.models)
    plan, tried = search(planner, {'left': TURNED_LEFT, 'right': SHIFTED_RIGHT}, start)
    assert plan['approach_routes'] == {'left': 'joint', 'right': 'direct'}
    # (direct, direct) fails in the left approach IK, so (direct, lift) and (direct, joint) are skipped;
    # (lift, direct) fails the same way and takes (lift, lift) and (lift, joint) with it.
    assert tried == [(1, {'left': 'direct', 'right': 'direct'}), (4, {'left': 'lift', 'right': 'direct'}),
                     (7, {'left': 'joint', 'right': 'direct'})]
    assert planner._ik_cache is None   # the per-plan IK memo does not outlive the plan


def test_ik_memo_returns_the_first_solve_for_a_repeated_target_and_seed(planner):
    model = planner.models['left']
    target = np.eye(4); target[:3, :3] = tool_down(); target[:3, 3] = [.38, .05, .10]
    seed = np.asarray(start_state(planner.models)['left']['joints_deg'])
    tolerance = {'position_mm': 10., 'orientation_deg': 3.}
    planner._ik_cache = {}
    first = planner._cached_ik('left', model, target, seed, tolerance, False)
    assert planner._cached_ik('left', model, target.copy(), seed.copy(), tolerance, False) is first
    assert len(planner._ik_cache) == 1
    # Another seed, another arm or the free-orientation solve are separate entries.
    planner._cached_ik('left', model, target, seed+.5, tolerance, False)
    planner._cached_ik('left', model, target, seed, tolerance, True)
    assert len(planner._ik_cache) == 3
    # Outside a plan nothing is memoised.
    planner._ik_cache = None
    again = planner._cached_ik('left', model, target, seed, tolerance, False)
    assert again is not first
    np.testing.assert_array_equal(again[0], first[0])


def test_a_task_segment_ik_failure_stops_the_search_immediately(planner):
    start = start_state(planner.models)
    error, tried = search(planner, {'left': segment([.38, .05, .10], [.95, .05, .10])}, start)
    assert isinstance(error, TaskError) and 'path at' in str(error)
    assert [number for number, _ in tried] == [1]


def test_a_task_phase_collision_stops_the_search_immediately(planner):
    # Both wrists end 9 cm apart between the bases.
    start = start_state(planner.models)
    specs = {'left': segment([.40, -.05, .10], [.40, -.25, .10]), 'right': segment([.40, -.54, .10], [.40, -.34, .10])}
    error, tried = search(planner, specs, start)
    assert isinstance(error, TaskError) and 'would hit the right arm' in str(error)
    assert [number for number, _ in tried] == [1]


def test_approach_phase_errors_carry_their_arm_and_task_errors_do_not(planner):
    start = start_state(planner.models)
    compiled = compile_arms({'left': TURNED_LEFT}, start)
    with pytest.raises(ApproachError) as exc:
        planner._plan(with_approaches(compiled, start, {'left': 'direct'}), start)
    assert exc.value.arm == 'left' and 'left path at' in str(exc.value)
    compiled = compile_arms({'left': segment([.38, .05, .10], [.95, .05, .10])}, start)
    with pytest.raises(TaskError) as exc:
        planner._plan(with_approaches(compiled, start, {'left': 'direct'}), start)
    assert not isinstance(exc.value, ApproachError)
