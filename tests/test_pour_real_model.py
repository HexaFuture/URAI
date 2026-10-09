"""The pour candidate search against the PiPER-X kinematics of the example rig; nothing moves."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.pour import EXACT_DEG, EXACT_MM, pour_route, side_grasp_rotation

TOLERANCE = {'position_mm': 30., 'orientation_deg': 60.}
CUP_HEIGHT_M = .076
GRASP_FRACTION = .45


def disc(center_xy, top, radius, n=64):
    """A ``locate_object``-shaped rim of ``n`` points at height ``top`` around ``center_xy``."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.c_[center_xy[0] + radius * np.cos(angles), center_xy[1] + radius * np.sin(angles), np.full(n, top)]
    return {'center': np.asarray(center_xy, dtype=float), 'top': float(top), 'points': points}


class RecordedProbe:
    """The arm's real reach probe (``PiperPlanner.pose_error``); every call and its answer are kept for the test."""

    def __init__(self, planner, arm, seed):
        self.planner, self.arm, self.seed = planner, arm, np.asarray(seed, dtype=float)
        self.calls = []

    def __call__(self, xyz, rotation, seed):
        answer = self.planner.pose_error(self.arm, xyz, rotation, self.seed if seed is None else seed, TOLERANCE)
        self.calls.append({'seed': None if seed is None else np.asarray(seed, dtype=float), 'answer': answer})
        return answer


def right_arm_route(planner, start, reach_m, sideways_m=.20):
    """Pour a cup ``reach_m`` straight ahead of the right base into a bowl ``sideways_m`` beside it."""
    model = planner.models['right']
    base = np.asarray(model.base_xy, dtype=float)
    table_z = model.table_z_mm / 1000
    cup = disc(base + [reach_m, 0.], table_z + CUP_HEIGHT_M, .035)
    target = disc(base + [reach_m, sideways_m], table_z + .027, .06)
    probe = RecordedProbe(planner, 'right', start['right']['joints_deg'])
    route = pour_route(cup, target, table_z, model.fingertip_bias_mm, base, 100., 2.5, GRASP_FRACTION, .08, .05,
                       reach_probe=probe, pose_tolerance=(TOLERANCE['position_mm'], TOLERANCE['orientation_deg']))
    return route, probe


def test_the_right_arm_pours_a_cup_at_0_52_m_with_exact_key_poses(planner, working_start):
    result, probe = right_arm_route(planner, working_start, .52)
    assert result['reach_error_mm'] <= EXACT_MM and result['reach_error_deg'] <= EXACT_DEG
    assert result['approach_offset_deg'] in (0., -60., 60., -30., 30.)
    assert result['pour_offset_deg'] in (90., -90., 45., -45., 0.)
    starts = [call['seed'] is None for call in probe.calls]
    assert starts[0] and sum(starts) <= 5    # one chain per approach candidate, each restarted from the start pose
    # Every waypoint of the chosen route solves exactly from the previous solution, not just the probed keys.
    seed = np.asarray(working_start['right']['joints_deg'], dtype=float)
    for point, keyframe in zip(result['arm']['path']['points'], result['arm']['orientation']['points']):
        rotation = Rotation.from_euler('xyz', keyframe[1:], degrees=True).as_matrix()
        position_error, rotation_error, seed = planner.pose_error('right', point, rotation, seed, TOLERANCE)
        assert position_error <= EXACT_MM and rotation_error <= EXACT_DEG


def test_the_search_chains_each_candidate_from_the_previous_solution_and_stops_at_the_first_exact_one(
        planner, working_start):
    """A candidate restarts from the start pose (seed None) at its pregrasp, then pregrasp, grasp and lift are
    solved in order; every pour direction of that candidate restarts from its lift solution."""
    result, probe = right_arm_route(planner, working_start, .52)
    lift, last_pour_start = None, None
    for index, call in enumerate(probe.calls):
        if call['seed'] is None:
            start, lift = index, None
            continue
        if index == start+3:
            lift = probe.calls[start+2]['answer'][2]
        if lift is not None and np.array_equal(call['seed'], lift):
            last_pour_start = index
        else:
            np.testing.assert_array_equal(call['seed'], probe.calls[index-1]['answer'][2])
    # The search ended on the first chain whose every pose is exact.
    final = probe.calls[last_pour_start:]
    assert all(call['answer'][0] <= EXACT_MM and call['answer'][1] <= EXACT_DEG for call in final)
    assert (result['reach_error_mm'], result['reach_error_deg']) == pytest.approx(
        (max(call['answer'][0] for call in probe.calls[start:start+3]+final),
         max(call['answer'][1] for call in probe.calls[start:start+3]+final)))


def test_the_right_arm_grasps_at_0_45_m_exactly_with_the_tool_pitched_down(planner, working_start):
    """At 0.45 m a level grasp low above the table pins j4 near its limit; pitching the tool nose-down about the
    closing axis frees the wrist, the radial approach then solves exactly and ends the search."""
    result, probe = right_arm_route(planner, working_start, .45)
    assert result['reach_error_mm'] <= EXACT_MM and result['reach_error_deg'] <= EXACT_DEG
    assert result['approach_offset_deg'] == 0 and result['tool_pitch_deg'] in (15., 30.)
    assert 2 <= sum(call['seed'] is None for call in probe.calls) <= 3   # level chain(s), then the first pitched one


def test_the_right_arm_cannot_roll_the_tool_y_up_at_the_grasp(planner, working_start):
    model = planner.models['right']
    base = np.asarray(model.base_xy, dtype=float)
    seed = working_start['right']['joints_deg']
    natural = side_grasp_rotation(np.array([1., 0.]))
    # Rolling the tool Y up needs 180 degrees about the tool axis; j6 stops at +-120, leaving the pose 60 short.
    mirrored = natural @ Rotation.from_euler('z', 180, degrees=True).as_matrix()
    for reach_m in (.45, .52):
        grasp = np.r_[base + [reach_m, 0.], model.table_z_mm / 1000 + CUP_HEIGHT_M * GRASP_FRACTION - model.fingertip_bias_mm / 1000]
        _, natural_error, _ = planner.pose_error('right', grasp, natural, seed, TOLERANCE)
        _, mirrored_error, _ = planner.pose_error('right', grasp, mirrored, seed, TOLERANCE)
        assert natural_error <= 6
        assert mirrored_error == pytest.approx(60, abs=1)


def test_a_cup_out_of_reach_is_refused_naming_the_best_candidate(planner, working_start):
    with pytest.raises(ValueError, match='倒水路线不可达：最好的候选（接近方向偏离径向 .*工具俯角 .*误差 .*换另一只臂'):
        right_arm_route(planner, working_start, .95)
