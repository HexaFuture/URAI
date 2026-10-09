"""The grasp-line reachability search against the PiPER-X kinematics of the example rig; nothing moves."""
import numpy as np
import pytest
from urai.backend import SimulationBackend
from urai.grasp import grasp_from_line, grasp_terms
from urai.grasp_solver import append_grasp_lift, line_rotation, solve_line_grasp
from urai.trajectory import compile_arm

TOLERANCE = {'position_mm': 10., 'orientation_deg': 3.}
FAR_OFFSET_XY = np.array([.5551, -.2817])      # 0.62 m from the base: past the upright reach


def reach_probe(planner, arm):
    def probe(xyz, rotation, seed):
        return planner.pose_error(arm, xyz, rotation, seed, TOLERANCE)
    return probe


def test_line_axis_and_level_fingertips_survive_every_tilt():
    for yaw in [-130., -90., 0., 35., 90.]:
        expected = np.array([np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw)), 0.])
        for tilt in [0., 20., 35., 50., 65., 75.]:
            for side in [-1, 1]:
                r = line_rotation(yaw, tilt, side)
                np.testing.assert_allclose(r[:, 0], expected, atol=1e-12)
                np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
                assert np.linalg.det(r) == pytest.approx(1.)
                assert r[2, 2] < 0  # a side approach keeps a downward pitch


def test_a_near_line_is_grasped_straight_down_and_closes_before_the_lift(planner, working_start):
    contact = np.array([.30, -.05, .02])
    result = solve_line_grasp(contact, -90., planner.models['left'].base_xy, .10, .01, reach_probe(planner, 'left'),
                              working_start['left']['joints_deg'])
    assert result['mode'] == 'top_down' and result['tilt_deg'] == 0
    np.testing.assert_allclose(result['lift']-result['target'], [0, 0, .04])
    np.testing.assert_allclose(result['target']-result['rotation'][:, 1]*.01, contact, atol=1e-12)


@pytest.mark.parametrize('arm', ['left', 'right'])
def test_a_far_line_tilts_but_keeps_the_jaw_axis_level_and_the_fingertip_anchor(arm, planner, working_start):
    model = planner.models[arm]
    contact = np.r_[np.asarray(model.base_xy)+FAR_OFFSET_XY, .0236]
    result = solve_line_grasp(contact, -90., model.base_xy, .10, .01, reach_probe(planner, arm),
                              working_start[arm]['joints_deg'])
    assert result['tilt_deg'] > 0 and result['mode'] == 'tilted_side'
    assert abs(result['rotation'][2, 0]) < 1e-10                    # both fingertips at one height
    np.testing.assert_allclose(result['target']-result['rotation'][:, 1]*.01, contact, atol=1e-12)
    np.testing.assert_allclose(result['pre']-result['target'], -result['rotation'][:, 2]*.10, atol=1e-12)
    # Every pose of the chosen candidate solves within the solver's tolerance on one IK branch.
    q = np.asarray(working_start[arm]['joints_deg'], dtype=float)
    for point in (result['target'], result['pre'], result['target'], result['lift']):
        position_error, rotation_error, solved = planner.pose_error(arm, point, result['rotation'], q, TOLERANCE)
        assert position_error <= 10. and rotation_error <= 3.
        q = solved


def test_a_line_out_of_reach_is_refused_with_both_wrist_modes_named(planner, working_start):
    with pytest.raises(ValueError, match='竖直及倾斜侧抓均不可达'):
        solve_line_grasp(np.array([.95, .30, .02]), -90., planner.models['left'].base_xy, .10, .01,
                         reach_probe(planner, 'left'), working_start['left']['joints_deg'])


def test_lift_keeps_close_at_contact_not_end_of_lift():
    d = {'lift_xyz': [.5, 0, .06], 'arm': {'path': {'points': [[.5, 0, .12], [.5, 0, .02]]},
         'orientation': {'points': [[0, 180, 0, 0], [1, 180, 0, 0]]},
         'gripper_events': [{'s': 0, 'opening_mm': 70}, {'s': 1, 'opening_mm': 0, 'wait_for_arrival': True}]}}
    append_grasp_lift(d)
    assert d['arm']['gripper_events'][1]['s'] == .5
    assert d['arm']['path']['points'][1] == [.5, 0, .02]
    assert d['arm']['path']['points'][-1] == [.5, 0, .06]


def test_a_far_line_plans_its_whole_trajectory_without_motion(planner, working_start):
    model = planner.models['left']
    planner.table_checks = False
    result = grasp_from_line(SimulationBackend().capture(), [[380, 220], [410, 220]],
                             surface_xyz=[*(np.asarray(model.base_xy)+FAR_OFFSET_XY), .0336], preferred_yaw_deg=-90.,
                             base_xy=model.base_xy, reach_probe=reach_probe(planner, 'left'),
                             seed_joints=working_start['left']['joints_deg'], clearance_m=.10, speed_m_s=.10,
                             **grasp_terms('left', model, False))
    append_grasp_lift(result)
    compiled = compile_arm(result['arm'], working_start['left'])
    plan = planner.plan({'left': compiled}, working_start, pose_tolerance=TOLERANCE, motion_profile='throw')
    assert plan['duration_s'] > 0
    close = next(e for e in plan['arms']['left']['gripper_events'] if e.get('opening_mm') == 0)
    assert close['hold_end_time_s'] < plan['duration_s']


def test_a_far_release_plans_a_position_only_descent_without_the_solver_or_a_lift(planner, working_start):
    model = planner.models['left']
    planner.table_checks = False
    result = grasp_from_line(SimulationBackend().capture(), [[380, 220], [380, 220]], endpoint_action='release',
                             surface_xyz=[*(np.asarray(model.base_xy)+FAR_OFFSET_XY), .0036],
                             reach_probe=reach_probe(planner, 'left'), base_xy=model.base_xy, speed_m_s=.1,
                             clearance_m=.1)
    assert result['grasp_mode'] == 'position_only' and 'lift_xyz' not in result
    plan = planner.plan({'left': compile_arm(result['arm'], working_start['left'])}, working_start,
                        pose_tolerance=TOLERANCE, motion_profile='throw')
    assert plan['duration_s'] > 0
    [event] = plan['arms']['left']['gripper_events']
    assert event['opening_mm'] == 70 and event['wait_for_arrival']
