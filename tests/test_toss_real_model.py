"""The overhand throw against the PiPER-X kinematics of the example rig; nothing moves."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.backend.model_checks import ik_errors, self_collision_reason, solve_ik
from urai.toss import (OVERHAND_CARRY_SPEED_M_S, OVERHAND_CARRY_STEP_DEG, OVERHAND_FOLLOW_AFTER_RELEASE_S,
                       OVERHAND_JOINT_SPEED_CEILING_DEG_S, OVERHAND_LINES, OVERHAND_RATE_SCALE, OVERHAND_STEP_S,
                       OVERHAND_TOOL_CONE_DEG, flight_distance, overhand_draft, overhand_release, overhand_samples,
                       release_window_s, toss_between)
from urai.transfer import place_transfer
from test_transfer import object_frame

TOLERANCE = {'position_mm': 30., 'orientation_deg': 60.}


def probe_for(planner, arm):
    def probe(xyz, rotation, seed):
        return planner.pose_error(arm, xyz, rotation, seed, TOLERANCE)
    return probe


def block(center_xy, table_z):
    """A ``locate_object``-shaped 4 cm block standing 2 cm tall at ``center_xy``."""
    top = table_z+.02
    corners = np.array([[-.02, -.02], [.02, -.02], [.02, .02], [-.02, .02]])
    return {'center': np.asarray(center_xy, dtype=float), 'top': top, 'width': .04, 'axis_deg': 0.,
            'points': np.c_[np.asarray(center_xy)+corners, np.full(4, top)]}


def left_arm_throw(planner, start, target_offset_xy):
    model = planner.models['left']
    base = np.asarray(model.base_xy, dtype=float)
    table_z = model.table_z_mm/1000
    found = block(base+[.40, .05], table_z)
    target = np.r_[base+np.asarray(target_offset_xy), table_z]
    placement = place_transfer(found, target, table_z, .08, model.fingertip_bias_mm, drop=True)
    draft = overhand_draft(found, placement, target, table_z, .08, .10, 0., 5., .08, base, model.fk_tcp_world,
                           probe_for(planner, 'left'), np.asarray(start['left']['joints_deg']), (30., 60.))
    return model, base, table_z, draft


def line_named(name):
    return next(line for line in OVERHAND_LINES if line['name'] == name)


def test_release_is_a_low_forward_thrust_whose_flight_lands_on_the_target(planner, working_start):
    model, base, table_z, draft = left_arm_throw(planner, working_start, [.60, 0.])
    line = line_named(draft['throw_line'])
    release = np.asarray(draft['target_xyz'])
    windup = np.asarray(draft['windup_xyz'])
    start, end = release_window_s(line)
    assert start-1e-9 <= draft['release_time_s'] <= end+1e-9
    assert .10 <= release[2]-table_z <= .31 and np.linalg.norm(release[:2]-base) < .46
    assert np.linalg.norm(windup[:2]-base) < .20 and windup[2]-table_z < .20      # folded low in front of the base
    assert draft['tool_off_deg'] <= OVERHAND_TOOL_CONE_DEG+1e-6 and abs(draft['release_elevation_deg']) < 30.
    assert .7 <= draft['toss_speed_m_s'] <= 1.25 and draft['landing_shortfall_m'] == 0.
    poses, joints, times, release_index = overhand_samples(model.fk_tcp_world, draft['release_joints_deg'][0],
                                                           draft['release_time_s'], line)
    tips = poses[:, :3, 3]
    velocities = np.gradient(tips, times, axis=0)*draft['rate_scale']
    landing = np.linalg.norm(tips[release_index, :2]-base)+flight_distance(tips[release_index], velocities[release_index], table_z)
    assert landing == pytest.approx(.60, abs=.03)
    for design in joints:
        assert model.table_check(design) is None and self_collision_reason(model, design) is None
    # The reference runs the arm's lag plus a launch window past the release instant, then stops short of the
    # line's end.
    assert times[-1] == pytest.approx(draft['release_time_s']+OVERHAND_FOLLOW_AFTER_RELEASE_S, abs=.02)
    assert times[-1] <= line['follow_s']+1e-9
    assert joints[-1, 1] < line['release_joints_deg'][1]+line['rates_deg_s'][1]*line['follow_s']-5.
    assert end+OVERHAND_FOLLOW_AFTER_RELEASE_S == pytest.approx(line['follow_s'])


def test_lines_run_at_the_firmware_joint_speed_ceiling_and_are_never_scaled_past_it(planner, working_start):
    # Commanding faster than the firmware follows only bends the joint-space line.
    for line in OVERHAND_LINES:
        assert np.abs(line['rates_deg_s']).max() == pytest.approx(OVERHAND_JOINT_SPEED_CEILING_DEG_S)
    assert OVERHAND_RATE_SCALE[1]*max(np.abs(l['rates_deg_s']).max() for l in OVERHAND_LINES) <= OVERHAND_JOINT_SPEED_CEILING_DEG_S+1e-9
    _, _, _, fast = left_arm_throw(planner, working_start, [.68, 0.])      # inside the fast line's full-speed window
    assert fast['throw_line'] == 'fast' and fast['landing_shortfall_m'] == 0.
    assert fast['rate_scale'] == pytest.approx(OVERHAND_RATE_SCALE[1])   # the requested 5 m/s cannot raise the rates


def test_targets_beyond_or_inside_the_lines_are_handled(planner, working_start):
    _, _, _, far = left_arm_throw(planner, working_start, [1.30, 0.])
    assert far['landing_shortfall_m'] > .05 and far['rate_scale'] == pytest.approx(OVERHAND_RATE_SCALE[1])
    _, _, _, near = left_arm_throw(planner, working_start, [.45, 0.])
    assert near['rate_scale'] < .95 and near['landing_shortfall_m'] == 0.
    with pytest.raises(ValueError, match='扔过头'):
        left_arm_throw(planner, working_start, [.33, 0.])


def test_planner_ik_follows_the_joint_line_from_the_hover_to_the_follow_through(planner, working_start):
    model, base, table_z, draft = left_arm_throw(planner, working_start, [.60, 0.])
    poses, joints, times, release_index = overhand_samples(model.fk_tcp_world, draft['release_joints_deg'][0],
                                                           draft['release_time_s'], line_named(draft['throw_line']))
    q, pe, re = ik_errors(model, poses[0], joints[0])
    assert pe < 1e-6 and re < 1e-6
    # Hover -> joint-space carry -> wind-up -> throw: the chained IK follows every designed configuration and no
    # pitch joint touches its limit on the way (touching it is what makes the IK jump branches).
    hover_joints = np.asarray(draft['hover_joints_deg'])
    k = draft['carry_samples']
    assert k > 0
    carry_joints = hover_joints+(joints[0]-hover_joints)*np.linspace(0, 1, k+2)[1:-1, None]
    points = np.asarray(draft['arm']['path']['points'])
    keys = draft['arm']['orientation']['points']
    designs = np.vstack([carry_joints, joints])
    limits = np.array([model.kin.limits_deg[i+1] for i in range(6)])
    assert (designs[:, 1:4] > limits[1:4, 0]+4.).all() and (designs[:, 1:4] < limits[1:4, 1]-4.).all()
    q = hover_joints.copy()
    worst = 0.
    for i, design in enumerate(designs):
        pose = np.eye(4)
        pose[:3, 3] = points[3+i]
        pose[:3, :3] = Rotation.from_euler('xyz', keys[2+i][1:], degrees=True).as_matrix()
        q, pe, re = solve_ik(model, pose, q, TOLERANCE)
        assert pe < 1. and re < 1.
        worst = max(worst, float(np.abs(q-design).max()))
        assert self_collision_reason(model, design) is None
    assert worst < 1.
    assert points[3:3+k, 2].min() >= points[1, 2]+.03


def test_out_of_reach_target_releases_late_but_keeps_the_full_launch_window_for_the_lagging_arm(planner, working_start):
    # With the release too close to the line's end the reference stops while the arm, a quarter second behind
    # it, is still short of the release pose; the jaws then open as it brakes and the object drops at its feet.
    model, base, table_z, draft = left_arm_throw(planner, working_start, [1.30, 0.])
    line = line_named(draft['throw_line'])
    assert draft['landing_shortfall_m'] > 0
    assert draft['release_time_s'] == pytest.approx(release_window_s(line)[1], abs=1e-6)
    times = np.asarray(draft['throw_times_s'])
    assert times[-1]-draft['release_time_s'] == pytest.approx(OVERHAND_FOLLOW_AFTER_RELEASE_S, abs=.02)
    assert times[-1] <= line['follow_s']+1e-9


@pytest.mark.parametrize('manual', [False, True])
def test_overhand_draft_runs_the_full_speed_line_and_lets_go_where_the_flight_lands_on_the_target(
        manual, planner, working_start):
    model = planner.models['left']
    base = np.asarray(model.base_xy, dtype=float)
    table_z = model.table_z_mm/1000
    kinematics = {'fk': model.fk_tcp_world, 'reach_probe': probe_for(planner, 'left'),
                  'seed_joints': working_start['left']['joints_deg']}
    target = np.r_[base+[.60, 0.], table_z]
    result = toss_between(object_frame(), [396, 226], target, table_z=table_z, base_xy=base, style='overhand',
                          toss_speed_m_s=5., lead_s=.06, **kinematics,
                          **({'grasp_pixels': [[300, 226], [490, 226]]} if manual else {}))
    assert result['toss_style'] == 'overhand' and result['placement'] == 'toss'
    # The target is re-levelled against the table fitted in the frame, like every landing point.
    np.testing.assert_allclose(result['toss_target_xyz'][:2], target[:2])
    target = np.asarray(result['toss_target_xyz'])
    bearing = float(np.degrees(np.arctan2(target[1]-base[1], target[0]-base[0])))
    forward = model.fk_tcp_world(OVERHAND_LINES[0]['release_joints_deg'])[:2, 3]-base
    base_joint = bearing-float(np.degrees(np.arctan2(forward[1], forward[0])))
    options = {}
    for line in OVERHAND_LINES:
        try:
            options[line['name']] = overhand_release(model.fk_tcp_world, base_joint, base, target, 5., line)
        except ValueError:
            pass
    # The draft keeps the line that lands on the target at the highest release speed (a shortfall loses).
    expected = min(options.items(), key=lambda kv: (kv[1][2] > 0, kv[1][2] if kv[1][2] > 0 else -kv[1][3]))[0]
    assert result['throw_line'] == expected
    line = line_named(expected)
    t_r, scale, shortfall, speed_at_release = options[expected]
    assert result['release_time_s'] == pytest.approx(t_r) and result['rate_scale'] == pytest.approx(scale)
    assert result['landing_shortfall_m'] == pytest.approx(shortfall) == 0.
    assert result['toss_speed_m_s'] == pytest.approx(speed_at_release, abs=.02)
    poses, joints, times, release = overhand_samples(model.fk_tcp_world, base_joint, t_r, line)
    # The stroke stops OVERHAND_FOLLOW_AFTER_RELEASE_S after the release instant, not at the end of the line.
    end = min(line['follow_s'], t_r+OVERHAND_FOLLOW_AFTER_RELEASE_S)
    grid = np.arange(-line['windup_s'], end+1e-9, OVERHAND_STEP_S)
    assert set(np.round(grid, 6)) <= set(np.round(times, 6)) and times[release] == pytest.approx(t_r)
    assert times[-1] == pytest.approx(end, abs=OVERHAND_STEP_S) and times[-1] <= end+1e-9
    np.testing.assert_allclose(joints[release, 1:], line['release_joints_deg'][1:]+line['rates_deg_s'][1:]*t_r)
    assert result['release_joints_deg'] == pytest.approx(joints[release].tolist())
    tips = poses[:, :3, 3]
    velocities = np.gradient(tips, times, axis=0)*scale
    landing = np.linalg.norm(tips[release, :2]-base)+flight_distance(tips[release], velocities[release], target[2])
    assert landing == pytest.approx(.60, abs=.03)
    assert result['throw_distance_m'] == pytest.approx(flight_distance(tips[release], velocities[release], target[2]))
    assert 0 <= result['tool_off_deg'] <= OVERHAND_TOOL_CONE_DEG+1e-6
    points = np.asarray(result['arm']['path']['points'])
    np.testing.assert_allclose(points[1], result['source_xyz'])
    np.testing.assert_allclose(points[2], points[0])
    # The carry is a joint-space line from the hover configuration to the wind-up, sampled every 2.5 degrees.
    hover_joints = np.asarray(result['hover_joints_deg'])
    assert np.linalg.norm(model.fk_tcp_world(hover_joints)[:3, 3]-points[0]) < .003
    delta = joints[0]-hover_joints
    k = result['carry_samples']
    assert k == max(1, int(np.ceil(np.abs(delta).max()/OVERHAND_CARRY_STEP_DEG)))-1
    assert points.shape == (3+k+len(times), 3)
    carry_joints = hover_joints+delta*np.linspace(0, 1, k+2)[1:-1, None]
    np.testing.assert_allclose(points[3:3+k], [model.fk_tcp_world(q)[:3, 3] for q in carry_joints], atol=1e-9)
    assert points[3:3+k, 2].min() >= points[1, 2]+.03
    np.testing.assert_allclose(points[3+k:], tips)
    np.testing.assert_allclose(result['windup_xyz'], tips[0])
    np.testing.assert_allclose(result['target_xyz'], tips[release])
    n = len(points)-1
    keys = result['arm']['orientation']['points']
    approach = 2 if keys[1][0] == pytest.approx(2/n) else 3
    assert [key[0] for key in keys[approach:]] == pytest.approx([(3+i)/n for i in range(k+len(times))])
    for key, pose in zip(keys[approach:], [*[model.fk_tcp_world(q) for q in carry_joints], *poses]):
        np.testing.assert_allclose(Rotation.from_euler('xyz', key[1:], degrees=True).as_matrix(), pose[:3, :3], atol=1e-9)
    speed = dict((round(s_, 9), v) for s_, v in result['arm']['speed'])
    tip_speed = np.linalg.norm(velocities, axis=1)
    assert speed[round((3+k+release)/n, 9)] == pytest.approx(tip_speed[release])
    assert speed[round(3/n, 9)] == OVERHAND_CARRY_SPEED_M_S and speed[round((2+k)/n, 9)] == OVERHAND_CARRY_SPEED_M_S
    assert speed[round(1., 9)] == pytest.approx(tip_speed[-1])
    events = result['arm']['gripper_events']
    assert [e['s'] for e in events] == pytest.approx([0, 1/n, (3+k)/n, (3+k+release)/n])
    assert events[3] == {'s': (3+k+release)/n, 'opening_mm': 70, 'lead_s': .06, 'on_measured': True, 'ramp_s': 0.}
    assert events[2] == {'s': (3+k)/n, 'hold_s': .2, 'wait_for_arrival': True}
    # toss_speed_m_s caps the peak fingertip speed of the stroke by scaling the joint rates down
    capped = toss_between(object_frame(), [396, 226], target, table_z=table_z, base_xy=base, style='overhand',
                          toss_speed_m_s=.5, **kinematics)
    assert OVERHAND_RATE_SCALE[0] <= capped['rate_scale'] < result['rate_scale']
