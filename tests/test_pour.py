"""Side grasp and pour from one stroke drawn from a cup to a target container."""
import copy
import numpy as np
import pytest
from conftest import make_client
from scipy.spatial.transform import Rotation
from urai.backend import SimulationBackend
from urai.pour import pour_height
from urai.trajectory import compile_arm

CUP_PIXEL = [396, 226]
TARGET_PIXEL = [560, 290]
STROKE = [CUP_PIXEL, [480, 260], TARGET_PIXEL]
BASE_XY = (.25, .12)


def pour_frame(cup_depth=.90):
    """Flat table at 1 m depth, a 10 cm tall cup block and a 5 cm tall target block."""
    frame = SimulationBackend().capture()
    frame.depth[:] = 1.
    frame.depth[210:242, 380:412] = cup_depth
    frame.depth[270:310, 540:580] = .95
    return frame


def rotation_of(keyframe):
    return Rotation.from_euler('xyz', keyframe[1:], degrees=True).as_matrix()


def arc_length(tilt_deg):
    """Waypoints of the pouring arc: one every 5 degrees from upright to the full tilt."""
    return int(np.ceil(tilt_deg/5.))+1


def test_side_grasp_pose_is_horizontal_radial_with_the_tool_y_down():
    from urai.pour import POUR_OFFSETS_DEG, pour_from_stroke
    result = pour_from_stroke(pour_frame(), STROKE, table_z=0., base_xy=BASE_XY)
    assert result['route_mode'] == 'pour'
    approach = np.asarray(result['approach_direction_xy'])
    cup_xy = np.asarray(result['source_xyz'][:2])
    radial = (cup_xy - BASE_XY) / np.linalg.norm(cup_xy - BASE_XY)
    # Without kinematics the first candidates stand: the radial approach and no measured reach errors.
    np.testing.assert_allclose(approach, radial, atol=1e-9)
    assert result['approach_offset_deg'] == 0
    assert result['reach_error_mm'] is None and result['reach_error_deg'] is None
    upright = rotation_of(result['arm']['orientation']['points'][0])
    np.testing.assert_allclose(upright @ upright.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(upright) == pytest.approx(1)
    np.testing.assert_allclose(upright[:, 2], [*approach, 0.], atol=1e-9)
    assert upright[2, 0] == pytest.approx(0, abs=1e-9)
    assert upright[:2, 0] @ approach == pytest.approx(0, abs=1e-9)
    # The wrist's natural roll for a horizontal tool points the tool Y down; Y up is 60 deg past the j6 limit.
    np.testing.assert_allclose(upright[:, 1], [0, 0, -1], atol=1e-9)
    # The mouth tips along the offered direction closest to the drawn cup-to-target line.
    target_xy = np.asarray(result['target_center_xy'])
    drawn = (target_xy - cup_xy) / np.linalg.norm(target_xy - cup_xy)
    target_radial = np.rad2deg(np.arctan2(*(target_xy - BASE_XY)[::-1]))
    offered = {offset: np.array([np.cos(np.deg2rad(target_radial + offset)), np.sin(np.deg2rad(target_radial + offset))])
               for offset in POUR_OFFSETS_DEG}
    expected = max(offered, key=lambda offset: offered[offset] @ drawn)
    assert result['pour_offset_deg'] == expected
    np.testing.assert_allclose(result['pour_direction_xy'], offered[expected], atol=1e-9)
    assert 'roll_deg' not in result


@pytest.mark.parametrize('base_xy', [BASE_XY, (.6, -.5)])
def test_the_pour_tips_the_mouth_toward_the_pour_direction_in_five_degree_steps(base_xy):
    """Each arc keyframe is the upright grasp turned about one horizontal axis, 5 degrees further than the last."""
    from urai.pour import pour_from_stroke
    tilt = 100.
    result = pour_from_stroke(pour_frame(), STROKE, table_z=0., base_xy=base_xy, tilt_deg=tilt)
    keyframes = result['arm']['orientation']['points']
    points = result['arm']['path']['points']
    count = arc_length(tilt)
    assert len(keyframes) == len(points) == 3+2*count-1+3          # grasp and lift, the arc out and back, release
    n = len(points)-1
    assert [k[0] for k in keyframes] == pytest.approx([i/n for i in range(len(points))])
    upright = rotation_of(keyframes[0])
    for k in (1, 2, 3, n-2, n-1, n):
        np.testing.assert_allclose(rotation_of(keyframes[k]), upright, atol=1e-9)
    pour_direction = np.asarray(result['pour_direction_xy'])
    hold = result['pour_hold_index']
    assert hold == 3+count-1
    for step, k in enumerate(range(3, 3+count)):
        rotvec = Rotation.from_matrix(rotation_of(keyframes[k]) @ upright.T).as_rotvec(degrees=True)
        assert np.linalg.norm(rotvec) == pytest.approx(min(5.*step, tilt), abs=1e-6)
        if step:
            axis = rotvec/np.linalg.norm(rotvec)
            assert axis[2] == pytest.approx(0, abs=1e-9)
            assert axis @ [*pour_direction, 0.] == pytest.approx(0, abs=1e-9)
        # The way back is the same arc in reverse.
        np.testing.assert_allclose(rotation_of(keyframes[2*hold-k]), rotation_of(keyframes[k]), atol=1e-9)
    tilted = rotation_of(keyframes[hold])
    cup_axis = -tilted[:, 1]  # the cup stands along the tool -Y
    np.testing.assert_allclose(cup_axis[:2]/np.linalg.norm(cup_axis[:2]), pour_direction, atol=1e-9)
    assert cup_axis[2] == pytest.approx(np.cos(np.deg2rad(tilt)))
    approach = np.asarray(result['approach_direction_xy'])
    assert approach @ (np.asarray(result['source_xyz'][:2]) - base_xy) > 0


def test_waypoints_and_events_follow_the_pour_sequence():
    """Grasp, lift, tip along the arc with the low rim held just inside the target rim, tip back, put down."""
    from urai.pour import pour_from_stroke
    frame = pour_frame()
    tilt = 130.
    result = pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY, hold_s=2., grasp_fraction=.45)
    spec = result['arm']
    points = np.asarray(spec['path']['points'])
    count = arc_length(tilt)
    hold = result['pour_hold_index']
    assert hold == 3+count-1 and len(points) == 3+2*count-1+3
    n = len(points)-1
    cup_height = result['cup_height_mm'] / 1000
    assert cup_height == pytest.approx(.10, abs=.005)
    radius = result['cup_diameter_mm'] / 2000
    approach = np.asarray(result['approach_direction_xy'])
    pour_direction = np.asarray(result['pour_direction_xy'])
    grasp = points[1]
    np.testing.assert_allclose(grasp, result['source_xyz'])
    assert grasp[2] == pytest.approx(cup_height * .45)
    np.testing.assert_allclose(points[0], grasp - [*(approach * (radius + .06)), 0.])
    assert (points[0][:2] - grasp[:2]) @ approach < 0
    # The arc out, held at its end, and the same arc back.
    arc = points[3:3+count]
    np.testing.assert_allclose(points[hold+1:hold+count][::-1], arc[:-1])
    np.testing.assert_allclose(result['target_xyz'], points[hold])
    # Throughout the tilt the low side of the rim stays at the lip point just inside the target's near rim.
    above = cup_height*(1-.45)
    angles = np.deg2rad(np.linspace(0., tilt, count))
    lip = arc[:, :2]+(above*np.sin(angles)+radius*np.cos(angles))[:, None]*pour_direction
    np.testing.assert_allclose(lip, np.tile(result['pour_lip_xy'], (count, 1)), atol=1e-9)
    target_top = frame.pick(*TARGET_PIXEL, mode='surface')[2]
    assert points[hold][2] == pytest.approx(pour_height(target_top, cup_height, .45, radius, tilt, .10))
    lift_z = max(grasp[2]+.08, arc[:, 2].max())
    np.testing.assert_allclose(points[2], [*grasp[:2], lift_z])
    np.testing.assert_allclose(points[-3], points[2])
    np.testing.assert_allclose(points[-2], grasp)
    np.testing.assert_allclose(points[-1], points[0])
    events = spec['gripper_events']
    assert [e['s'] for e in events] == pytest.approx([0, 1/n, hold/n, (n-1)/n])
    assert [e.get('opening_mm') for e in events] == [70, 0, None, 70]
    assert events[1]['effort'] == 300 and all('effort' not in e for e in events[::2] + events[2:3])
    assert [e.get('verify') for e in events] == [None, 'holding', None, None]
    assert events[2] == {'s': hold/n, 'hold_s': 2., 'wait_for_arrival': True}
    assert all(e['wait_for_arrival'] for e in events)
    assert spec['approach'] == {'speed_m_s': .8, 'clearance_m': .12}
    assert spec['speed'] == .8


def test_pour_draft_compiles_with_slow_rotation_and_a_stationary_pause():
    from urai.pour import pour_from_stroke
    result = pour_from_stroke(pour_frame(), STROKE, table_z=0., base_xy=BASE_XY)
    spec = result['arm']
    compiled = compile_arm(spec, SimulationBackend().state()['left'])
    rotations = Rotation.from_matrix(compiled['rotation_matrices'])
    steps = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude())
    assert steps.max() <= 2.5 + 1e-6
    pause = compiled['gripper_events'][2]
    assert 'opening_mm' not in pause
    assert pause['hold_end_time_s'] - pause['time_s'] >= 2.5 - 1e-9
    index = int(np.where(compiled['time_s'] == pause['time_s'])[0][0])
    np.testing.assert_allclose(compiled['xyz'][index], spec['path']['points'][result['pour_hold_index']])
    np.testing.assert_allclose(compiled['xyz'][index + 1], compiled['xyz'][index])
    tilt = np.rad2deg((rotations[0].inv() * rotations[index]).magnitude())
    assert tilt == pytest.approx(130, abs=1e-6)
    for event in compiled['gripper_events']:
        i = int(np.where(compiled['time_s'] == event['time_s'])[0][0])
        np.testing.assert_allclose(compiled['xyz'][i + 1], compiled['xyz'][i])


def test_pour_rejects_table_endpoints_short_cups_and_bad_parameters_but_not_wide_cups():
    from urai.pour import pour_from_stroke
    frame = pour_frame()
    with pytest.raises(ValueError, match='终点'):
        pour_from_stroke(frame, [CUP_PIXEL, [650, 400]], table_z=0., base_xy=BASE_XY)
    with pytest.raises(ValueError, match='同一个物体'):
        pour_from_stroke(frame, [[386, 226], [406, 226]], table_z=0., base_xy=BASE_XY)
    wide = SimulationBackend().capture()
    wide.depth[270:310, 540:580] = .95
    # The estimated diameter does not reject: a handle and depth noise widen a real cup well past its rim.
    result = pour_from_stroke(wide, [[395, 220], TARGET_PIXEL], table_z=0., base_xy=BASE_XY)
    assert result['cup_diameter_mm'] > 60
    with pytest.raises(ValueError, match='50 mm'):
        pour_from_stroke(pour_frame(cup_depth=.96), STROKE, table_z=0., base_xy=BASE_XY)
    with pytest.raises(ValueError, match='60–170'):
        pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY, tilt_deg=30.)
    with pytest.raises(ValueError, match='停留'):
        pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY, hold_s=5.)
    with pytest.raises(ValueError, match='比例'):
        pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY, grasp_fraction=.9)
    with pytest.raises(ValueError, match='物体'):
        pour_from_stroke(frame, [[650, 400], TARGET_PIXEL], table_z=0., base_xy=BASE_XY)


def test_pour_api_uses_the_selected_arm_base_and_does_not_mutate_the_service(sim_service):
    """The simulation backend has no arm model, so the arm's current tool position stands in for its base."""
    service = sim_service
    service.observe()
    client = make_client(service)
    before = copy.deepcopy(service.draft)
    # The observation is a synthetic depth image of a cup and a target block.
    frame = pour_frame()
    service.frame = frame
    for arm in ('left', 'right'):
        response = client.post('/api/pour', json={'observation_id': frame.id, 'arm': arm, 'pixels': STROKE,
                                                  'tilt_deg': 90, 'hold_s': 1.5, 'grasp_fraction': .5,
                                                  'mouth_clearance_m': .08, 'grip_effort': 400})
        assert response.status_code == 200, response.text
        result = response.json()
        base_xy = np.asarray(service.backend.state()[arm]['xyz'][:2])
        cup_xy = np.asarray(result['source_xyz'][:2])
        assert np.asarray(result['approach_direction_xy']) @ (cup_xy - base_xy) > 0
        assert result['tilt_deg'] == 90 and result['arm']['gripper_events'][2]['hold_s'] == 1.5
        assert result['mouth_clearance_m'] == .08 and result['grip_effort'] == 400
        assert result['arm']['gripper_events'][1]['effort'] == 400
        assert result['source_xyz'][2] == pytest.approx(result['cup_height_mm'] / 1000 * .5)
        # The tool puts the cup back and retreats; the service returns home in joint space.
        points = result['arm']['path']['points']
        np.testing.assert_allclose(points[-2], result['source_xyz'], atol=1e-9)
        np.testing.assert_allclose(points[-1], points[0], atol=1e-9)
        assert result['reach_error_mm'] is None  # the simulation has no kinematics to rank candidates with
    assert service.draft == before and service.execution['state'] == 'idle'
    # The generated draft goes through the ordinary draft -> preview chain of the simulation backend.
    saved = client.put('/api/draft', json={'observation_id': frame.id, 'arms': {'right': result['arm']},
                                           'expected_revision': service.revision})
    assert saved.status_code == 200, saved.text
    plan = client.post('/api/preview', json={'expected_revision': saved.json()['revision']})
    assert plan.status_code == 200, plan.text
    planned = plan.json()['arms']['right']
    assert plan.json()['duration_s'] > 1.5 + .8 + 1. + .8
    assert 'opening_mm' not in planned['gripper_events'][2]
    assert planned['gripper_events'][1]['effort'] == 400            # the torque limit survives compilation
    assert planned['gripper_events'][2]['hold_end_time_s'] - planned['gripper_events'][2]['time_s'] >= 1.5 - 1e-9
    assert client.post('/api/pour', json={'observation_id': 'old', 'pixels': STROKE}).status_code == 409
    assert client.post('/api/pour', json={'observation_id': frame.id, 'arm': 'bad', 'pixels': STROKE}).status_code == 409


def hollow_target_frame():
    """The pour target is a 5 cm tall, 120 x 120 px square ring (a tin): its floor is at table height, only the rim is
    raised, and the stroke end sits 38 px from the rim, beyond locate_object's snap radius."""
    frame = pour_frame()
    frame.depth[270:310, 540:580] = 1.
    frame.depth[240:360, 500:620] = .95
    frame.depth[252:348, 512:608] = 1.
    return frame


def test_pour_target_drawn_inside_a_hollow_container_resolves_to_its_rim():
    from urai.pour import pour_from_stroke
    from urai.transfer import locate_container, locate_object
    frame = hollow_target_frame()
    with pytest.raises(ValueError, match='没有可分离的物体'):
        locate_object(frame, TARGET_PIXEL, 0.)
    rim = locate_container(frame, TARGET_PIXEL, 0.)
    ring_centre = frame.pick(560, 300, mode='plane', z=.05)[:2]     # the ring's rim is centred on pixel (560, 300)
    np.testing.assert_allclose(rim['center'], ring_centre, atol=.01)
    assert rim['top'] == pytest.approx(.05, abs=.003)
    result = pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY)
    np.testing.assert_allclose(result['target_center_xy'], ring_centre, atol=.01)
    assert result['target_height_mm'] == pytest.approx(50, abs=3)
    # a solid target behaves as before, and empty table without a rim around it is still refused
    solid = locate_object(pour_frame(), TARGET_PIXEL, 0.)
    np.testing.assert_allclose(locate_container(pour_frame(), TARGET_PIXEL, 0.)['center'], solid['center'], atol=.003)
    with pytest.raises(ValueError, match='容器'):
        pour_from_stroke(frame, [CUP_PIXEL, [650, 420]], table_z=0., base_xy=BASE_XY)


TIN_CENTRE = np.array([.63, -.60])
TIN_RADIUS = .07
TIN_WALL_M = .004
TIN_HEIGHT = .05
HEAD_CAMERA_ORIGIN = np.array([.007, -.232, .806])   # a head camera about 0.8 m above the table


def looking_camera(origin, at):
    """Camera pose whose optical axis runs from ``origin`` to ``at``; image right is world -y, image up world +x."""
    z = at-origin
    z = z/np.linalg.norm(z)
    x = np.cross([0., 0., -1.], z)
    x = x/np.linalg.norm(x)
    t = np.eye(4)
    t[:3, :3] = np.c_[x, np.cross(z, x), z]
    t[:3, 3] = origin
    return t


def render_tin(frame, centre, radius, wall, height):
    """Ray-cast depth of a white table carrying one open tin: each pixel keeps the first of the near outer wall,
    the rim, the far inner wall, the floor and the table that its ray meets."""
    h, w = frame.depth.shape
    v, u = np.indices((h, w))
    d = np.stack([(u-frame.k[0, 2])/frame.k[0, 0], (v-frame.k[1, 2])/frame.k[1, 1], np.ones((h, w))], -1) @ frame.t[:3, :3].T
    o = frame.t[:3, 3]
    depth = np.full((h, w), np.inf)

    def consider(s, valid):
        ok = valid & np.isfinite(s) & (s > 0) & (s < depth)
        depth[ok] = s[ok]

    def hit(s):
        p = o+d*s[..., None]
        return np.linalg.norm(p[..., :2]-centre, axis=-1), p[..., 2]

    for z, inner, outer in ((0., 0., radius-wall), (0., radius, np.inf), (height, radius-wall, radius)):   # floor, table, rim
        s = (z-o[2])/d[..., 2]
        r, _ = hit(s)
        consider(s, (r >= inner) & (r <= outer))
    oc = o[:2]-centre
    a = (d[..., :2]**2).sum(-1)
    b = 2*(d[..., :2] @ oc)
    for rho, far in ((radius, False), (radius-wall, True)):    # near outer wall, far inner wall
        disc = b*b-4*a*(oc @ oc-rho*rho)
        root = np.sqrt(np.where(disc >= 0, disc, np.nan))
        s = (-b+root)/(2*a) if far else (-b-root)/(2*a)
        _, z = hit(s)
        consider(s, (disc >= 0) & (z >= 0) & (z <= height))
    frame.depth[:] = np.where(np.isfinite(depth), depth, np.nan)
    frame.rgb[:] = 235
    return frame


def partial_tin_frame():
    """The head camera tilted so the tin's far rim leaves the top of the picture: only its near half is seen,
    with the floor showing above the near rim."""
    frame = SimulationBackend().capture()
    frame.t = looking_camera(HEAD_CAMERA_ORIGIN, np.array([.30, -.40, 0.]))
    return render_tin(frame, TIN_CENTRE, TIN_RADIUS, TIN_WALL_M, TIN_HEIGHT)


def test_pour_target_drawn_on_the_floor_of_a_tin_cut_by_the_picture_edge_resolves_to_its_rim_circle():
    from urai.transfer import leveled_cloud, locate_container, rim_circle
    frame = partial_tin_frame()
    toward = (TIN_CENTRE-HEAD_CAMERA_ORIGIN[:2])/np.linalg.norm(TIN_CENTRE-HEAD_CAMERA_ORIGIN[:2])
    far_rim = frame.project(np.r_[TIN_CENTRE+TIN_RADIUS*toward, TIN_HEIGHT])[0]
    assert far_rim[1] < 0                                   # the far rim is outside the picture
    floor = frame.project(np.r_[TIN_CENTRE+.02*toward, 0.])[0]
    u, v = np.rint(floor).astype(int)
    assert 0 <= v < 40 and 0 <= u < frame.depth.shape[1]
    cloud, _ = leveled_cloud(frame, 0.)
    assert abs(cloud[v, u, 2]) < .002                       # what is drawn there is the tin's floor, at table height
    rim = locate_container(frame, [u, v], 0., cloud=cloud)
    np.testing.assert_allclose(rim['center'], TIN_CENTRE, atol=.01)
    assert rim['radius'] == pytest.approx(TIN_RADIUS, abs=.005)
    assert rim['top'] == pytest.approx(TIN_HEIGHT, abs=.003)
    centre, radius = rim_circle(rim['points'], frame.t[:2, 3])
    np.testing.assert_allclose(centre, rim['center'])
    # the lower part of the near wall is below the depth-rise rule; the rim circle still claims it
    wall = frame.project(np.r_[TIN_CENTRE-TIN_RADIUS*toward, .008])[0]
    low = locate_container(frame, wall, 0., cloud=cloud)
    np.testing.assert_allclose(low['center'], rim['center'], atol=1e-6)
    # bare table 4 cm outside the wall is refused; the visible half's centroid would have been 2 cm too near
    outside = frame.project(np.r_[TIN_CENTRE-(TIN_RADIUS+.04)*toward, 0.])[0]
    with pytest.raises(ValueError, match='容器'):
        locate_container(frame, outside, 0., cloud=cloud)
    assert np.linalg.norm(rim['points'][:, :2].mean(axis=0)-TIN_CENTRE) > .02


def test_pour_height_and_grip_effort_follow_the_mouth_clearance_and_effort_parameters():
    from urai.pour import CUP_GRIP_EFFORT, POUR_MOUTH_CLEARANCE_M, pour_from_stroke
    frame = pour_frame()
    base = pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY)
    higher = pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY, mouth_clearance_m=.14, grip_effort=500)
    assert base['mouth_clearance_m'] == POUR_MOUTH_CLEARANCE_M and base['grip_effort'] == CUP_GRIP_EFFORT
    hold = base['pour_hold_index']
    z0, z1 = base['arm']['path']['points'][hold][2], higher['arm']['path']['points'][hold][2]
    assert z1-z0 == pytest.approx(.04)
    assert base['pour_height_mm'] == pytest.approx(z0*1000)
    assert z0 == pytest.approx(pour_height(.05, base['cup_height_mm']/1000, .30, base['cup_diameter_mm']/2000, 130., .10))
    # past 90 degrees the mouth centre drops below the grasp point and the rim's low side hangs a radius further
    assert pour_height(.05, .10, .45, .04, 100., .06) == pytest.approx(.05+.06+.055*np.sin(np.deg2rad(10))+.04*np.cos(np.deg2rad(10)))
    assert base['arm']['path']['points'][2][2] >= z0                 # the lift clears the pour height
    assert higher['arm']['gripper_events'][1]['effort'] == 500
    for bad in ({'mouth_clearance_m': .5}, {'grip_effort': 10}, {'grip_effort': 300.5}):
        with pytest.raises(ValueError):
            pour_from_stroke(frame, STROKE, table_z=0., base_xy=BASE_XY, **bad)


def test_side_grasp_pitch_turns_the_tool_nose_down_about_the_closing_axis():
    from urai.pour import pour_from_stroke, side_grasp_rotation
    level, down = side_grasp_rotation([1., 0.]), side_grasp_rotation([1., 0.], 15.)
    np.testing.assert_allclose(down[:, 0], level[:, 0], atol=1e-12)                 # the closing axis stays horizontal
    assert down[2, 2] == pytest.approx(-np.sin(np.deg2rad(15.)))                    # the tool axis points 15 deg down
    assert level[2, 2] == pytest.approx(0., abs=1e-12)
    with pytest.raises(ValueError, match='倾倒角'):
        pour_from_stroke(pour_frame(), STROKE, table_z=0., base_xy=BASE_XY, tilt_deg=175.)
