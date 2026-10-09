"""Sidearm toss drafts, the acceleration envelope and early release dispatch."""
import numpy as np
import pytest
from conftest import make_client
from scipy.spatial.transform import Rotation
from urai.backend import SimulationBackend
from urai.settings import motion_limits
from urai.toss import (ARC_SAMPLES, DEFAULT_LEAD_S, DEFAULT_TOSS_SPEED_M_S, RELEASE_INDEX, SWING_ANGULAR_SPEED_DEG_S,
                       SWING_MAX_BEARING_DEG, SWING_RADII_M, SWING_TILT_DEG, swing_geometry,
                       toss_between as any_toss, width_along)
from urai.trajectory import CONTROL_PERIOD_S, compile_arm, control_period_mask, gripper_event, time_samples
from test_transfer import object_frame

import functools

toss_between = functools.partial(any_toss, style='sidearm')   # these tests cover the sidearm swing

BASE_XY = [0., 0.]
TARGET = [.60, -.45, 0.]


def jaw_axis_xy(rpy_deg):
    """Horizontal direction the parallel jaws close along for an RPY orientation."""
    axis = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()[:, 0][:2]
    return axis/np.linalg.norm(axis)


def cross_2d(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def test_swing_geometry_puts_the_target_on_the_release_tangent():
    for target in ([.60, -.45], [.55, .35], [.62, .10]):
        release, sense, radius = swing_geometry(BASE_XY, target, [.30, .10])
        point = radius*np.array([np.cos(release), np.sin(release)])
        tangent = sense*np.array([-np.sin(release), np.cos(release)])
        assert radius == pytest.approx(SWING_RADII_M[0])
        assert abs(point @ (np.array(target)-point)) < 1e-9
        assert (np.array(target)-point) @ tangent > 0
    release, sense, radius = swing_geometry(BASE_XY, [.40, 0.], [.30, .10])
    assert radius == pytest.approx(.35)
    assert swing_geometry(BASE_XY, [.60, -.45], [.30, .10], radii=(.45,))[2] == pytest.approx(.45)
    # Straight ahead at 0.70 m the 0.45 m circle would need a wind-up past the side line; a wider circle fits.
    release, sense, radius = swing_geometry(BASE_XY, [.70, 0.], [.30, .10])
    assert radius >= .45 and abs(np.degrees(release))+30 <= SWING_MAX_BEARING_DEG+1e-9
    # Farther still, the release angle on the 0.45 m circle would put the wind-up past the side line.
    release, sense, radius = swing_geometry(BASE_XY, [.95, 0.], [.30, .10])
    assert radius > .50 and abs(np.degrees(release))+30 <= SWING_MAX_BEARING_DEG+1e-9
    assert (DEFAULT_TOSS_SPEED_M_S, DEFAULT_LEAD_S) == (1.5, .08)
    with pytest.raises(ValueError, match='半径'):
        swing_geometry(BASE_XY, [.30, 0.], [.30, .10])
    # A target ahead and to the left: the clockwise arc would carry the base joint past 90 degrees, so
    # the counter-clockwise swing is taken even though its wind-up is farther from the object.
    release, sense, radius = swing_geometry(BASE_XY, [.52, .31], [.30, .23])
    assert sense == 1. and abs(np.degrees(release)) < SWING_MAX_BEARING_DEG
    with pytest.raises(ValueError, match='正前方'):
        swing_geometry(BASE_XY, [-.30, .50], [.30, .23])


def test_toss_swings_about_the_base_with_the_jaw_opening_facing_the_target():
    frame = object_frame()
    result = toss_between(frame, [396, 226], TARGET, toss_speed_m_s=1.0, lead_s=.12, base_xy=BASE_XY)
    points = np.array(result['arm']['path']['points'])
    n = len(points)-1
    assert len(points) == 4+ARC_SAMPLES
    grasp, windup, release, follow = points[1], points[3], points[3+RELEASE_INDEX], points[-1]
    base = np.array(BASE_XY)
    for point in points[3:]:
        assert np.linalg.norm(point[:2]-base) == pytest.approx(result['swing_radius_m'], abs=1e-9)
    to_target = np.array(TARGET[:2])-release[:2]
    direction = np.array(result['throw_direction_xy'])
    assert abs((release[:2]-base) @ to_target) < 1e-9 and to_target @ direction > 0
    assert windup[2] < release[2] < follow[2]
    np.testing.assert_allclose(result['target_xyz'], release, atol=1e-12)
    events = result['arm']['gripper_events']
    assert events[-1] == {'s': pytest.approx((3+RELEASE_INDEX)/n), 'opening_mm': 70, 'lead_s': .12, 'on_measured': True, 'ramp_s': 0.}
    assert events[1]['ramp_s'] == .5
    assert events[-2] == {'s': pytest.approx(3/n), 'hold_s': .2, 'wait_for_arrival': True}
    assert max(v for _, v in result['arm']['speed']) == 1.0 and result['arm']['accel_m_s2'] == 8.
    assert result['arm']['angular_speed_deg_s'] == SWING_ANGULAR_SPEED_DEG_S
    keys = result['arm']['orientation']['points']
    assert len(keys) == 3+ARC_SAMPLES and keys[-1][0] == 1
    # The jaws close along the radial direction at the grasp and all along the arc: the opening faces the tangent.
    radial = (grasp[:2]-base)/np.linalg.norm(grasp[:2]-base)
    assert abs(cross_2d(jaw_axis_xy(keys[0][1:]), radial)) < 1e-6
    for key, point in zip(keys[2:], points[3:]):
        radial = (point[:2]-base)/np.linalg.norm(point[:2]-base)
        assert abs(cross_2d(jaw_axis_xy(key[1:]), radial)) < 1e-6
    # The grasp leans by reach like every other grasp; along the arc the tool axis leans SWING_TILT_DEG off vertical.
    from urai.transfer import grasp_tilt_deg
    tool_axes = [Rotation.from_euler('xyz', key[1:], degrees=True).as_matrix()[:, 2] for key in keys]
    assert np.degrees(np.arccos(-tool_axes[0][2])) == pytest.approx(grasp_tilt_deg(grasp[:2], base), abs=.5)
    for axis in tool_axes[2:]:
        assert np.degrees(np.arccos(-axis[2])) == pytest.approx(SWING_TILT_DEG, abs=.5)
    assert result['swing_radius_m'] == pytest.approx(.50)
    plan = compile_arm(result['arm'], SimulationBackend().state()['left'])
    release_time = plan['gripper_events'][-1]['time_s']
    i = int(np.searchsorted(plan['time_s'], release_time))
    speed = np.linalg.norm(plan['xyz'][i]-plan['xyz'][i-1])/(plan['time_s'][i]-plan['time_s'][i-1])
    assert speed > .7




def bar_frame(along_x):
    """A 40 mm tall bar 100 x 30 mm around world (0.40, 0) under the nadir camera, its long side along world X or Y.

    Under this camera world X runs down the image rows and world Y across the columns, 0.98 m / 600 px per pixel.
    """
    frame = object_frame()
    frame.depth[:] = 1.
    rows, columns = (61, 18) if along_x else (18, 61)
    v, u = 240, 522
    frame.depth[v-rows//2:v+rows//2, u-columns//2:u+columns//2] = .96
    return frame, [u, v]


def test_toss_reports_width_across_the_radial_jaw_axis_without_refusing():
    # The base sits at the origin, so the radial jaw axis at the bar is +X: a bar along +X measures wide across
    # it. That is reported, not refused - the measurement is not reliable enough to be a gate.
    frame, pixel = bar_frame(along_x=True)
    wide = toss_between(frame, pixel, [.75, .10, 0.], base_xy=BASE_XY)
    pixel_m = .96/600                                     # one pixel at the bar's top
    assert wide['arm']['path']['points'] and wide['object_width_mm'] == pytest.approx((60*pixel_m+.003)*1000, abs=3)
    frame, pixel = bar_frame(along_x=False)
    narrow = toss_between(frame, pixel, [.75, .10, 0.], base_xy=BASE_XY)
    assert narrow['object_width_mm'] == pytest.approx((18*pixel_m+.003)*1000, abs=3)
    assert width_along(np.array([[0., 0.], [.1, 0.], [.1, .02]]), 0.) == pytest.approx(.103, abs=.002)


def test_toss_requires_the_arm_base():
    with pytest.raises(ValueError, match='基座'):
        toss_between(object_frame(), [396, 226], TARGET)


def test_toss_rejects_targets_too_close():
    from urai.transfer import objects_in_polygon
    frame = object_frame()
    [block] = objects_in_polygon(frame, [[300, 150], [500, 150], [500, 300], [300, 300]], table_z=0)
    near = (np.array(block['center_xy'])+[.05, .05]).tolist()+[0.]
    with pytest.raises(ValueError, match='太近|离物体'):
        toss_between(frame, [396, 226], near, base_xy=BASE_XY)


def test_acceleration_envelope_shortens_the_ramps():
    xyz = np.c_[np.linspace(0, .5, 101), np.zeros(101), np.zeros(101)]
    rotations = Rotation.from_euler('xyz', np.zeros((101, 3)))
    slow, *_ = time_samples(xyz, rotations, np.full(101, 1.0))
    fast, *_ = time_samples(xyz, rotations, np.full(101, 1.0), accel_m_s2=5.)
    assert fast[-1] < slow[-1]/2
    spec = {'path': {'mode': 'waypoints', 'points': [[.25, .12, .24], [.75, .12, .24]]}, 'speed': 1.0, 'accel_m_s2': 5.}
    assert compile_arm(spec, SimulationBackend().state()['left'])['accel_m_s2'] == 5.
    with pytest.raises(ValueError, match='accel'):
        compile_arm({**spec, 'accel_m_s2': 500.}, SimulationBackend().state()['left'])


def test_release_event_fields_are_validated():
    assert gripper_event({'s': .5, 'opening_mm': 70, 'lead_s': .1})['lead_s'] == .1
    with pytest.raises(ValueError, match='lead_s'):
        gripper_event({'s': .5, 'opening_mm': 70, 'hold_s': .5, 'lead_s': .1})
    with pytest.raises(ValueError, match='lead_s'):
        gripper_event({'s': .5, 'hold_s': .5, 'lead_s': .1})
    assert gripper_event({'s': .5, 'opening_mm': 0, 'hold_s': 1., 'ramp_s': .5})['ramp_s'] == .5
    with pytest.raises(ValueError, match='ramp_s'):
        gripper_event({'s': .5, 'hold_s': .5, 'ramp_s': .5})
    with pytest.raises(ValueError, match='ramp_s'):
        gripper_event({'s': .5, 'opening_mm': 0, 'ramp_s': 3.})
    assert gripper_event({'s': .5, 'opening_mm': 0, 'effort': 300})['effort'] == 300
    with pytest.raises(ValueError, match='effort'):
        gripper_event({'s': .5, 'hold_s': .5, 'effort': 300})
    with pytest.raises(ValueError, match='effort'):
        gripper_event({'s': .5, 'opening_mm': 0, 'effort': 6000})
    assert gripper_event({'s': .5, 'opening_mm': 70, 'on_measured': True})['on_measured'] is True
    with pytest.raises(ValueError, match='on_measured'):
        gripper_event({'s': .5, 'opening_mm': 70, 'hold_s': .5, 'on_measured': True})


def test_a_lead_time_dispatches_the_release_early_on_the_wall_clock():
    """The simulation backend runs a 0.6 s plan in real time; an opening due at 0.5 s with a 0.1 s lead goes out
    at about 0.4 s."""
    from urai.service import DispatchCancellation
    backend = SimulationBackend()
    plan = {'duration_s': .6, 'arms': {'left': {'time_s': np.array([0., .6]), 'xyz': np.tile([.25, .12, .24], (2, 1)),
            'rotation_matrices': np.tile(np.eye(3), (2, 1, 1)),
            'gripper_events': [{'time_s': .5, 'opening_mm': 70, 'lead_s': .1}]}}}
    log = []
    result = backend.execute(plan, backend.state(), DispatchCancellation(), lambda fraction, t: log.append((t, backend.openings['left'])))
    assert result['completed']
    first_open = min(t for t, opening in log if opening == 70)
    assert .39 <= first_open <= .45
    assert max(t for t, opening in log if opening == 0) < .4


def wait_for_queue(client, timeout_s=120.):
    import time
    deadline = time.monotonic()+timeout_s
    while time.monotonic() < deadline and client.get('/api/tasks').json()['active']:
        time.sleep(.2)
    return client.get('/api/tasks').json()['items'][0]


def test_the_toss_queue_needs_the_throw_profile_and_tosses_from_the_task_queue(sim_service):
    assert motion_limits('throw')['joint_speed_deg_s'] == 150.
    with pytest.raises(ValueError, match='运动档位'):
        motion_limits('unlimited')
    service = sim_service
    service.observe()
    client = make_client(service)
    item = {'object_pixel': [395, 220], 'placement': 'toss'}
    assert client.put('/api/drop-point', json={'xyz': [.60, -.45, 0.]}).status_code == 200
    refused = client.post('/api/tasks', json={'observation_id': service.frame.id, 'toss_style': 'sidearm', 'items': [item]})
    assert refused.status_code == 409 and 'throw 运动档' in refused.json()['detail']
    assert client.put('/api/settings', json={'motion_profile': 'throw'}).status_code == 200
    bad = client.post('/api/tasks', json={'observation_id': service.frame.id, 'toss_speed_m_s': 9., 'items': [item]})
    assert bad.status_code == 409
    assert client.post('/api/tasks', json={'observation_id': service.frame.id, 'toss_style': 'lob', 'items': [item]}).status_code == 409
    response = client.post('/api/tasks', json={'observation_id': service.frame.id, 'toss_style': 'sidearm', 'items': [item]})
    assert response.status_code == 200, response.text
    assert response.json()['items'][0]['placement'] == 'toss'
    done = wait_for_queue(client)
    assert done['status'] == 'completed', done
    assert done['release_speed_m_s'] > .7
    assert done['toss_style'] == 'sidearm' and 0 < done['swing_radius_m'] <= SWING_RADII_M[0]
    # The default overhand throw needs the arm's kinematics, which the simulation backend does not have.
    response = client.post('/api/tasks', json={'observation_id': service.frame.id, 'items': [item]})
    assert response.status_code == 200, response.text
    done = wait_for_queue(client)
    assert done['status'] == 'failed' and '运动学' in done['error'], done


def test_fast_references_are_never_denser_than_the_control_period():
    state = SimulationBackend().state()['left']
    fast = compile_arm({'path': {'mode': 'waypoints', 'points': [[.25, .12, .24], [.45, .12, .24], [.45, .32, .24]]},
                        'speed': 1.0, 'accel_m_s2': 8., 'gripper_events': [{'s': .5, 'opening_mm': 0, 'hold_s': .5, 'wait_for_arrival': True}]}, state)
    steps = np.diff(fast['time_s'])
    assert steps[steps > 0].min() >= CONTROL_PERIOD_S-1e-9
    assert .5 in fast['s'] and len(fast['s']) < 80
    slow = compile_arm({'path': {'mode': 'waypoints', 'points': [[.25, .12, .24], [.45, .12, .24]]}, 'speed': .03}, state)
    assert len(slow['s']) == 201
    s = np.linspace(0, 1, 11)
    times = s*.05
    keep = control_period_mask(s, times, [s[3]])
    assert keep.tolist() == [True, False, False, True, False, False, False, True, False, False, True]
    assert keep[3] and times[3]-times[0] < CONTROL_PERIOD_S  # pinned poses are kept even when crowded


def test_ballistic_speed_and_flight_distance_agree():
    from urai.toss import ballistic_speed, flight_distance
    v = ballistic_speed(.30, .50, 10.)
    e = np.deg2rad(10.)
    t = .30/(v*np.cos(e))
    assert .50+v*np.sin(e)*t-.5*9.81*t**2 == pytest.approx(0., abs=1e-9)
    assert flight_distance([0, 0, .50], [v*np.cos(e), 0, v*np.sin(e)], 0.) == pytest.approx(.30)
    assert flight_distance([0, 0, .50], [1., 0, 0], 0.) == pytest.approx(np.sqrt(2*.5/9.81))
    assert ballistic_speed(.20, .68, 10.) < ballistic_speed(.45, .68, 10.)
    with pytest.raises(ValueError, match='放低或放远'):
        ballistic_speed(.10, -.5, 10.)


def test_overhand_toss_needs_kinematics_and_a_known_style(planner):
    model = planner.models['left']

    def probe(xyz, rotation, seed):
        return planner.pose_error('left', xyz, rotation, seed)
    with pytest.raises(ValueError, match='运动学'):
        any_toss(object_frame(), [396, 226], TARGET, base_xy=BASE_XY)
    with pytest.raises(ValueError, match='运动学'):
        any_toss(object_frame(), [396, 226], TARGET, base_xy=BASE_XY, fk=model.fk_tcp_world)
    for style in ('lob', 'whip'):
        with pytest.raises(ValueError, match='抛法只接受'):
            any_toss(object_frame(), [396, 226], TARGET, base_xy=BASE_XY, style=style, fk=model.fk_tcp_world,
                     reach_probe=probe, seed_joints=np.zeros(6))
