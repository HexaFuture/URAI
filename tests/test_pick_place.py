"""Two-point pick and place: the endpoints select the object and its support, lift and carry are automatic."""
import copy
import numpy as np
import pytest
from conftest import make_client
from urai.backend import SimulationBackend
from urai.trajectory import compile_arm
from urai.transfer import pick_place_from_stroke
from test_transfer import LANDING_PIXEL, OBJECT_PIXEL, object_frame, scene


def test_pick_place_uses_endpoints_and_lifts_before_carry_then_lowers_before_release():
    frame = object_frame()
    direct = pick_place_from_stroke(frame, [OBJECT_PIXEL, LANDING_PIXEL], clearance_m=.12, speed_m_s=.08)
    curved = pick_place_from_stroke(frame, [OBJECT_PIXEL, [300, 30], [650, 410], LANDING_PIXEL], clearance_m=.12, speed_m_s=.08)
    assert direct == curved  # Decorative ink cannot accidentally create a low carry trajectory.
    spec = direct['arm']
    points = np.array(spec['path']['points'])
    hover, grasp, lift, arrival, place, retract = points
    np.testing.assert_allclose(hover, lift)
    np.testing.assert_allclose(lift[:2], grasp[:2])
    np.testing.assert_allclose(arrival[:2], place[:2])
    assert lift[2] == arrival[2] and lift[2] >= .06 + .12 - 1e-6
    assert place[2] < arrival[2] and retract[2] > place[2]
    assert direct['placement'] == 'place' and spec['speed'] == .08
    assert spec['approach'] == {'speed_m_s': .8, 'clearance_m': .12}
    plan = compile_arm(spec, SimulationBackend().state()['left'])
    assert [e['opening_mm'] for e in plan['gripper_events']] == [70, 0, 70]
    for event, point in zip(plan['gripper_events'], [hover, grasp, place]):
        i = np.where(plan['time_s'] == event['time_s'])[0][0]
        np.testing.assert_allclose(plan['xyz'][i], point)
        np.testing.assert_allclose(plan['xyz'][i+1], point)
        assert event['wait_for_arrival'] and event['hold_s'] > 0


def test_pick_place_rejects_invalid_pixels_and_raised_or_unknown_destination():
    for pixels in ([], [OBJECT_PIXEL], [OBJECT_PIXEL, [np.nan, 290]], [OBJECT_PIXEL, [800, 290]], [OBJECT_PIXEL, [397, 227]]):
        with pytest.raises(ValueError):
            pick_place_from_stroke(object_frame(), pixels)
    frame = object_frame()
    frame.depth[285:296, 555:566] = np.nan
    with pytest.raises(ValueError, match='depth'):
        pick_place_from_stroke(frame, [OBJECT_PIXEL, LANDING_PIXEL])
    frame.depth[280:300, 550:570] = .92
    with pytest.raises(ValueError, match='桌面'):
        pick_place_from_stroke(frame, [OBJECT_PIXEL, LANDING_PIXEL])


def test_pick_place_api_is_pure_and_checks_arm_and_observation(sim_service):
    service = sim_service
    service.observe()
    client = make_client(service)
    before = (copy.deepcopy(service.draft), service.revision, copy.deepcopy(service.execution))
    body = {'observation_id': service.frame.id, 'arm': 'right', 'pixels': [[395, 220], LANDING_PIXEL]}
    response = client.post('/api/pick-place', json=body)
    assert response.status_code == 200, response.text
    assert response.json()['route_mode'] == 'two-point'
    assert (service.draft, service.revision, service.execution) == before
    assert client.post('/api/pick-place', json={**body, 'arm': 'invalid'}).status_code == 409
    assert client.post('/api/pick-place', json={**body, 'observation_id': 'old'}).status_code == 409


def test_click_on_valid_small_object_is_not_stolen_by_larger_neighbor():
    from urai.transfer import locate_object
    frame = scene([(210, 226, 380, 396, .94), (202, 234, 398, 430, .94)])
    found = locate_object(frame, [387, 217])
    assert found['mask'][217, 387]
    assert not found['mask'][217, 410]


def pinch_scene():
    """A frame whose table carries something flat: cloth never rises far enough to be segmented."""
    frame = object_frame()
    frame.depth[:] = 1
    return frame


def test_a_stroke_from_bare_cloth_pinches_where_it_was_drawn_instead_of_refusing():
    """Folding a shirt means grasping a part of it. Flat material has no segmentable height, so the stroke start is
    pinched where it was drawn instead of being refused for having no separable object."""
    from urai.transfer import PINCH_DEPTH_M, PINCH_RELEASE_HEIGHT_M
    frame = pinch_scene()
    result = pick_place_from_stroke(frame, [[300, 200], [520, 300]], table_z=0.)
    assert result['pinch'] and result['object_width_mm'] == 20.
    points = np.array(result['arm']['path']['points'])
    hover, grasp, lift, arrival, place, retract = points
    # The fingers press into the table: a shirt is a millimetre thick, and fingertips held above the table
    # close on air.
    assert grasp[2] == pytest.approx(-PINCH_DEPTH_M)
    assert place[2] == pytest.approx(PINCH_RELEASE_HEIGHT_M-PINCH_DEPTH_M+.005)
    assert result['finger_z_mm'] == pytest.approx(-PINCH_DEPTH_M*1000)
    # The grasp is where the stroke started, not the centre of anything.
    np.testing.assert_allclose(grasp[:2], frame.pick(300, 200, mode='plane', z=0.)[:2], atol=1e-9)
    np.testing.assert_allclose(place[:2], frame.pick(520, 300, mode='plane', z=0.)[:2], atol=1e-9)
    # One layer of cloth measures below the 1.5 mm the holding gate needs, so verifying it would abort the run.
    events = result['arm']['gripper_events']
    assert [event['opening_mm'] for event in events] == [70, 0, 70]
    assert all('verify' not in event for event in events)
    assert compile_arm(result['arm'], SimulationBackend().state()['left'])['gripper_events'][1]['ramp_s'] == .5


def test_a_pinch_closes_along_the_stroke_and_a_segmented_object_keeps_its_own_axis():
    frame = pinch_scene()
    for start, end in ([[300, 200], [520, 300]], [[300, 300], [520, 200]]):
        result = pick_place_from_stroke(frame, [start, end], table_z=0.)
        travel = np.array(result['target_xyz'][:2])-np.array(result['source_xyz'][:2])
        yaw = result['arm']['orientation']['points'][0][3]
        along = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
        assert abs(abs(along @ travel/np.linalg.norm(travel))-1) < 1e-6      # jaws close along the pull
    # A block that does segment is unaffected: its own narrow axis still decides the closing direction.
    block = pick_place_from_stroke(object_frame(), [OBJECT_PIXEL, LANDING_PIXEL], table_z=0.)
    assert not block['pinch'] and block['arm']['orientation']['points'][0][3] == 0.


def test_pinch_can_be_forced_on_an_object_the_camera_does_segment():
    """A shirt bunched enough to segment would otherwise send the fingers to its centroid; folding needs them
    exactly on the drawn corner."""
    frame = object_frame()
    corner = [409, 239]
    forced = pick_place_from_stroke(frame, [corner, LANDING_PIXEL], table_z=0., pinch=True)
    segmented = pick_place_from_stroke(frame, [corner, LANDING_PIXEL], table_z=0.)
    assert forced['pinch'] and not segmented['pinch']
    np.testing.assert_allclose(forced['source_xyz'][:2], frame.pick(*corner, mode='plane', z=0.)[:2], atol=1e-9)
    assert np.linalg.norm(np.array(forced['source_xyz'][:2])-np.array(segmented['source_xyz'][:2])) > .01
    assert forced['source_xyz'][2] < segmented['source_xyz'][2]


def test_top_down_keeps_the_wrist_vertical_where_a_transfer_would_otherwise_lean():
    """Leaning reaches farther but drags the fingers sideways across whatever lies next to the grasp; on a
    folded sleeve that undoes the fold."""
    from urai.transfer import grasp_tilt_deg
    frame = object_frame()
    stroke = [OBJECT_PIXEL, LANDING_PIXEL]
    leaning = pick_place_from_stroke(frame, stroke, table_z=0., base_xy=(0., 0.))
    upright = pick_place_from_stroke(frame, stroke, table_z=0., base_xy=(0., 0.), top_down=True)
    assert grasp_tilt_deg(leaning['source_xyz'][:2], (0., 0.)) > 5          # the lean is real here
    assert leaning['arm']['orientation']['points'][0][1:3] != [180., 0.]
    assert not leaning['top_down'] and upright['top_down']
    for _, roll, pitch, _ in upright['arm']['orientation']['points']:
        assert (roll, pitch) == (180., 0.)


def test_a_pinch_lands_on_the_table_plane_where_black_cloth_returns_no_depth():
    """The drop point of a fold is on the fabric itself, which reads +-20 mm when it reads at all."""
    from urai.transfer import Unsegmented
    frame = pinch_scene()
    frame.depth[280:320, 500:540] = np.nan
    pinched = pick_place_from_stroke(frame, [[300, 200], [520, 300]], table_z=0.)
    np.testing.assert_allclose(pinched['target_xyz'][:2], frame.pick(520, 300, mode='plane', z=0.)[:2], atol=1e-9)
    # An object transfer still refuses: there its landing height is the one thing the camera must measure.
    blind = object_frame()
    blind.depth[280:320, 500:540] = np.nan
    with pytest.raises(ValueError, match='depth'):
        pick_place_from_stroke(blind, [OBJECT_PIXEL, [520, 300]], table_z=0.)
    assert issubclass(Unsegmented, ValueError)


def test_the_pinch_depth_is_a_bounded_parameter():
    """How hard the jaws press into the table decides whether one layer of cloth is caught at all."""
    from urai.transfer import MAX_PINCH_DEPTH_M
    frame = pinch_scene()
    deep = pick_place_from_stroke(frame, [[300, 200], [520, 300]], table_z=0., pinch_depth_mm=12.)
    assert np.array(deep['arm']['path']['points'])[1, 2] == pytest.approx(-.012)
    flush = pick_place_from_stroke(frame, [[300, 200], [520, 300]], table_z=0., pinch_depth_mm=0.)
    assert np.array(flush['arm']['path']['points'])[1, 2] == pytest.approx(0.)
    for bad in (-1., MAX_PINCH_DEPTH_M*1000+1):
        with pytest.raises(ValueError, match='下探'):
            pick_place_from_stroke(frame, [[300, 200], [520, 300]], table_z=0., pinch_depth_mm=bad)
    # An object grasp is unaffected: its fingers still aim below the object's top, above the table.
    block = pick_place_from_stroke(object_frame(), [OBJECT_PIXEL, LANDING_PIXEL], table_z=0., pinch_depth_mm=12.)
    assert np.array(block['arm']['path']['points'])[1, 2] > .008
