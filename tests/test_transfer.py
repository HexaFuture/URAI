"""Top-down tabletop transfers: segmentation, grasp and release heights, the event dwells and the direct transfer
the task queue uses. Frames are synthetic depth images of a flat table; the planning test runs the real planner."""
import numpy as np
import pytest
from conftest import make_client
from urai.approach import with_approaches
from urai.backend import SimulationBackend
from urai.trajectory import compile_arm, compile_arms
from urai.transfer import PLACE_RELEASE_HEIGHT_M, pick_place_from_stroke

OBJECT_PIXEL = [396, 226]
LANDING_PIXEL = [560, 290]


def object_frame():
    """The simulation's nadir camera 1 m above a flat table carrying one 60 mm tall block."""
    frame = SimulationBackend().capture()
    frame.depth[:] = 1
    frame.depth[210:242, 380:412] = .94
    return frame


def scene(blocks):
    """Flat table at 1 m depth with raised blocks given as (v0, v1, u0, u1, depth_m)."""
    frame = SimulationBackend().capture()
    frame.depth[:] = 1.
    for v0, v1, u0, u1, depth in blocks:
        frame.depth[v0:v1, u0:u1] = depth
    return frame


def test_gripper_dwells_keep_exact_event_pose_and_precede_lifting():
    spec = {'path': {'mode': 'waypoints', 'points': [[.2, 0, .2], [.2, 0, .05], [.3, 0, .2], [.4, 0, .05], [.4, 0, .2],
                                                     [.5, 0, .2], [.6, 0, .2]]},
            'speed': .1, 'gripper_events': [{'s': 1/6, 'opening_mm': 0, 'hold_s': .8, 'wait_for_arrival': True},
                                            {'s': .5, 'opening_mm': 70, 'hold_s': .6, 'wait_for_arrival': True}]}
    state = SimulationBackend().state()
    p = compile_arm(spec, state['left'])
    for event, point in zip(p['gripper_events'], [spec['path']['points'][1], spec['path']['points'][3]]):
        i = np.where(p['time_s'] == event['time_s'])[0][0]
        assert event['hold_s'] > 0
        assert event['wait_for_arrival']
        np.testing.assert_allclose(p['xyz'][i], point)
        np.testing.assert_allclose(p['xyz'][i+1], point)
        assert p['time_s'][i+1]-p['time_s'][i] >= event['hold_s']-1e-9
        assert i in p['stop_indices'] and i+1 in p['stop_indices']
    joined = with_approaches({'left': p}, state, {'left': 'direct'})['left']
    assert joined['gripper_events'][0]['time_s'] == pytest.approx(p['gripper_events'][0]['time_s']+joined['task_offset_s'])


def test_pick_place_uses_object_and_landing_depth_not_background():
    frame = object_frame()
    result = pick_place_from_stroke(frame, [OBJECT_PIXEL, LANDING_PIXEL], table_z=0)
    spec = result['arm']
    points = np.array(spec['path']['points'])
    assert result['object_width_mm'] < 70
    assert 0 < points[1, 2] < .06
    assert points[-2, 2] == pytest.approx(points[1, 2]+PLACE_RELEASE_HEIGHT_M+.005)
    assert min(points[2:-2, 2]) >= .12-1e-9
    assert [e['opening_mm'] for e in spec['gripper_events']] == [70, 0, 70]
    assert spec['gripper_events'][1]['verify'] == 'holding'
    assert all(e['wait_for_arrival'] and e['hold_s'] >= .6 for e in spec['gripper_events'])
    plan = compile_arm(spec, SimulationBackend().state()['left'])
    for event in plan['gripper_events']:
        i = np.where(plan['time_s'] == event['time_s'])[0][0]
        assert np.linalg.norm(plan['xyz'][i+1]-plan['xyz'][i]) < 1e-12
    # Missing depth away from both ends does not change anything: only the two endpoints are measured.
    frame.depth[120:160, 460:510] = np.nan
    again = pick_place_from_stroke(frame, [OBJECT_PIXEL, LANDING_PIXEL], table_z=0)
    np.testing.assert_allclose(again['arm']['path']['points'], spec['path']['points'])


def test_pick_place_rejects_unknown_landing_depth_but_reports_a_wide_object():
    frame = object_frame()
    frame.depth[288:294, 558:564] = np.nan
    with pytest.raises(ValueError, match='depth'):
        pick_place_from_stroke(frame, [OBJECT_PIXEL, LANDING_PIXEL], table_z=0)
    # A width that reads wider than the jaws is reported, not refused: the estimate is inflated by handles,
    # neighbours and depth halos, and the jaws open to 70 mm anyway.
    wide = pick_place_from_stroke(SimulationBackend().capture(), [[395, 220], LANDING_PIXEL], table_z=0)
    assert wide['object_width_mm'] > 70 and wide['arm']['path']['points']


def test_draft_preview_and_execute_refuse_a_revision_race_and_a_cancel(sim_service):
    service = sim_service
    service.observe()
    client = make_client(service)
    spec = {'path': {'mode': 'waypoints', 'points': [[.25, .12, .24], [.26, .12, .24]]}, 'speed': .1}
    body = {'observation_id': service.frame.id, 'arms': {'left': spec}, 'expected_revision': service.revision-1}
    assert client.put('/api/draft', json=body).status_code == 409
    body['expected_revision'] = service.revision
    saved = client.put('/api/draft', json=body).json()
    epoch = client.get('/api/state').json()['cancel_epoch']
    preview = client.post('/api/preview', json={'expected_revision': saved['revision'], 'expected_cancel_epoch': epoch})
    assert preview.status_code == 200, preview.text
    client.post('/api/cancel')
    assert client.post('/api/execute', json={'preview_id': preview.json()['id'], 'cancel_epoch': epoch}).status_code == 409
    assert service.execution['state'] != 'running'
    assert client.post('/api/preview', json={'expected_cancel_epoch': epoch}).status_code == 409


def test_real_planner_keeps_the_transfer_event_dwells_stationary(planner, working_start):
    """The retimed joint curve of a pick-and-place stands still through every gripper dwell."""
    model = planner.models['left']
    spec = pick_place_from_stroke(object_frame(), [OBJECT_PIXEL, LANDING_PIXEL], table_z=model.table_z_mm/1000,
                                  base_xy=model.base_xy, preferred_yaw_deg=working_start['left']['rpy_deg'][2])['arm']
    planned = planner.plan(compile_arms({'left': spec}, working_start), working_start)['arms']['left']
    assert planned['gripper_events'][0]['time_s'] == planned['task_offset_s']
    for event in planned['gripper_events']:
        assert event['hold_end_time_s']-event['time_s'] >= event['hold_s']-1e-8
        q = planned['curve'](event['time_s'])
        for t in np.linspace(event['time_s'], event['hold_end_time_s'], 9):
            np.testing.assert_allclose(planned['curve'](t), q, atol=1e-8)
            np.testing.assert_allclose(planned['curve'](t, 1), 0, atol=1e-8)


def test_objects_in_polygon_lists_raised_regions_near_to_far_with_jaw_check():
    from urai.transfer import objects_in_polygon
    near = (100, 132, 300, 332, .94)
    wide = (200, 270, 380, 450, .94)
    far = (300, 332, 500, 532, .96)
    outside = (400, 432, 100, 132, .94)
    frame = scene([near, wide, far, outside])
    polygon = [[280, 80], [560, 80], [560, 350], [280, 350]]
    objects = objects_in_polygon(frame, polygon, table_z=0.)
    assert [o['index'] for o in objects] == [0, 1, 2]
    assert [o['graspable'] for o in objects] == [True, False, True]
    assert '宽' in objects[1]['reason'] and objects[0]['reason'] is None
    centres = [o['center_xy'][0] for o in objects]
    assert centres == sorted(centres)
    for o, block in zip(objects, (near, wide, far)):
        u, v = o['pixel']
        assert block[0] <= v < block[1] and block[2] <= u < block[3]
        assert frame.depth[v, u] == block[4]
        assert o['height_mm'] == pytest.approx((1-block[4])*1000, abs=5)
    assert objects[0]['width_mm'] < 60 < 100 < objects[1]['width_mm']
    assert objects == objects_in_polygon(frame, polygon, table_z=0.)
    with pytest.raises(ValueError, match='3'):
        objects_in_polygon(frame, [[1, 1], [2, 2]])
    assert objects_in_polygon(frame, [[0, 0], [50, 0], [50, 50]]) == []


def test_transfer_between_matches_the_two_point_pick_place():
    """The task queue's direct transfer and the pick_place stroke build the same draft from the same two ends."""
    from urai.transfer import transfer_between
    frame = object_frame()
    stroke = pick_place_from_stroke(frame, [OBJECT_PIXEL, LANDING_PIXEL], table_z=0)
    landing = frame.pick(*LANDING_PIXEL, mode='surface')
    direct = transfer_between(frame, OBJECT_PIXEL, landing, table_z=0)
    assert direct['source_xyz'] == stroke['source_xyz']
    assert direct['target_xyz'] == stroke['target_xyz']
    assert direct['arm']['gripper_events'] == stroke['arm']['gripper_events']
    assert [event['s'] for event in direct['arm']['gripper_events']] == [0., 1/5, 4/5]
    assert direct['arm']['orientation'] == stroke['arm']['orientation']
    direct_points = direct['arm']['path']['points']
    assert direct_points == stroke['arm']['path']['points']
    assert direct_points[1] == direct['source_xyz'] and direct_points[4] == direct['target_xyz']
    assert direct['route_mode'] == 'direct' and stroke['route_mode'] == 'two-point'
    assert direct['placement'] == stroke['placement'] == 'place'
    with pytest.raises(ValueError, match='物体'):
        transfer_between(frame, [600, 260], landing, table_z=0)
    with pytest.raises(ValueError, match='落点'):
        transfer_between(frame, OBJECT_PIXEL, [.4, float('nan'), 0.], table_z=0)


def test_landing_is_not_checked_for_occupancy_and_drop_releases_above_it():
    from urai.transfer import DROP_RELEASE_HEIGHT_M, transfer_between
    frame = scene([(210, 242, 380, 412, .94), (280, 312, 540, 572, .96)])
    landing = frame.pick(575, 296, mode='surface')
    # Another block sits beside the landing; the chosen landing is honoured without an occupancy check.
    assert pick_place_from_stroke(frame, [OBJECT_PIXEL, [575, 296]], table_z=0)['placement'] == 'place'
    assert transfer_between(frame, OBJECT_PIXEL, landing, table_z=0)['placement'] == 'place'
    dropped = transfer_between(frame, OBJECT_PIXEL, landing, table_z=0, drop=True)
    placed = transfer_between(object_frame(), OBJECT_PIXEL, landing, table_z=0)
    assert dropped['placement'] == 'drop'
    assert dropped['source_xyz'] == placed['source_xyz']
    assert dropped['target_xyz'][2] == pytest.approx(placed['target_xyz'][2]+DROP_RELEASE_HEIGHT_M-PLACE_RELEASE_HEIGHT_M)
    strip = lambda events: [{k: v for k, v in e.items() if k != 's'} for e in events]
    assert strip(dropped['arm']['gripper_events']) == strip(placed['arm']['gripper_events'])
    assert dropped['target_xyz'][:2] == pytest.approx(landing[:2].tolist())
    points = np.array(dropped['arm']['path']['points'])
    assert len(points) == 5 and points[3, 2] >= points[4, 2]+.08-1e-9   # arrival above the release, no climb after


def test_transfer_between_leans_the_release_back_toward_the_base():
    from scipy.spatial.transform import Rotation
    from urai.transfer import grasp_tilt_deg, transfer_between
    frame = object_frame()
    landing = [.45, -.25, 0.]
    result = transfer_between(frame, OBJECT_PIXEL, landing, base_xy=(0., 0.), release_tilt_deg=60., drop=True)
    frames = result['arm']['orientation']['points']
    assert len(frames) == 4 and result['release_tilt_deg'] == 60.
    assert frames[0][1:] == frames[1][1:]
    grasp_tool_z = Rotation.from_euler('xyz', frames[0][1:], degrees=True).as_matrix()[:, 2]
    assert np.degrees(np.arccos(-grasp_tool_z[2])) == pytest.approx(grasp_tilt_deg(result['source_xyz'][:2], (0., 0.)), abs=.5)
    tool_z = Rotation.from_euler('xyz', frames[-1][1:], degrees=True).as_matrix()[:, 2]
    assert np.degrees(np.arccos(-tool_z[2])) == pytest.approx(60., abs=1.)
    place = np.array(result['target_xyz'])
    assert tool_z[:2] @ place[:2] > 0                       # fingertips lean away from the base at the origin
    assert frames[1][0] < frames[2][0] < frames[3][0] == 1


def test_drop_transfers_end_at_the_release_point():
    from urai.transfer import transfer_between
    frame = object_frame()
    result = transfer_between(frame, OBJECT_PIXEL, [.45, -.25, 0.], drop=True)
    points = np.array(result['arm']['path']['points'])
    events = result['arm']['gripper_events']
    assert len(points) == 5                                    # hover, grasp, hover, arrival, release
    np.testing.assert_allclose(points[-1], result['target_xyz'])
    assert events[-1]['s'] == 1 and events[-1]['opening_mm'] == 70 and 'verify' not in events[-1]
    placed = transfer_between(frame, OBJECT_PIXEL, [.45, -.25, 0.], drop=False)
    assert len(placed['arm']['path']['points']) == 6           # placing still retracts upward


def test_depth_only_candidates_need_three_centimetres_but_coloured_ones_do_not():
    from urai.transfer import objects_in_polygon
    h, w = scene([]).depth.shape
    everything = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
    # A 2 cm grey bump is indistinguishable from the table depth noise of a head camera under bright light.
    assert objects_in_polygon(scene([(300, 332, 690, 722, .98)]), everything, table_z=0) == []
    # The same bump inside the orange patch of the synthetic image is a coloured object.
    coloured = objects_in_polygon(scene([(210, 242, 380, 412, .98)]), everything, table_z=0)
    assert len(coloured) == 1 and 8 < coloured[0]['height_mm'] < 35
    # A 4 cm grey block still qualifies by depth alone.
    assert len(objects_in_polygon(scene([(300, 332, 690, 722, .96)]), everything, table_z=0)) == 1


def test_placements_release_above_the_support_and_far_grasps_lean_toward_the_base():
    from scipy.spatial.transform import Rotation
    from urai.transfer import (GRASP_TILT_FULL_RADIUS_M, GRASP_TILT_START_RADIUS_M, GRIP_RAMP_S, MAX_GRASP_TILT_DEG,
                               grasp_tilt_deg, transfer_between)
    frame = object_frame()
    landing = frame.pick(575, 296, mode='surface')
    placed = transfer_between(frame, OBJECT_PIXEL, landing, table_z=0)
    grasp = np.array(placed['source_xyz'])
    # The place pose sits PLACE_RELEASE_HEIGHT_M higher than a grasp at the same finger height would.
    assert placed['target_xyz'][2] == pytest.approx(grasp[2]+PLACE_RELEASE_HEIGHT_M+.005+landing[2], abs=1e-9)
    close, release = placed['arm']['gripper_events'][1], placed['arm']['gripper_events'][2]
    assert close['ramp_s'] == GRIP_RAMP_S and release['ramp_s'] == GRIP_RAMP_S
    assert grasp_tilt_deg(grasp[:2], grasp[:2]-[GRASP_TILT_START_RADIUS_M-.01, 0.]) == 0.
    assert grasp_tilt_deg(grasp[:2], grasp[:2]-[GRASP_TILT_FULL_RADIUS_M+.1, 0.]) == MAX_GRASP_TILT_DEG
    near = transfer_between(frame, OBJECT_PIXEL, landing, table_z=0, base_xy=(grasp[:2]-[.20, 0.]).tolist())
    far = transfer_between(frame, OBJECT_PIXEL, landing, table_z=0, base_xy=(grasp[:2]-[.50, 0.]).tolist())
    tool = lambda key: Rotation.from_euler('xyz', key[1:], degrees=True).as_matrix()[:, 2]
    assert np.degrees(np.arccos(-tool(near['arm']['orientation']['points'][0])[2])) == pytest.approx(0., abs=1e-6)
    expected = grasp_tilt_deg(grasp[:2], grasp[:2]-[.50, 0.])
    assert 0 < expected < MAX_GRASP_TILT_DEG
    assert np.degrees(np.arccos(-tool(far['arm']['orientation']['points'][0])[2])) == pytest.approx(expected, abs=.5)
    assert far['arm']['orientation']['points'][1][1:] == far['arm']['orientation']['points'][0][1:]
