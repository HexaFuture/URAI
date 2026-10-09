"""Staged bimanual bottle opening on the synthetic backend: hold, twist, retract and place.

The synthetic backend has no arm model, so these tests exercise the geometry and the session bookkeeping of the
stages; the stages against the PiPER-X kinematics run end to end in test_sim_tools.py. Observations are the
synthetic nadir frame with a measured surface placed where a test clicks.
"""
import copy
import threading
import time
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.backend import SimulationBackend
from urai.service import Service
from urai.skills import SKILLS, plan, skill_context

AXIS = np.array([0., -1., 0.])


def surface_frame(*points):
    """The synthetic nadir frame of a bare table with a measured surface at each world point, and their pixels."""
    frame = SimulationBackend().capture()
    frame.depth[:] = 1.
    pixels = []
    for point in points:
        u, v = frame.project(point)[0]
        depth = float((np.asarray(point, dtype=float)-frame.t[:3, 3]) @ frame.t[:3, 2])
        frame.depth[int(v)-2:int(v)+3, int(u)-2:int(u)+3] = depth
        pixels.append([float(u), float(v)])
    return frame, pixels


def observe_points(service, *points):
    """Observe, then use a frame that measures ``points`` as the observation; returns their pixels."""
    service.observe()
    service.frame, pixels = surface_frame(*points)
    return pixels


def context(cap=(.4, 0., .25)):
    """A service whose observation shows an upright bottle's cap at ``cap``, and the left arm's context."""
    service = Service(SimulationBackend())
    [pixel] = observe_points(service, cap)
    return service, skill_context(service, 'left'), pixel


def rotation(rpy_deg):
    return Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()


def ready_to_twist(click=None):
    """The left arm holds a horizontal bottle whose cap sits 0.14 m along -Y; the right arm is free to twist."""
    service = Service(SimulationBackend())
    service.backend.openings['left'] = 30
    held = copy.deepcopy(service.backend.state()['left'])
    R = rotation(held['rpy_deg'])
    local = R.T @ (AXIS*.14)
    cap = np.asarray(held['xyz'])+R @ local
    [pixel] = observe_points(service, cap if click is None else click)
    service.bottle_cap = {'id': 'bottle', 'state': 'waiting_cap', 'holder': 'left', 'held_state': held,
                          'axis_local': (R.T @ AXIS).tolist(), 'cap_local': local.tolist(), 'observation_id': 'old-frame'}
    return service, skill_context(service, 'right'), cap, pixel


def fresh_context(service, arm, *points):
    """A new observation of the same scene (a new frame id) and the context of ``arm`` on it."""
    pixels = observe_points(service, *points)
    return skill_context(service, arm), pixels


def run(service, draft):
    """Preview and execute ``draft`` on the synthetic backend in real time; returns the final execution state."""
    service.set_draft({'observation_id': service.frame.id, 'arms': draft})
    preview = service.preview()
    service.execute(preview['id'])
    service.worker.join(timeout=120)
    return service.execution


def away_from_home(service, arm):
    return float(np.linalg.norm(service.backend.positions[arm]-service.backend.home_positions[arm]))


def test_prepare_registered_and_holds_bottle_with_horizontal_axis():
    service, ctx, pixel = context()
    assert 'bottle_cap_prepare' in SKILLS
    result = plan('bottle_cap_prepare', ctx, {'pixels': [pixel]})
    metadata = result['arm']['bottle_cap']
    assert metadata['stage'] == 'prepare'
    assert abs(metadata['axis'][2]) < 1e-9
    assert [e.get('opening_mm') for e in result['arm']['gripper_events']] == [70, 0]
    assert metadata['holder'] == 'left'
    R0 = rotation(result['arm']['orientation']['points'][0][1:])
    R1 = rotation(result['arm']['orientation']['points'][-1][1:])
    np.testing.assert_allclose(R1 @ R0.T @ np.array([0, 0, 1]), metadata['axis'], atol=1e-8)


def test_twist_without_history_uses_explicit_holder_and_fresh_cap_point():
    service, ctx, cap, pixel = ready_to_twist()
    service.bottle_cap = None
    service.backend.openings['left'] = 70  # an opening is not an object-presence detector
    result = plan('bottle_cap_twist', skill_context(service, 'right'), {'pixels': [pixel], 'holder_arm': 'left'})
    assert set(result['arms']) == {'right'}
    assert service.bottle_cap['source'] == 'operator_cap_point'
    assert service.bottle_cap['holder'] == 'left'
    np.testing.assert_allclose(result['cap_xyz'], cap-AXIS*.006)


def test_explicit_holder_replaces_unusable_history_without_regrasp():
    service, ctx, cap, pixel = ready_to_twist()
    service.bottle_cap = {'state': 'invalid'}
    result = plan('bottle_cap_twist', ctx, {'pixels': [pixel], 'holder_arm': 'left'})
    assert set(result['arms']) == {'right'}
    assert result['arms']['right']['bottle_cap']['stage'] == 'twist'
    assert len(result['phases']) == 6


def test_missing_history_does_not_bypass_arm_separation():
    holder = SimulationBackend().state()['left']['xyz']
    service, ctx, cap, pixel = ready_to_twist(click=holder)
    service.bottle_cap = None
    with pytest.raises(ValueError, match='120 mm'):
        plan('bottle_cap_twist', ctx, {'pixels': [pixel], 'holder_arm': 'left'})
    assert service.bottle_cap is None


def test_two_cycles_keep_cap_center_fixed_and_reset_open():
    from urai.skills.bottle_cap import twist_route
    center = np.array([.4, -.35, .25])
    points, rots, events, phases = twist_route(center, AXIS, 45., 2, .06, 800, 0.)
    assert len([e for e in events if e.get('opening_mm') == 0]) == 2
    assert len([e for e in events if e.get('opening_mm') == 70]) == 3
    for phase in phases:
        a, b = phase['grasp'], phase['turned']
        np.testing.assert_allclose(points[a:b+1], np.tile(center, (b-a+1, 1)))
        delta = Rotation.from_matrix(rots[b] @ rots[a].T).as_rotvec()
        assert np.dot(delta, AXIS) == pytest.approx(np.deg2rad(45))
        np.testing.assert_allclose(rots[phase['reset']], rots[a], atol=1e-8)
    for p in points[1:-1]:
        np.testing.assert_allclose(p, center)


def test_grip_offset_cannot_put_body_grip_below_table():
    service, ctx, pixel = context(cap=(.4, 0., .06))
    with pytest.raises(ValueError, match='瓶身'):
        plan('bottle_cap_prepare', ctx, {'pixels': [pixel], 'body_offset_mm': 140})


def test_a_completed_prepare_keeps_holding_and_publishes_the_session():
    """The stage runs through preview and execution; the holder stays where it is instead of returning home."""
    service, ctx, pixel = context()
    result = plan('bottle_cap_prepare', ctx, {'pixels': [pixel], 'speed_m_s': .15, 'approach_speed_m_s': .15,
                                              'angular_speed_deg_s': 45})
    execution = run(service, {'left': result['arm']})
    assert execution['state'] == 'completed', execution
    assert service.bottle_cap['state'] == 'waiting_cap'
    assert service.bottle_cap['observation_id'] == service.frame.id
    assert away_from_home(service, 'left') > .05
    np.testing.assert_allclose(service.bottle_cap['held_state']['xyz'], service.backend.positions['left'])


def test_twist_pose_changes_do_not_drag_physical_pinch_center():
    from urai.skills.bottle_cap import twist_route
    center = np.array([.4, -.35, .25])
    compensation = .01
    points, rots, _, _ = twist_route(center, AXIS, 60, 2, .06, 800, compensation)
    for p, r in zip(points[1:-1], rots[1:-1]):
        np.testing.assert_allclose(p-r[:, 1]*compensation, center, atol=1e-10)


def test_turn_session_rejects_missing_fresh_frame_and_holder_motion():
    from urai.skills.bottle_cap import held_session
    service, ctx, pixel = context()
    service.backend.openings['left'] = 25
    saved = copy.deepcopy(service.backend.state()['left'])
    service.bottle_cap = {'state': 'waiting_cap', 'holder': 'left', 'held_state': saved, 'id': 'test',
                          'observation_id': service.frame.id, 'axis_local': [0, 0, 1], 'cap_local': [0, 0, .14]}
    with pytest.raises(ValueError, match='重新观测'):
        plan('bottle_cap_twist', skill_context(service, 'right'), {'pixels': [pixel]})
    current = service.backend.state()
    current['left']['xyz'][0] += .02
    with pytest.raises(ValueError, match='持瓶臂已移动'):
        held_session(service, current)


def test_cancelled_session_cannot_be_resumed():
    from urai.skills.bottle_cap import held_session
    service, ctx, pixel = context()
    service.backend.openings['left'] = 25
    service.bottle_cap = {'state': 'waiting_cap', 'holder': 'left', 'cancel_epoch': service.cancel_epoch,
                          'held_state': copy.deepcopy(service.backend.state()['left'])}
    service.cancel()
    with pytest.raises(ValueError, match='持瓶|取消'):
        held_session(service, service.backend.state())


def test_twist_compiles_with_two_arrival_gated_close_then_open_resets():
    from urai.skills.bottle_cap import _spec, twist_route
    from urai.trajectory import compile_arm
    service, ctx, pixel = context()
    points, rots, events, _ = twist_route(np.array([.4, -.35, .25]), AXIS, 45, 2, .06, 800, .01)
    result = compile_arm(_spec(points, rots, events, ctx, 10.), ctx.current['left'])
    events = result['gripper_events']
    assert [e.get('opening_mm') for e in events] == [70, 0, 70, 0, 70]
    assert all(e['wait_for_arrival'] for e in events)
    assert all(events[i]['hold_end_time_s'] <= events[i+1]['time_s'] for i in range(4))


def test_cap_click_insets_from_end_face_and_dispatches_only_other_arm():
    service, ctx, cap, pixel = ready_to_twist()
    result = plan('bottle_cap_twist', ctx, {'pixels': [pixel], 'cap_inset_mm': 6})
    assert set(result['arms']) == {'right'}
    np.testing.assert_allclose(result['cap_xyz'], cap-AXIS*.006)


QUICK_TWIST = {'cycles': 1, 'turn_deg': 10, 'angular_speed_deg_s': 30, 'speed_m_s': .15, 'approach_speed_m_s': .15,
               'stand_off_mm': 40}


def test_a_completed_twist_keeps_the_bottle_and_never_homes():
    service, ctx, cap, pixel = ready_to_twist()
    result = plan('bottle_cap_twist', ctx, {'pixels': [pixel], **QUICK_TWIST})
    execution = run(service, result['arms'])
    assert execution['state'] == 'completed', execution
    assert service.bottle_cap['state'] == 'completed' and service.bottle_cap['cycles'] == 1
    assert execution['result']['holding_position']
    assert away_from_home(service, 'right') > .05


def test_holder_motion_during_a_twist_stops_it_and_keeps_the_session_for_relocalization():
    """The holder arm is pushed 2 cm while the other arm twists: the run stops with an error, the stage is not
    completed, nothing homes, and the session waits for a fresh observation."""
    service, ctx, cap, pixel = ready_to_twist()
    result = plan('bottle_cap_twist', ctx, {'pixels': [pixel], **QUICK_TWIST})
    service.set_draft({'observation_id': service.frame.id, 'arms': result['arms']})
    preview = service.preview()

    def push_the_holder():
        while service.execution.get('progress', 0.) <= .3:      # once the twisting arm is well under way
            time.sleep(.01)
        service.backend.positions['left'] = service.backend.positions['left']+[.02, 0., 0.]
    pusher = threading.Thread(target=push_the_holder, daemon=True)
    service.execute(preview['id'])
    pusher.start()
    service.worker.join(timeout=120)
    pusher.join(timeout=5)
    assert service.execution['state'] == 'error' and '持瓶臂已移动' in service.execution['error']
    assert service.bottle_cap['state'] == 'needs_relocalization'
    assert away_from_home(service, 'right') > .05


def test_visible_cap_surface_does_not_raise_grasp_axis():
    service, ctx, cap, pixel = ready_to_twist()
    ctx, [surface] = fresh_context(service, 'right', cap+AXIS*.003+np.array([.008, 0., .020]))
    result = plan('bottle_cap_twist', ctx, {'pixels': [surface], 'cap_inset_mm': 6})
    np.testing.assert_allclose(result['cap_xyz'], cap+AXIS*(.003-.006))
    assert len(result['phases']) == 6


def test_body_inset_moves_into_grip_and_preserves_bottle_geometry():
    service, ctx, pixel = context(cap=(.4, 0., .3))
    a = plan('bottle_cap_prepare', ctx, {'pixels': [pixel], 'body_inset_mm': 0})['arm']
    b = plan('bottle_cap_prepare', ctx, {'pixels': [pixel], 'body_inset_mm': 10})['arm']
    R = rotation(b['orientation']['points'][0][1:])
    np.testing.assert_allclose(np.array(b['path']['points'][1])-a['path']['points'][1], R[:, 2]*.010, atol=1e-9)
    predicted = [np.array(spec['path']['points'][-1])+rotation(spec['orientation']['points'][-1][1:]) @
                 np.array(spec['bottle_cap']['cap_local']) for spec in (a, b)]
    np.testing.assert_allclose(predicted[1], predicted[0], atol=1e-9)


def test_fresh_observation_replans_settled_holder_but_keeps_execution_guard():
    from urai.skills.bottle_cap import held_session
    service, ctx, cap, pixel = ready_to_twist()
    ctx.current['left']['joints_deg'][0] += .58
    service.observation_start = copy.deepcopy(ctx.current)
    old = service.bottle_cap['id']
    plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    assert service.bottle_cap['id'] != old
    held_session(service, ctx.current)
    moved = copy.deepcopy(ctx.current)
    moved['left']['joints_deg'][0] += .58
    with pytest.raises(ValueError, match='持瓶臂已移动'):
        held_session(service, moved)


def test_stale_image_cannot_reanchor_holder():
    service, ctx, cap, pixel = ready_to_twist()
    ctx.current['left']['joints_deg'][0] += .58
    service.observation_start = copy.deepcopy(ctx.current)
    service.bottle_cap['validated_observation_id'] = ctx.frame.id
    with pytest.raises(ValueError, match='更新观测'):
        plan('bottle_cap_twist', ctx, {'pixels': [pixel]})


def test_bad_cap_point_never_updates_holder_reference():
    service, ctx, cap, pixel = ready_to_twist(click=None)
    ctx, [high] = fresh_context(service, 'right', cap+[0, 0, .08])
    ctx.current['left']['joints_deg'][0] += .58
    service.observation_start = copy.deepcopy(ctx.current)
    original = copy.deepcopy(service.bottle_cap)
    with pytest.raises(ValueError, match='偏离'):
        plan('bottle_cap_twist', ctx, {'pixels': [high]})
    assert service.bottle_cap == original


def test_failed_twist_needs_fresh_image_before_replanning():
    from urai.skills.bottle_cap import failed_stage
    service, ctx, cap, pixel = ready_to_twist()
    result = plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    failed_stage(service, {'input_draft': {'arms': result['arms']}}, 'empty grasp')
    service.observation_start = copy.deepcopy(ctx.current)
    with pytest.raises(ValueError, match='更新观测'):
        plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    ctx, [pixel] = fresh_context(service, 'right', cap)
    recovered = plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    assert service.bottle_cap['state'] == 'waiting_cap'
    assert len(recovered['phases']) == 6


def test_worker_retract_preserves_inactive_holding_record():
    from urai.skills.bottle_cap import complete_stage
    service, ctx, cap, pixel = ready_to_twist()
    service.bottle_cap['state'] = 'needs_relocalization'
    complete_stage(service, {'arms': {'right': {}}, 'input_draft': {'arms': {'right': {}}}})
    assert service.bottle_cap['state'] == 'needs_relocalization'


def test_body_grasp_is_flat_deeper_and_finishes_with_horizontal_bottle_axis():
    service, ctx, pixel = context()
    arm = plan('bottle_cap_prepare', ctx, {'pixels': [pixel]})['arm']
    R0 = rotation(arm['orientation']['points'][0][1:])
    R1 = rotation(arm['orientation']['points'][-1][1:])
    assert abs(R0[2, 2]) < 1e-9
    assert SKILLS['bottle_cap_prepare'].defaults()['body_inset_mm'] == 15
    assert abs((R1 @ np.array(arm['bottle_cap']['axis_local']))[2]) < 1e-9


def test_completed_twist_can_repeat_with_fresh_frame_and_invalidates_old_plan():
    from urai.skills.bottle_cap import complete_stage, validate_stage
    service, ctx, cap, pixel = ready_to_twist()
    old = plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    old_plan = {'input_draft': {'arms': old['arms']}, 'observation_id': ctx.frame.id}
    complete_stage(service, old_plan)
    service.observation_start = copy.deepcopy(ctx.current)
    with pytest.raises(ValueError, match='更新观测'):
        plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    ctx, [pixel] = fresh_context(service, 'right', cap)
    repeated = plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    assert len(repeated['phases']) == 6
    assert service.bottle_cap['state'] == 'waiting_cap'
    assert repeated['arms']['right']['bottle_cap']['id'] != old['arms']['right']['bottle_cap']['id']
    with pytest.raises(ValueError, match='会话已改变'):
        validate_stage(service, old_plan, ctx.current)


def test_twist_default_effort_is_1500():
    service, ctx, cap, pixel = ready_to_twist()
    result = plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    closes = [e for e in result['arms']['right']['gripper_events'] if e['opening_mm'] == 0]
    assert len(closes) == 6 and all(e['effort'] == 1500 for e in closes)


def ready_to_place(table_point=(.4, 0., 0.)):
    service, ctx, cap, pixel = ready_to_twist(click=table_point)
    return service, ctx, cap, pixel


def test_place_uprights_bottle_releases_at_support_and_uses_holder():
    service, ctx, cap, pixel = ready_to_place()
    service.bottle_cap['bottle_height_m'] = .21
    result = plan('bottle_cap_place', ctx, {'pixels': [pixel]})
    assert set(result['arms']) == {'left', 'right'}
    arm = result['arms']['left']
    release = result['release_index']
    R = rotation(arm['orientation']['points'][release][1:])
    session = service.bottle_cap
    bottom = np.array(session['cap_local'])-np.array(session['axis_local'])*.21
    np.testing.assert_allclose(np.array(arm['path']['points'][release])+R @ bottom, [.4, 0., 0.], atol=1e-8)
    np.testing.assert_allclose(R @ np.array(session['axis_local']), [0, 0, 1], atol=1e-8)
    assert len(arm['gripper_events']) == 1 and arm['gripper_events'][0]['opening_mm'] == 70
    assert arm['gripper_events'][0]['s'] == release/(len(arm['path']['points'])-1)


def test_place_rejects_non_table_target():
    service, ctx, cap, pixel = ready_to_place((.4, 0., .2))
    with pytest.raises(ValueError, match='桌面'):
        plan('bottle_cap_place', ctx, {'pixels': [pixel]})


def test_default_six_sixty_degree_turns_and_retract_retry():
    service, ctx, cap, pixel = ready_to_twist()
    result = plan('bottle_cap_twist', ctx, {'pixels': [pixel]})
    arm = result['arms']['right']
    assert arm['bottle_cap']['cycles'] == 6
    rots = [Rotation.from_euler('xyz', p[1:], degrees=True) for p in arm['orientation']['points']]
    for phase in result['phases']:
        delta = (rots[phase['turned']]*rots[phase['grasp']].inv()).as_rotvec()
        assert np.dot(delta, AXIS) == pytest.approx(np.deg2rad(60))
    service.bottle_cap = None
    back = plan('bottle_cap_retract', ctx, {'pixels': [pixel], 'holder_arm': 'left'})
    assert set(back['arms']) == {'right'}
    spec = back['arms']['right']
    assert spec['bottle_cap']['stage'] == 'retract'
    assert spec['gripper_events'][0]['opening_mm'] == 70
    points = np.array(spec['path']['points'])
    assert np.linalg.norm(points[-1]-points[0]) == pytest.approx(.07)


def test_a_completed_retract_does_not_home():
    service, ctx, cap, pixel = ready_to_twist()
    service.bottle_cap = None     # no history: an independent retract after a failed first grab
    result = plan('bottle_cap_retract', ctx, {'pixels': [pixel], 'holder_arm': 'left', 'retreat_mm': 20})
    execution = run(service, result['arms'])
    assert execution['state'] == 'completed', execution
    start = np.asarray(ctx.current['right']['xyz'])
    np.testing.assert_allclose(service.backend.positions['right'], np.asarray(result['arms']['right']['path']['points'][-1]),
                               atol=1e-9)
    assert np.linalg.norm(service.backend.positions['right']-start) == pytest.approx(.02)


def test_place_waits_for_worker_retreat_before_holder_motion():
    from urai.trajectory import compile_arm
    service, ctx, cap, pixel = ready_to_place()
    result = plan('bottle_cap_place', ctx, {'pixels': [pixel]})
    worker = compile_arm(result['arms']['right'], ctx.current['right'])
    holder = compile_arm(result['arms']['left'], ctx.current['left'])
    finish = worker['time_s'][-1]
    assert holder['start_hold_s'] > finish
    np.testing.assert_allclose(holder['xyz'][0], holder['xyz'][1])
    assert worker['gripper_events'][0]['opening_mm'] == 70
    assert holder['gripper_events'][0]['time_s'] > finish


def test_place_reverses_saved_positions_without_new_rotation_center():
    service, ctx, cap, pixel = ready_to_place((.4, 0, 0))
    held = ctx.current['left']
    R = rotation(held['rpy_deg'])
    upright = Rotation.align_vectors([[0, 0, 1]], [AXIS])[0].as_matrix() @ R
    positions = [(np.array(held['xyz'])+[.015, 0, .03]).tolist(), held['xyz']]
    service.bottle_cap['upright_positions'] = positions
    service.bottle_cap['upright_rotations'] = [upright.tolist(), R.tolist()]
    result = plan('bottle_cap_place', ctx, {'pixels': [pixel]})
    np.testing.assert_allclose(result['arms']['left']['path']['points'][1:3], list(reversed(positions)))


@pytest.mark.parametrize('already_clear', [.07, .20])
def test_place_counts_existing_cap_clearance_before_worker_home(already_clear):
    service, ctx, cap, pixel = ready_to_place()
    ctx.current['right']['xyz'] = (cap+AXIS*already_clear).tolist()
    result = plan('bottle_cap_place', ctx, {'pixels': [pixel], 'worker_retreat_mm': 170})
    points = np.asarray(result['arms']['right']['path']['points'])
    np.testing.assert_allclose(points[-1], cap+AXIS*max(.17, already_clear), atol=1e-9)
    assert result['arms']['right']['gripper_events'][0]['opening_mm'] == 70
