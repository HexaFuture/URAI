"""Grasp-line drafts: a short line across an object fixes the grasp point and the closing axis."""
import numpy as np
import pytest
from conftest import make_client
from urai.approach import with_approaches
from urai.backend import SimulationBackend
from urai.grasp import grasp_from_line, grasp_terms
from urai.trajectory import compile_arm, compile_arms

LINE = [[380, 220], [410, 220]]


def test_grasp_line_uses_surface_midpoint_and_world_closing_direction():
    backend = SimulationBackend()
    frame = backend.capture()
    result = grasp_from_line(frame, LINE, inset_mm=10)
    spec = result['arm']
    p = np.array(spec['path']['points'])
    center = frame.pick(395, 220, 'surface')
    np.testing.assert_allclose(p[-1], center-[0, 0, .01])
    assert p[0, 2]-center[2] == pytest.approx(.12)
    compiled = compile_arm(spec, backend.state()['left'])
    np.testing.assert_allclose(np.abs(compiled['rotation_matrices'][-1][:, 0]), [0, 1, 0], atol=1e-8)
    np.testing.assert_allclose(compiled['rotation_matrices'][-1][:, 2], [0, 0, -1], atol=1e-8)
    assert compiled['gripper_events'][0]['time_s'] == 0
    assert compiled['time_s'][1] >= .6
    np.testing.assert_array_equal(compiled['xyz'][0], compiled['xyz'][1])
    assert compiled['stop_indices'][0] == 1
    assert compiled['gripper_events'][-1]['hold_end_time_s'] == compiled['time_s'][-1]
    assert compiled['gripper_events'][-1]['wait_for_arrival']
    joined = with_approaches({'left': compiled}, backend.state(), {'left': 'direct'})['left']
    i = next(i for i in joined['stop_indices'] if joined['time_s'][i] >= joined['task_offset_s']+.6-1e-9)
    assert joined['time_s'][i] >= joined['task_offset_s']+.6-1e-9
    np.testing.assert_array_equal(joined['xyz'][i-1], joined['xyz'][i])


def test_grasp_rejects_unknown_depth_and_clamps_long_line():
    frame = SimulationBackend().capture()
    result = grasp_from_line(frame, [[300, 220], [500, 220]])
    assert result['width_mm'] == 70
    assert result['width_clamped']
    assert result['drawn_width_mm'] > 70
    frame.depth[:] = np.nan
    with pytest.raises(ValueError, match='depth'):
        grasp_from_line(frame, LINE)


def test_grasp_api_generates_draft_without_mutation_or_motion(sim_service):
    service = sim_service
    service.observe()
    client = make_client(service)
    request = {'observation_id': service.frame.id, 'pixels': LINE, 'inset_mm': 10}
    response = client.post('/api/grasp', json=request)
    assert response.status_code == 200
    # The jaws are ramped shut over half a second; the torque limit is what stops them on the object.
    assert response.json()['arm']['gripper_events'][-1] == {'s': 1, 'opening_mm': 0, 'hold_s': .6, 'wait_for_arrival': True,
                                                            'ramp_s': .5}
    assert service.draft == {} and service.execution['state'] == 'idle'
    request['observation_id'] = 'old'
    assert client.post('/api/grasp', json=request).status_code == 409


def test_pregrasp_is_above_surface_even_for_deep_inset():
    frame = SimulationBackend().capture()
    result = grasp_from_line(frame, LINE, inset_mm=50, clearance_m=.02)
    hover, target = np.array(result['arm']['path']['points'])
    assert hover[2]-result['surface_xyz'][2] == pytest.approx(.02)
    assert hover[2]-target[2] == pytest.approx(.07)


def test_grasp_axis_uses_nearby_equivalent_yaw_independent_of_stroke_direction():
    frame = SimulationBackend().capture()
    a = grasp_from_line(frame, LINE, preferred_yaw_deg=-92)
    b = grasp_from_line(frame, LINE[::-1], preferred_yaw_deg=-92)
    assert a['arm']['orientation'] == b['arm']['orientation']
    assert a['arm']['orientation']['points'][0][3] == pytest.approx(-90)
    np.testing.assert_allclose(a['arm']['path']['points'], b['arm']['path']['points'])


def test_grasp_api_uses_selected_arm_current_orientation(sim_service):
    service = sim_service
    service.backend.rotations['left'] = [180, 0, -92]
    service.backend.rotations['right'] = [180, 0, 80]
    service.observe()
    client = make_client(service)
    for arm, yaw in [('left', -90), ('right', 90)]:
        response = client.post('/api/grasp', json={'observation_id': service.frame.id, 'pixels': LINE, 'arm': arm})
        assert response.status_code == 200
        # A leaning pose keeps the jaw direction, but its xyz-Euler yaw is no longer exactly the jaw yaw.
        assert response.json()['arm']['orientation']['points'][0][3] == pytest.approx(yaw, abs=1.)
    assert client.post('/api/grasp', json={'observation_id': service.frame.id, 'pixels': LINE, 'arm': 'bad'}).status_code == 409


def table_line(service, center_xy):
    """Pixels of a 30 mm line across bare table at ``center_xy`` in the service's current observation."""
    table_z = service.table_z('left')
    ends = [[center_xy[0], center_xy[1]-.015, table_z], [center_xy[0], center_xy[1]+.015, table_z]]
    return service.frame.project(ends).tolist()


def test_default_inset_respects_the_calibrated_fingertip_floor_and_reports_it(piper_service):
    """With table checks on, a 10 mm inset into bare table stops 1 mm above the model's fingertip floor."""
    service = piper_service
    service.observe()
    model = service.backend.models['left']
    response = make_client(service).post('/api/grasp', json={'observation_id': service.frame.id, 'arm': 'left',
                                                              'pixels': table_line(service, [.30, -.05]), 'inset_mm': 10})
    assert response.status_code == 200, response.text
    result = response.json()
    floor = (model.table_z_mm-model.fingertip_below_table_mm-model.fingertip_bias_mm+1.)/1000
    assert result['arm']['path']['points'][-1][2] == pytest.approx(floor)
    # The surface is levelled to the calibrated table first, then the target is bounded by the floor.
    assert result['surface_xyz'][2] == pytest.approx(model.table_z_mm/1000, abs=1e-4)   # float32 depth
    assert result['actual_inset_mm'] == pytest.approx(model.table_z_mm-floor*1000, abs=.1)
    assert result['table_limited'] is True
    assert result['grasp_mode'] == 'top_down'


def test_default_inset_with_table_checks_off_is_bounded_to_a_three_mm_table_press(piper_service):
    from urai.grasp import TABLE_PRESS_M
    service = piper_service
    service.set_settings({'table_checks': False})
    service.observe()
    model = service.backend.models['left']
    response = make_client(service).post('/api/grasp', json={'observation_id': service.frame.id, 'arm': 'left',
                                                              'pixels': table_line(service, [.30, -.05]), 'inset_mm': 10})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['arm']['path']['points'][-1][2] == pytest.approx(model.table_z_mm/1000-TABLE_PRESS_M)
    assert result['table_limited'] is True


def test_table_levelling_changes_only_z_and_preserves_measured_surface_xy():
    """On a tilted depth table the measured midpoint keeps its XY; only its height is re-levelled."""
    frame = SimulationBackend().capture()
    height, width = frame.depth.shape
    frame.depth[:] = 1.-np.tile(.02*np.arange(width)/(width-1)-.006, (height, 1))   # table reads -6 mm .. +14 mm
    raw = frame.pick(395, 220, 'surface')
    assert abs(raw[2]) > .002
    result = grasp_from_line(frame, LINE, table_z_m=0.)
    np.testing.assert_allclose(result['surface_xyz'][:2], raw[:2])
    assert result['surface_xyz'][2] == pytest.approx(0., abs=1e-3)


def test_grasp_floor_rejects_surface_below_calibration_instead_of_grasping_air():
    frame = SimulationBackend().capture()
    frame.depth[:] = 1.
    with pytest.raises(ValueError, match='观测表面低于'):
        grasp_from_line(frame, LINE, min_tcp_z_m=.01)


def test_release_line_keeps_grip_until_the_lowered_endpoint():
    sim = SimulationBackend()
    frame = sim.capture()
    result = grasp_from_line(frame, LINE, endpoint_action='release')
    spec = result['arm']
    compiled = compile_arm(spec, sim.state()['left'])
    assert spec['grasp_line'] is False and spec['start_hold_s'] == 0
    assert spec['orientation'] == {'mode': 'free'}
    assert len(compiled['gripper_events']) == 1
    event = compiled['gripper_events'][0]
    assert event['opening_mm'] == 70 and event['s'] == 1 and event['wait_for_arrival']
    assert event['hold_end_time_s']-event['time_s'] >= .6-1e-8
    index = np.searchsorted(compiled['time_s'], event['time_s'])
    np.testing.assert_allclose(compiled['xyz'][index], spec['path']['points'][-1])
    assert compiled['xyz'][index, 2] < compiled['xyz'][0, 2]


def test_release_line_api_and_hold_selection_do_not_preopen(sim_service):
    service = sim_service
    service.observe()
    client = make_client(service)
    request = {'observation_id': service.frame.id, 'pixels': LINE, 'endpoint_action': 'release'}
    result = client.post('/api/grasp', json=request)
    assert result.status_code == 200
    assert [e['s'] for e in result.json()['arm']['gripper_events']] == [1]
    request['endpoint_action'] = 'none'
    assert client.post('/api/grasp', json=request).json()['arm']['gripper_events'] == []
    request['endpoint_action'] = 'invalid'
    assert client.post('/api/grasp', json=request).status_code == 409


def test_open_dwell_is_preserved_by_the_real_planner_with_approach(planner, working_start):
    model = planner.models['left']
    spec = grasp_from_line(SimulationBackend().capture(), LINE, base_xy=model.base_xy,
                           preferred_yaw_deg=working_start['left']['rpy_deg'][2])['arm']
    planned = planner.plan(compile_arms({'left': spec}, working_start), working_start)['arms']['left']
    user = planned['user_trajectory']
    opening, closing = planned['gripper_events']
    assert opening['time_s'] == user['time_s'][0] == planned['task_offset_s']
    assert closing['hold_end_time_s'] == user['time_s'][-1]
    assert user['time_s'][1]-opening['time_s'] >= .6-1e-8
    q = planned['curve'](opening['time_s'])
    for t in np.linspace(opening['time_s'], user['time_s'][1], 9):
        np.testing.assert_allclose(planned['curve'](t), q, atol=1e-8)
        np.testing.assert_allclose(planned['curve'](t, 1), 0, atol=1e-8)


def test_a_grasp_line_ramps_the_jaws_shut_and_takes_a_softer_torque():
    """Snapping shut at the block torque squeezes a soft object (a bread half) out of the fingers."""
    frame = SimulationBackend().capture()
    soft = grasp_from_line(frame, LINE, grip_effort=300)['arm']['gripper_events'][-1]
    assert soft == {'s': 1, 'opening_mm': 0, 'hold_s': .6, 'wait_for_arrival': True, 'ramp_s': .5, 'effort': 300}
    default = grasp_from_line(frame, LINE)['arm']['gripper_events'][-1]
    assert 'effort' not in default and default['ramp_s'] == .5   # the runtime's own torque, ramped
    release = grasp_from_line(frame, LINE, endpoint_action='release', grip_effort=300)['arm']['gripper_events'][-1]
    assert release['opening_mm'] == 70 and release['effort'] == 300 and release['ramp_s'] == .5
    assert grasp_from_line(frame, LINE, endpoint_action='none')['arm']['gripper_events'] == []
    for bad in (0, 49, 5001, 300.5):
        with pytest.raises(ValueError, match='夹持力矩'):
            grasp_from_line(frame, LINE, grip_effort=bad)
    plan = compile_arm(grasp_from_line(frame, LINE, grip_effort=300)['arm'], SimulationBackend().state()['left'])
    assert plan['gripper_events'][-1]['ramp_s'] == .5 and plan['gripper_events'][-1]['effort'] == 300


def test_a_grasp_line_takes_the_drawn_midpoint_however_low_it_reads():
    """Surfaces reading below the table are not refused.

    A refusal would compare the midpoint against one global table constant, while the same real table can span
    35 mm of world Z across a frame. The risk it guarded against is real - a white plate on a white table reads
    through and the fingers descend to that reading - so the depth is taken as measured and the risk stays with
    whoever draws the line.
    """
    frame = SimulationBackend().capture()
    frame.depth[:] = 1.
    table = frame.pick(395, 220, 'surface')[2]
    # Levelling against the fitted table leaves a flat table reading where it was, and refuses nothing.
    assert grasp_from_line(frame, LINE, table_z_m=table)['surface_xyz'][2] == pytest.approx(table)
    read_through = SimulationBackend().capture()
    read_through.depth[:] = 1.
    read_through.depth[218:223, 393:398] = 1.02
    assert grasp_from_line(read_through, LINE, table_z_m=table)['surface_xyz'][2] < table
    # A real object above the table is unaffected.
    raised = SimulationBackend().capture()
    raised.depth[:] = 1.
    raised.depth[210:242, 380:412] = .94
    assert grasp_from_line(raised, LINE, table_z_m=table)['surface_xyz'][2] > table
    assert grasp_from_line(frame, LINE)['surface_xyz'][2] == pytest.approx(table)
    # Read through to two centimetres under the table: used as given, not refused.
    sunk = SimulationBackend().capture()
    sunk.depth[:] = 1.
    sunk.depth[210:242, 380:412] = 1.02
    result = grasp_from_line(sunk, LINE)
    assert result['surface_xyz'][2] < table-.015
    assert np.asarray(result['arm']['path']['points'])[-1][2] < table-.015
    assert grasp_from_line(raised, LINE)['surface_xyz'][2] > table


def test_leaning_toward_the_base_stays_available_and_top_down_turns_it_off():
    """Two real cases pull opposite ways, so this is the caller's choice rather than a default.

    A grasp line 0.6 m out needs the lean: an upright tool reaches about 0.33 m from a PiPER base. A garment
    needs the opposite: a 40-degree approach puts one finger down first and the pinch misses the fabric.
    """
    frame = SimulationBackend().capture()
    far = [0., 0.]           # the grasp sits about 0.42 m from here: past the upright reach
    leaning = grasp_from_line(frame, LINE, base_xy=far)['arm']['orientation']['points'][0][1:]
    vertical = grasp_from_line(frame, LINE, base_xy=far, top_down=True)['arm']['orientation']['points'][0][1:]
    assert vertical[0] == 180 and vertical[1] == 0, '全程竖直：手腕朝下'
    assert leaning != vertical, '默认仍然向基座后仰，远处的抓取线靠它才够得到'
    # Close to the base there is nothing to gain, so both agree.
    near = grasp_from_line(frame, LINE, base_xy=[.35, -.2])['arm']['orientation']['points'][0][1:]
    assert near == vertical


def test_the_endpoint_and_the_skill_registry_plan_from_the_same_calibration(arm_models):
    """The grasp endpoint, the skills and the task queue all take these terms from grasp.grasp_terms, so the same
    stroke cannot plan two different motions."""
    from urai.grasp import PINCH_COMPENSATION_M, TABLE_PRESS_M
    left, right = arm_models['left'], arm_models['right']
    off = grasp_terms('left', left, table_checks=False)
    assert off['min_tcp_z_m'] == pytest.approx(left.table_z_mm/1000-TABLE_PRESS_M), '不查桌面时压到桌面下 3 mm'
    assert off['pinch_compensation_m'] == PINCH_COMPENSATION_M['left']
    on = grasp_terms('right', right, table_checks=True)
    expected = right.t_world_base[2, 3]+(right.table_z_mm-right.fingertip_below_table_mm-right.fingertip_bias_mm+1.)/1000
    assert on['min_tcp_z_m'] == pytest.approx(expected), '查桌面时沿用指尖下限加 1 mm 余量'
    assert on['pinch_compensation_m'] == PINCH_COMPENSATION_M['right'] < 0, '右腕工具镜像安装，符号相反'
    assert grasp_terms('left', None, table_checks=False) == {
        'table_z_m': None, 'min_tcp_z_m': None, 'pinch_compensation_m': PINCH_COMPENSATION_M['left']}


@pytest.mark.parametrize('action', ['release', 'none'])
def test_place_only_never_searches_grasp_pose_or_lift(action, planner, working_start):
    """Releasing a held object needs a place, not a new grasp: even with the arm's real reach probe the line plans
    a position-only descent without a solved grasp pose or a lift."""
    model = planner.models['left']

    def probe(xyz, rotation, seed):
        return planner.pose_error('left', xyz, rotation, seed)
    result = grasp_from_line(SimulationBackend().capture(), [[300, 220], [700, 220]], endpoint_action=action,
                             reach_probe=probe, seed_joints=working_start['left']['joints_deg'],
                             base_xy=model.base_xy, inset_mm=50)
    assert result['grasp_mode'] == 'position_only'
    assert 'lift_xyz' not in result and 'grasp_solver_candidates' not in result
    assert result['arm']['orientation'] == {'mode': 'free'}
    np.testing.assert_allclose(result['arm']['path']['points'][-1], np.array(result['surface_xyz'])+[0, 0, .02])
    assert all(e['s'] == 1 for e in result['arm']['gripper_events'])


def test_single_point_release_accepts_coincident_endpoints(sim_service):
    service = sim_service
    service.observe()
    client = make_client(service)
    body = {'observation_id': service.frame.id, 'pixels': [[380, 220], [380, 220]], 'endpoint_action': 'release'}
    response = client.post('/api/grasp', json=body)
    assert response.status_code == 200, response.text
    assert response.json()['arm']['orientation'] == {'mode': 'free'}
    assert [e['s'] for e in response.json()['arm']['gripper_events']] == [1]
    assert client.post('/api/grasp', json={**body, 'endpoint_action': 'grasp'}).status_code == 409
