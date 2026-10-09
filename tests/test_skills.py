"""The skill registry: one catalogue the GUI renders and an agent calls, over the existing planners."""
import numpy as np
import pytest
from conftest import make_client
from urai.backend import SimulationBackend
from urai.service import Service

#: The tools of this release, as the catalogue lists them.
RELEASED = {'pick_place', 'grasp_line', 'pour', 'carry_line', 'bottle_cap_prepare', 'bottle_cap_twist',
            'bottle_cap_retract', 'bottle_cap_place', 'toss', 'bottle_upright_prepare',
            'bottle_upright_twist', 'bottle_upright_retract', 'bottle_upright_release', 'bottle_body_recenter'}

CUP = (210, 242, 380, 412, .94)      # 60 mm tall at 1 m table depth
BOWL = (300, 332, 500, 532, .97)     # 30 mm tall
CUP_PIXEL = [396, 226]
BOWL_PIXEL = [516, 316]


def scene_service(blocks=(CUP, BOWL)):
    """A service whose observation is a flat table with the given raised blocks."""
    service = Service(SimulationBackend())
    service.observe()
    service.frame.depth[:] = 1.
    for v0, v1, u0, u1, depth in blocks:
        service.frame.depth[v0:v1, u0:u1] = depth
    return service


def client_with_scene(blocks=(CUP, BOWL)):
    service = scene_service(blocks)
    return service, make_client(service)


def stroke(service, pixels, **data):
    # Every drafting endpoint has its own speed default; the skill route takes the console's numbers, so the
    # comparisons below send both explicitly.
    return {'arm': 'left', 'observation_id': service.frame.id, 'pixels': pixels,
            'clearance_m': .08, 'speed_m_s': .10, **data}


def test_catalogue_lists_every_skill_with_its_stroke_arms_and_declared_inputs():
    service, client = client_with_scene()
    listed = client.get('/api/skills').json()['skills']
    names = [skill['name'] for skill in listed]
    assert sorted(names) == sorted(RELEASED)
    for skill in listed:
        assert skill['label'] and skill['group'] and skill['summary']
        assert skill['stroke'] in ('two_points', 'stroke', 'lasso', 'point') and skill['stroke_hint']
        assert skill['arms'] in ('single', 'dual')
        for key, spec in skill['inputs'].items():
            assert spec['type'] in ('number', 'integer', 'choice', 'boolean') and spec['label']
            assert 'default' in spec
            if spec['type'] in ('number', 'integer'):
                assert spec['min'] <= spec['default'] <= spec['max']
    pour = next(skill for skill in listed if skill['name'] == 'pour')
    assert pour['inputs']['tilt_deg']['min'] == 60 and pour['inputs']['tilt_deg']['max'] == 170
    assert [option['value'] for option in next(s for s in listed if s['name'] == 'grasp_line')
            ['inputs']['endpoint_action']['options']] == ['grasp', 'release', 'none']


def test_skill_route_reproduces_the_original_endpoint_exactly():
    """The registry must not become a second implementation: same stroke, same draft."""
    service, client = client_with_scene()
    body = stroke(service, [CUP_PIXEL, [560, 290]])
    original = client.post('/api/pick-place', json=body).json()
    through_registry = client.post('/api/skills/pick_place', json=body).json()
    assert through_registry['arm'] == original['arm']
    assert through_registry['skill'] == 'pick_place' and through_registry['summary']


def test_grasp_line_skill_matches_the_grasp_endpoint_on_the_same_numbers():
    service, client = client_with_scene()
    pixels = [[380, 220], [410, 220]]
    numbers = dict(inset_mm=8, clearance_m=.09, speed_m_s=.03, grip_effort=400)
    original = client.post('/api/grasp', json=stroke(service, pixels, **numbers)).json()
    registry = client.post('/api/skills/grasp_line', json=stroke(service, pixels, **numbers)).json()
    assert registry['arm'] == original['arm'] and registry['width_mm'] == original['width_mm']


def test_pour_skill_matches_the_pour_endpoint():
    service, client = client_with_scene()
    body = stroke(service, [CUP_PIXEL, BOWL_PIXEL], tilt_deg=120, hold_s=1.5)
    original = client.post('/api/pour', json=body)
    registry = client.post('/api/skills/pour', json=body)
    assert original.status_code == registry.status_code
    if original.status_code == 200:
        assert registry.json()['arm'] == original.json()['arm']
        assert registry.json()['tilt_deg'] == 120
    else:
        assert original.json()['detail'] == registry.json()['detail']


def test_numbers_outside_the_declared_range_are_refused_with_the_input_label():
    service, client = client_with_scene()
    response = client.post('/api/skills/pour', json=stroke(service, [CUP_PIXEL, BOWL_PIXEL], tilt_deg=200))
    assert response.status_code == 409 and '倾角' in response.json()['detail']
    response = client.post('/api/skills/grasp_line',
                           json=stroke(service, [[380, 220], [410, 220]], endpoint_action='squeeze'))
    assert response.status_code == 409 and '终点动作' in response.json()['detail']
    assert client.post('/api/skills/no_such_skill', json=stroke(service, [CUP_PIXEL, BOWL_PIXEL])).status_code == 409


def test_commit_saves_the_draft_so_an_agent_can_preview_straight_away():
    service, client = client_with_scene()
    before = service.revision
    response = client.post('/api/skills/pick_place', json=stroke(service, [CUP_PIXEL, [560, 290]], commit=True))
    assert response.status_code == 200, response.json()
    result = response.json()
    assert result['revision'] == before+1 == service.revision
    assert service.draft['arms']['left'] == result['arm']
    assert client.post('/api/preview', json={'expected_revision': result['revision']}).status_code == 200


def test_stroke_kinds_take_only_what_the_skill_draws():
    from urai.skills import stroke_pixels
    frame = SimulationBackend().capture()
    ink = [[10, 10], [20, 20], [30, 40], [60, 80]]
    np.testing.assert_allclose(stroke_pixels(frame, ink, 'two_points'), [[10, 10], [60, 80]])
    np.testing.assert_allclose(stroke_pixels(frame, ink, 'stroke'), ink)
    np.testing.assert_allclose(stroke_pixels(frame, ink, 'point'), [[10, 10]])
    np.testing.assert_allclose(stroke_pixels(frame, ink, 'lasso'), ink)
    with pytest.raises(ValueError, match='终点'):
        stroke_pixels(frame, [[10, 10], [12, 12]], 'two_points')
    with pytest.raises(ValueError, match='区域'):
        stroke_pixels(frame, [[10, 10], [60, 80]], 'lasso')
    with pytest.raises(ValueError, match='画面'):
        stroke_pixels(frame, [[10, 10], [10000, 80]], 'stroke')
    with pytest.raises(ValueError, match='有效'):
        stroke_pixels(frame, [[10, np.nan]], 'point')


def test_a_skill_must_return_a_draft_for_the_arms_it_declares():
    from urai.skills import SKILLS, Skill, plan, skill_context
    from urai.skills.motion import close_event, down_rotation, draft, open_event, rpy_deg
    service = scene_service()
    context = skill_context(service, 'left')

    def both_arms(ctx, ink):
        specs = {}
        for arm in ('left', 'right'):
            here = ctx if ctx.arm == arm else ctx.other_arm()
            xyz = np.asarray(here.current[arm]['xyz'], dtype=float)
            pose = rpy_deg(down_rotation(0.))
            specs[arm] = draft([xyz, xyz+[0, 0, .05]], [[0, *pose], [1, *pose]],
                               [open_event(0.), close_event(1.)], here.speed_m_s, here.clearance_m)
        return {'summary': '双臂', 'arms': specs}

    SKILLS['probe_dual'] = Skill(name='probe_dual', label='双臂检查', group='测试', summary='',
                                 plan=both_arms, arms='dual')
    SKILLS['probe_broken'] = Skill(name='probe_broken', label='坏技能', group='测试', summary='',
                                   plan=lambda ctx, ink: {'summary': '缺草稿'})
    try:
        result = plan('probe_dual', context, {'pixels': [[380, 220], [410, 220]]})
        assert set(result['arms']) == {'left', 'right'} and result['skill'] == 'probe_dual'
        with pytest.raises(ValueError, match='arm 草稿'):
            plan('probe_broken', context, {'pixels': [[380, 220], [410, 220]]})
    finally:
        del SKILLS['probe_dual'], SKILLS['probe_broken']


def test_context_carries_the_arm_calibration_and_refuses_impossible_settings():
    from urai.skills import skill_context
    service = scene_service()
    context = skill_context(service, 'right', clearance_m=.12, speed_m_s=.04)
    assert context.arm == 'right' and context.clearance_m == .12 and context.speed_m_s == .04
    assert context.probe([.3, 0, .2], np.eye(3)) is None      # the simulation has no calibrated model
    assert context.other_arm().arm == 'left'
    assert context.cloud.shape == (*service.frame.depth.shape, 3)
    assert context.cloud is context.other_arm()._cloud        # the levelled cloud belongs to the scene
    for bad in ({'clearance_m': .5}, {'speed_m_s': 2.}):
        with pytest.raises(ValueError):
            skill_context(service, 'left', **bad)


def test_a_tool_event_never_claims_to_hold_anything_and_a_grasp_always_does():
    '''The executor's holding gate only passes a measured opening of 1.5-67 mm, so verifying a closed empty
    gripper would stop the run the moment the tool is formed.'''
    from urai.skills.motion import close_event, tool_event
    tool = tool_event(0.)
    assert 'verify' not in tool and tool['opening_mm'] == 0 and tool['wait_for_arrival'] and tool['ramp_s'] > 0
    assert tool_event(.5, 1.2, 700) == {**tool, 's': .5, 'hold_s': 1.2, 'effort': 700}
    assert close_event(1.)['verify'] == 'holding'
