"""Every skill takes a travel speed and an approach speed; both reach the draft it returns."""
import pytest
from test_skills import CUP_PIXEL, client_with_scene, stroke
from urai.skills import describe_all

#: (speed_m_s, approach_speed_m_s) defaults: the registry's 0.8 m/s, or a skill's own slower declaration.
SPEED_DEFAULTS = {'bottle_cap_prepare': (.10, .10), 'bottle_cap_twist': (.08, .08), 'bottle_cap_place': (.08, .04)}


def test_default_profile_and_catalogue_speeds():
    service, client = client_with_scene()
    assert service.motion_profile == 'normal'
    assert client.get('/api/state').json()['motion_profile'] == 'normal'
    for skill in describe_all():
        speed, approach = SPEED_DEFAULTS.get(skill['name'], (.8, .8))
        assert skill['inputs']['speed_m_s']['default'] == speed, skill['name']
        assert skill['inputs']['approach_speed_m_s']['default'] == approach, skill['name']


def test_speed_reaches_generated_draft():
    service, client = client_with_scene()
    result = client.post('/api/skills/pick_place', json=stroke(service, [CUP_PIXEL, [560, 290]], speed_m_s=.08,
                                                                approach_speed_m_s=.025))
    assert result.status_code == 200, result.text
    arm = result.json()['arm']
    assert arm['speed'] == .08
    assert arm['approach']['speed_m_s'] == .025


@pytest.mark.parametrize('value', [0, -1, 1.01, 'invalid'])
def test_invalid_approach_rejected(value):
    service, client = client_with_scene()
    response = client.post('/api/skills/pick_place', json=stroke(service, [CUP_PIXEL, [560, 290]], approach_speed_m_s=value))
    assert response.status_code == 409 and '接近速度' in response.json()['detail']
    assert service.execution['state'] == 'idle'
