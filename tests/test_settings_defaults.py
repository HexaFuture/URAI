"""Defaults of a fresh service, the paper-trial settings file and the quick start, checked against docs/safety.md."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from conftest import make_client

from urai.app import apply_settings_file
from urai.routes.meta import QUICK_START
from urai.settings import MOTION_PROFILES

ROOT = Path(__file__).resolve().parents[1]
SAFETY = ROOT/'docs'/'safety.md'
PAPER_SETTINGS = ROOT/'configs'/'paper-settings.json'


def section(title):
    """Body of the docs/safety.md section whose heading contains ``title``."""
    match = re.search(rf'^## [^\n]*{re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)', SAFETY.read_text(), flags=re.MULTILINE | re.DOTALL)
    assert match, f'docs/safety.md has no section "{title}"'
    return match.group(1)


def table_rows(body):
    """Cells of the rows of the first Markdown table in ``body``, without the header and the rule."""
    lines = [line.strip() for line in body.splitlines() if line.strip().startswith('|')]
    return [[cell.strip() for cell in line.strip('|').split('|')] for line in lines[2:]]


def documented_settings():
    """``{setting: (default cell, paper-trial cell)}`` of the safety-relevant defaults table."""
    rows = table_rows(section('Safety-relevant defaults'))
    return {re.match(r'`([a-z_]+)', name).group(1): (default, paper) for name, default, paper in rows}


def profile_text(name):
    limits = MOTION_PROFILES[name]
    return (f"`{name}`: {limits['joint_speed_deg_s']:g} deg/s, {limits['joint_acceleration_deg_s2']:g} deg/s², "
            f"controller speed {limits['controller_speed_percent']} %")


def assert_documented(settings, column):
    """``settings`` (the shape of ``GET /api/settings``) reads as ``column`` (0 default, 1 paper) of the table."""
    cells = {key: pair[column] for key, pair in documented_settings().items()}
    assert set(cells) == {'motion_profile', 'tracking_limit_deg', 'table_checks', 'pose_tolerance', 'auto_prepare'}

    def on(flag):
        return 'on' if flag else 'off'

    assert cells['motion_profile'].startswith(profile_text(settings['motion_profile']))
    assert float(re.match(r'[\d.]+', cells['tracking_limit_deg']).group()) == settings['tracking_limit_deg']
    assert cells['table_checks'].startswith(on(settings['table_checks']))
    tolerance = settings['pose_tolerance']
    assert cells['pose_tolerance'].startswith(f"{tolerance['position_mm']:g} mm / {tolerance['orientation_deg']:g} deg")
    prepare = settings['auto_prepare']
    assert cells['auto_prepare'] == f"{on(prepare['refresh_observation'])}, {on(prepare['restore_can'])}"


@pytest.fixture(params=['sim_service', 'piper_service'])
def service(request):
    """A fresh service on each real backend: the synthetic one and the PiPER-X one on the kinematic simulator."""
    return request.getfixturevalue(request.param)


def test_a_fresh_service_has_the_documented_defaults(service):
    settings = make_client(service).get('/api/settings').json()
    assert settings == service.settings()
    assert_documented(settings, column=0)
    assert settings['motion_profile'] == 'normal' and settings['tracking_limit_deg'] == 3.
    assert settings['table_checks'] is True and service.backend.table_checks is True
    assert settings['pose_tolerance'] == {'position_mm': 10., 'orientation_deg': 3.}
    assert f"homing to {settings['home_speed_m_s']:g} m/s" in SAFETY.read_text()
    assert [profile['profile'] for profile in settings['motion_profiles']] == ['fine', 'normal', 'fast', 'throw']
    assert all(profile == {'profile': name, **MOTION_PROFILES[name]}
               for profile, name in zip(settings['motion_profiles'], MOTION_PROFILES))


def test_the_removed_unlimited_profile_is_refused(service):
    before = service.settings()
    response = make_client(service).put('/api/settings', json={'motion_profile': 'unlimited'})
    assert response.status_code == 409
    assert service.settings() == before


def test_the_paper_settings_file_applies_and_matches_the_safety_table(service):
    expected = {key: value for key, value in json.loads(PAPER_SETTINGS.read_text()).items() if not key.startswith('_')}
    applied = apply_settings_file(service, PAPER_SETTINGS)
    assert applied == service.settings()
    assert {key: applied[key] for key in expected} == expected
    assert_documented(applied, column=1)
    assert service.backend.table_checks is False and service.backend.tracking_limit_deg == 0.
    assert service.motion_profile == 'throw'


def test_the_paper_settings_body_is_what_put_settings_accepts(sim_service):
    body = {key: value for key, value in json.loads(PAPER_SETTINGS.read_text()).items() if not key.startswith('_')}
    response = make_client(sim_service).put('/api/settings', json=body)
    assert response.status_code == 200
    assert {key: response.json()[key] for key in body} == body


def test_the_quick_start_lists_the_same_robot_commands_as_the_safety_page():
    rows = {re.fullmatch(r'`([A-Z]+ /api/[^`]+)`', row[0]).group(1): row[1]
            for row in table_rows(section('What moves the robot'))}
    motion = QUICK_START['motion']
    moving = set(motion['moves_the_robot'])
    commanding = set(motion['may_send_robot_commands'])
    assert moving | commanding <= set(rows)
    # The one documented row that commands nothing itself changes the limits of later motions.
    assert set(rows) - moving - commanding == {'PUT /api/settings'}
    assert 'later motions' in rows['PUT /api/settings']
    assert 'Sends CAN commands' in rows['POST /api/preview'] and 'POST /api/preview' in commanding
    # The drafting endpoints the safety page promises never command the robot are not among them.
    drafting = re.search(r'Drafting requests \(([^)]*)\) never', section('What moves the robot'), flags=re.DOTALL).group(1)
    paths = re.findall(r'`(?:[A-Z]+ )?(/api/[^`]+)`', drafting)
    assert {'/api/skills/<name>', '/api/draft'} <= set(paths)
    for path in paths:
        assert not any(path in endpoint for endpoint in moving | commanding), path
