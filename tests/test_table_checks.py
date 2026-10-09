"""The table checks and the tracking limit are explicit settings, on by default.

Geometry runs on the real PiPER-X arm models of the example rig; the settings go through the real HTTP app; the
tracking guard runs in the real executor on the kinematic simulator.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from test_dispatch import joint_plan, run, single_joint, state_from_joints

from urai.backend import SimulationBackend
from urai.backend.model_checks import solve_ik
from urai.service import Service
from urai.trajectory import compile_arms

#: Left-arm pose whose bias-corrected fingertip is 35 mm below the table and that breaks no other check.
FINGERTIP_BELOW_TABLE = np.array([-46.7, 77.5, -5.8, 11.1, -42.9, -62.])
#: The tool 9 mm from the arm's own base column: inside the capsule model's 15 mm margin, but no contact.
TOOL_NEAR_BASE = np.array([-41.6, 37.1, -.6, 29.8, -75.8, 91.5])
#: The tool body inside the arm's own base column.
TOOL_IN_BASE = np.array([34.6, 69.1, -.5, 85.6, 33., 36.1])


def test_table_checks_are_on_by_default_and_skip_the_floor_when_switched_off(planner):
    assert planner.table_checks is True
    for feedback in (False, True):
        with pytest.raises(ValueError, match=r'^left: left fingertip z=-34\.8mm \(bias-corrected\) is below'):
            planner._geometry_check(FINGERTIP_BELOW_TABLE, np.zeros(6), feedback=feedback)
    planner.table_checks = False
    planner._geometry_check(FINGERTIP_BELOW_TABLE, np.zeros(6))
    planner._geometry_check(FINGERTIP_BELOW_TABLE, np.zeros(6), feedback=True)


def test_self_collision_refuses_interpenetration_but_not_the_capsule_margin(planner, arm_models):
    clearance, mover, target = arm_models['left'].self_collision_clearance(TOOL_NEAR_BASE)
    assert 0. < clearance < 15. and (mover, target) == ('tool body', 'base column')
    planner._geometry_check(TOOL_NEAR_BASE, np.zeros(6))
    with pytest.raises(ValueError, match=r'^left: left self-collision: tool body vs base column \(clearance -57 mm\)'):
        planner._geometry_check(TOOL_IN_BASE, np.zeros(6))
    with pytest.raises(ValueError, match=r'^right: right self-collision: tool body vs base column'):
        planner._geometry_check(np.zeros(6), TOOL_IN_BASE)


def start_at(models, arm, xyz, rpy_deg):
    """Planner start state with ``arm`` holding the TCP pose ``xyz``/``rpy_deg`` and the other arm at rest."""
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
    target[:3, 3] = xyz
    joints = {a: np.zeros(6) for a in models}
    joints[arm] = solve_ik(models[arm], target, np.array([0., 60., -60., 0., 40., 0.]))[0]
    return state_from_joints(models, joints)


def test_the_planner_keeps_samples_below_the_floor_when_table_checks_are_off(planner, arm_models):
    planner.table_checks = False
    start = start_at(arm_models, 'left', [.3, 0., .05], [180., 0., -90.])
    spec = {'path': {'mode': 'waypoints', 'points': [[.3, 0., .05], [.3, 0., -.0055]]}, 'speed': .05,
            'orientation': {'mode': 'keyframes', 'points': [[0, 180, 0, -90], [1, 180, 0, -90]]}}
    plan = planner.plan(compile_arms({'left': spec}, start), start)
    assert plan['arms']['left']['xyz'][:, 2].min() == pytest.approx(-.0055)
    assert not any('贴近桌面' in note for note in plan['notes'])
    assert any('已关闭' in note for note in plan['notes'])


def test_settings_toggle_table_checks_and_reject_non_booleans(client_for):
    service = Service(SimulationBackend())
    service.observe()
    client = client_for(service)
    assert client.get('/api/settings').json()['table_checks'] is True
    assert client.get('/api/state').json()['table_checks'] is True
    assert client.put('/api/settings', json={'table_checks': 'yes'}).status_code == 409
    saved = client.put('/api/settings', json={'table_checks': False})
    assert saved.status_code == 200 and saved.json()['table_checks'] is False
    assert service.backend.table_checks is False
    assert client.get('/api/state').json()['backend']['table_checks'] is False
    assert client.put('/api/settings', json={'motion_profile': 'fine'}).json()['table_checks'] is False


def test_the_tracking_limit_setting_reaches_the_executor_and_zero_switches_it_off(piper_backend, client_for):
    """The default 3 degree guard refuses a reference 4 degrees away from the measured joints; 6 and 0 (off) do not."""
    client = client_for(Service(piper_backend))
    assert client.get('/api/settings').json()['tracking_limit_deg'] == 3.
    assert client.put('/api/settings', json={'tracking_limit_deg': -1}).status_code == 409
    assert client.put('/api/settings', json={'tracking_limit_deg': 95}).status_code == 409

    def offset_plan():
        return joint_plan(piper_backend, [0., .3], single_joint(piper_backend, [4., 4.]))
    with pytest.raises(RuntimeError, match='4.0 deg on j1 exceeded 3 deg'):
        run(piper_backend, offset_plan())
    assert client.put('/api/settings', json={'tracking_limit_deg': 6}).json()['tracking_limit_deg'] == 6.
    assert piper_backend.tracking_limit_deg == 6.
    assert run(piper_backend, offset_plan())[0]['completed']
    assert client.put('/api/settings', json={'tracking_limit_deg': 0}).json()['tracking_limit_deg'] == 0.
    assert run(piper_backend, offset_plan())[0]['completed']
