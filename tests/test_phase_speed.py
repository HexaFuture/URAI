"""Stage speed controls (approach, carry, wind-up, grasp hold, return home) reach the plans that use them."""
from __future__ import annotations

import json

import numpy as np
import pytest

from urai.backend import SimulationBackend
from urai.service import Service
from urai.settings import motion_limits
from urai.toss import overhand_draft
from urai.transfer import place_transfer

QUEUE_OPTIONS = {'toss_style': 'sidearm', 'clearance_m': .12, 'speed_m_s': .15, 'toss_speed_m_s': 1.5,
                 'toss_lead_s': .08, 'approach_speed_m_s': .18, 'carry_speed_m_s': .7, 'windup_speed_m_s': .21,
                 'grasp_hold_s': .6, 'home_speed_m_s': .8}


def test_the_queue_stage_speeds_reach_the_toss_draft_and_the_executed_task(piper_backend, tmp_path):
    """A toss item of the task queue is drafted, planned and executed by the real stack on the simulator; the run
    record shows the options in the draft and the return-home speed in the task. How the throw ends is not checked
    here."""
    service = Service(piper_backend, tmp_path)
    service.set_settings({'motion_profile': 'throw'})
    service.observe()
    block = piper_backend.runtime.cameras.scene.pose('red_block')[:2, 3]
    service.tasks.submit({'observation_id': service.frame.id, 'items': [
        {'object_xy': block.tolist(), 'arm': 'left', 'placement': 'toss', 'landing_xyz': [.55, .25, .0035]}],
        **QUEUE_OPTIONS})
    service.tasks.thread.join(60)
    assert not service.tasks.status()['active']
    [item] = service.tasks.status()['items']
    record = json.loads((tmp_path / f"{item['preview_id']}.json").read_text())
    arm = record['preview']['input_draft']['arms']['left']
    assert arm['approach']['speed_m_s'] == QUEUE_OPTIONS['approach_speed_m_s']
    speeds = [speed for _, speed in arm['speed']]
    assert QUEUE_OPTIONS['carry_speed_m_s'] in speeds and QUEUE_OPTIONS['windup_speed_m_s'] in speeds
    assert [event['hold_s'] for event in arm['gripper_events'] if event.get('verify') == 'holding'] == \
        [QUEUE_OPTIONS['grasp_hold_s']]
    assert record['task']['home_speed_m_s'] == QUEUE_OPTIONS['home_speed_m_s']


def test_the_home_speed_setting_is_validated_and_reported():
    service = Service(SimulationBackend())
    for bad in [0, -1, 1.1, float('nan'), True]:
        with pytest.raises(ValueError):
            service.set_settings({'home_speed_m_s': bad})
    service.set_settings({'home_speed_m_s': .7})
    assert service.settings()['home_speed_m_s'] == .7 and service.status()['home_speed_m_s'] == .7


def test_the_home_speed_setting_reaches_the_automatic_return_home(piper_backend):
    service = Service(piper_backend)
    service.set_settings({'home_speed_m_s': .7})
    service.observe()
    xyz = np.asarray(piper_backend.state()['left']['xyz'])
    stroke = {'mode': 'waypoints', 'points': [xyz.tolist(), (xyz+[.03, 0., 0.]).tolist()]}
    service.set_draft({'observation_id': service.frame.id, 'arms': {'left': {'path': stroke}}})
    service.execute(service.preview()['id'])
    service.worker.join(60)
    assert service.execution['state'] == 'completed'
    home = service.execution['result']['return_home_plan']['arms']['left']
    assert home['approach_config']['speed_m_s'] == .7 and max(home['requested_speed_m_s']) == .7
    assert service.execution['result']['return_home']['completed']
    np.testing.assert_allclose(piper_backend.runtime.joints_deg('left'), 0., atol=.7)


def test_a_faster_home_speed_shortens_the_home_within_the_joint_limits(planner, working_start):
    slow = planner.plan_home(working_start, ['left'], motion_profile='throw', open_grippers=False, speed_m_s=.2)
    fast = planner.plan_home(working_start, ['left'], motion_profile='throw', open_grippers=False, speed_m_s=.6)
    assert fast['duration_s'] < slow['duration_s']
    p = fast['arms']['left']
    t = np.linspace(p['time_s'][0], p['time_s'][-1], 2000)
    limits = motion_limits('throw')
    assert np.abs(p['curve'](t, 1)).max() <= limits['joint_speed_deg_s']*1.001
    assert np.abs(p['curve'](t, 2)).max() <= limits['joint_acceleration_deg_s2']*1.001


def overhand(planner, start, **stage):
    """The overhand throw draft for a 4 cm block in front of the left arm, landing 0.7 m ahead of its base."""
    model = planner.models['left']
    base = np.asarray(model.base_xy, dtype=float)
    table_z = model.table_z_mm/1000
    center = base+[.40, .05]
    corners = np.array([[-.02, -.02], [.02, -.02], [.02, .02], [-.02, .02]])
    found = {'center': center, 'top': table_z+.02, 'width': .04, 'axis_deg': 0.,
             'points': np.c_[center+corners, np.full(4, table_z+.02)]}
    target = np.r_[base+[.7, .1], table_z]
    placement = place_transfer(found, target, table_z, .08, model.fingertip_bias_mm, drop=True)

    def probe(xyz, rotation, seed):
        return planner.pose_error('left', xyz, rotation, seed, {'position_mm': 30., 'orientation_deg': 60.})
    return overhand_draft(found, placement, target, table_z, .08, .10, 0., 5., .08, base, model.fk_tcp_world, probe,
                          np.asarray(start['left']['joints_deg']), (30., 60.), **stage)


def test_overhand_stage_controls_change_timing_but_keep_the_throw_geometry(planner, working_start):
    default = overhand(planner, working_start)
    staged = overhand(planner, working_start, carry_speed_m_s=.6, windup_speed_m_s=.15, grasp_hold_s=.5)
    np.testing.assert_allclose(staged['arm']['path']['points'], default['arm']['path']['points'])
    grip = staged['arm']['gripper_events'][1]
    assert grip['hold_s'] == .5 and grip['verify'] == 'holding'
    speeds = [speed for _, speed in staged['arm']['speed']]
    assert .6 in speeds and .15 in speeds
    assert staged['toss_speed_m_s'] == default['toss_speed_m_s']
