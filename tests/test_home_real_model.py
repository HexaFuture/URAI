"""plan_home on the real PiPER-X models: a joint-space line to the all-zero pose through every planner check.

Nothing is executed. The start poses are joint angles measured on a dual-arm rig whose bases stand 0.59 m apart,
as in the bundled example calibration.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.backend import TaskError
from urai.robot.arm_model import inter_arm_reason, joint_limit_reason
from urai.settings import motion_limits

MEASURED_START = {'left': [-34.26, 148.06, -119.95, 13.74, 2.71, 20.6],
                  'right': [21.6, 102.27, -64.78, 57.42, 0., 29.09]}
#: The right arm reaching in over the left arm's homing path (TCP near (0.50, -0.25, 0.25) m).
RIGHT_IN_THE_WAY = [31.9, 102.6, -79.5, -.5, -7.9, 0.]


def arm_state(model, joints, gripper_mm=0.):
    pose = model.fk_tcp_world(np.asarray(joints, dtype=float))
    return {'joints_deg': list(joints), 'xyz': pose[:3, 3].tolist(),
            'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(), 'gripper_mm': gripper_mm}


def measured_start(models):
    return {arm: arm_state(models[arm], joints) for arm, joints in MEASURED_START.items()}


def assert_valid_home_plan(plan, models, start, arms, profile):
    limits = motion_limits(profile)
    for arm in arms:
        p = plan['arms'][arm]
        q0 = np.asarray(start[arm]['joints_deg'], dtype=float)
        # Joint-space straight line after the opening dwell: no IK anywhere on the way.
        np.testing.assert_array_equal(p['joints_deg'][0], q0)
        np.testing.assert_array_equal(p['joints_deg'][-1], np.zeros(6))
        fractions = 1-p['joints_deg'][1:-1, 0]/q0[0]
        np.testing.assert_allclose(p['joints_deg'][1:-1], q0[None, :]*(1-fractions)[:, None], atol=1e-9)
        if 'recovery_joint_bounds' not in p:
            assert all(joint_limit_reason(q) is None for q in p['joints_deg'])
        times = np.linspace(p['time_s'][0], p['time_s'][-1], 2000)
        assert np.abs(p['curve'](times, 1)).max() <= limits['joint_speed_deg_s'] * 1.001
        assert np.abs(p['curve'](times, 2)).max() <= limits['joint_acceleration_deg_s2'] * 1.001
        np.testing.assert_allclose(p['curve'](p['time_s'][-1]), 0, atol=1e-6)
        assert models[arm].floor_check(np.zeros(6)) is None
        [event] = p['gripper_events']
        assert event['opening_mm'] == 70 and 'verify' not in event and event['wait_for_arrival']
        assert event['hold_end_time_s'] - event['time_s'] >= .6 - 1e-9
    assert 0 < plan['duration_s'] <= 180
    assert plan['approach_duration_s'] == 0


@pytest.mark.parametrize('profile', ['fine', 'normal', 'fast'])
def test_real_model_plans_both_arms_home_from_a_measured_pose(planner, profile):
    start = measured_start(planner.models)
    plan = planner.plan_home(start, ['left', 'right'], motion_profile=profile)
    assert_valid_home_plan(plan, planner.models, start, ['left', 'right'], profile)
    assert plan['motion_limits']['profile'] == profile
    assert plan['approach_routes'] == {'left': 'home', 'right': 'home'}


def test_real_model_keeps_the_unselected_arm_still_and_checked(planner):
    models = planner.models
    start = measured_start(models)
    plan = planner.plan_home(start, ['left'])
    assert_valid_home_plan(plan, models, start, ['left'], 'normal')
    assert 'right' not in plan['arms']
    assert '左臂' in plan['notes'][0] and '右臂' not in plan['notes'][0]
    # The unselected arm stays where it is and still takes part in the inter-arm check of every sample.
    start['right'] = arm_state(models['right'], RIGHT_IN_THE_WAY)
    assert inter_arm_reason(models['left'], np.asarray(MEASURED_START['left']), models['right'], np.asarray(RIGHT_IN_THE_WAY)) is None
    with pytest.raises(TaskError, match='would hit the right arm'):
        planner.plan_home(start, ['left'])


def test_real_model_home_recovers_from_a_pose_just_outside_a_nominal_bound(planner):
    # A toss ended with the right j4 planned exactly on its 89 deg bound and measured 0.064 deg past it.
    models = planner.models
    start = measured_start(models)
    joints = [33.931, 95.337, -90.789, 89.064, 0., -21.059]
    assert joint_limit_reason(np.asarray(joints)) is not None
    start['right'] = arm_state(models['right'], joints, gripper_mm=70.)
    plan = planner.plan_home(start, ['right'])
    p = plan['arms']['right']
    np.testing.assert_allclose(p['joints_deg'][0], joints)
    np.testing.assert_array_equal(p['joints_deg'][-1], np.zeros(6))
    outside = np.maximum(p['joints_deg'][:, 3]-models['right'].kin.limits_deg[4][1], 0.)
    assert np.all(np.diff(outside) <= 1e-9) and outside[2:].max() == 0.
    assert any('从实测关节位置平滑返回' in note for note in plan['notes'])
    assert 'right' in plan['initial_joint_bounds']
