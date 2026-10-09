"""Joint limits of planned versus measured joints, the feedback geometry guard and the tracking guard.

Geometry checks run on the real PiPER-X arm models of the example rig; execution runs the real backend on the
kinematic simulator (see test_dispatch.py for the plan helpers).
"""
from __future__ import annotations

import time

import numpy as np
import pytest
from test_dispatch import Recorder, drift_deg, joint_plan, run, single_joint

from urai.backend.model_checks import canonical_planned_joints, model_joint_limit_reason
from urai.backend.retiming import joint_curve, tracking_gap

#: The right arm leaning towards the left one; the left arm turning the other way runs into it.
RIGHT_LEANING_IN = np.array([45., 70., -50., 0., 0., 0.])
LEFT_TURNING_IN = np.array([-45., 70., -50., 0., 0., 0.])
#: A left-arm pose whose bias-corrected fingertip is 35 mm below the table and that breaks no other check.
FINGERTIP_BELOW_TABLE = np.array([-46.7, 77.5, -5.8, 11.1, -42.9, -62.])


@pytest.mark.parametrize('index', range(6))
def test_planned_limits_accept_the_bounds_and_reject_targets_beyond_them(arm_models, index):
    model = arm_models['left']
    lower, upper = model.kin.limits_deg[index+1]
    for bound, sign in [(lower, -1), (upper, 1)]:
        q = np.zeros(6)
        q[index] = bound
        saved = q.copy()
        assert model_joint_limit_reason(model, q) is None
        np.testing.assert_array_equal(q, saved)
        q[index] = bound+sign*.001
        reason = model_joint_limit_reason(model, q)
        assert reason and f'j{index+1}=' in reason and '.001' in reason


def test_nonfinite_joints_are_rejected(planner, arm_models):
    q = np.zeros(6)
    q[2] = np.nan
    assert 'finite' in model_joint_limit_reason(arm_models['left'], q)
    with pytest.raises(ValueError, match='finite'):
        planner._geometry_check(q, np.zeros(6), feedback=True)


def test_planned_limits_only_canonicalize_floating_point_roundoff(arm_models):
    model = arm_models['left']
    lower = model.kin.limits_deg[5][0]
    q = np.zeros(6)
    q[4] = np.nextafter(lower, -np.inf)
    saved = q.copy()
    assert canonical_planned_joints(model, q)[4] == lower
    np.testing.assert_array_equal(q, saved)
    q[4] = lower-1e-6
    assert canonical_planned_joints(model, q)[4] == q[4]


@pytest.mark.parametrize('angle', [1.26, .075])
def test_a_measured_pose_beyond_a_nominal_limit_is_accepted_but_a_planned_one_is_not(planner, angle):
    """Servo settling and hand guiding leave measured joints slightly past a bound (j3 reads above 0 at rest)."""
    q = np.zeros(6)
    q[2] = angle
    planner._geometry_check(q, np.zeros(6), feedback=True)
    with pytest.raises(ValueError, match=r'joint limit exceeded \(j3='):
        planner._geometry_check(q, np.zeros(6))


def test_measured_feedback_still_checks_the_table_and_the_other_arm(planner):
    with pytest.raises(ValueError, match='below the table'):
        planner._geometry_check(FINGERTIP_BELOW_TABLE, np.zeros(6), feedback=True)
    with pytest.raises(ValueError, match='would hit the right arm'):
        planner._geometry_check(LEFT_TURNING_IN, RIGHT_LEANING_IN, feedback=True)


def test_every_control_cycle_checks_the_measured_arms_against_each_other(piper_backend):
    """A reference that drives the left arm into the right one is held at the inter-arm margin, before contact."""
    rt = piper_backend.runtime
    park = joint_plan(piper_backend, [0., 2.], np.array([np.zeros(6), RIGHT_LEANING_IN]), arm='right')
    assert run(piper_backend, park)[0]['completed']
    plan = joint_plan(piper_backend, [0., 3.], np.array([np.zeros(6), LEFT_TURNING_IN]))
    recorder = Recorder(rt)
    with pytest.raises(ValueError, match='would hit the right arm'):
        run(piper_backend, plan, recorder=recorder)
    stopped = rt.joints_deg('left')
    assert .2 < stopped[0]/LEFT_TURNING_IN[0] < .9     # stopped part of the way in
    assert drift_deg(rt) < .05


def swing(rate_deg_s, hold_s=1.):
    """Joint 4 swings ``rate_deg_s`` x 0.5 s from rest along a joint-space line, then holds: an overhand throw's shape."""
    times = np.array([0., .5, .5+hold_s])
    knots = np.zeros((3, 6))
    knots[1:, 3] = -rate_deg_s*.5
    return times, knots


def test_tracking_gap_takes_the_nearest_reference_point_within_the_lag_window():
    times, knots = swing(240.)
    curve = joint_curve(times, knots, (1,))
    late = curve(.4-.3)                                 # the arm is where the reference was 0.3 s ago
    assert np.abs(curve(.4)-late).max() > 60
    gap, _ = tracking_gap(curve, late, .4, .5)
    assert gap < 1e-6
    gap, worst = tracking_gap(curve, np.zeros(6), .8, .5)   # stuck at the start of the swing
    assert worst == 3 and gap == pytest.approx(abs(curve(.3)[3]), abs=1e-6)


def test_a_swing_followed_late_but_on_its_path_is_accepted(piper_backend):
    """At full controller speed the simulated joint trails the 160 deg/s swing by about twenty degrees, yet
    stays on it."""
    rt = piper_backend.runtime
    assert piper_backend.tracking_limit_deg == 3.
    times, knots = swing(160.)
    plan = joint_plan(piper_backend, times, knots, stops=(1,), profile='throw')
    curve = plan['arms']['left']['curve']
    result, recorder = run(piper_backend, plan, recorder=Recorder(rt))
    assert result['completed']
    plain = max(np.abs(row.joints-curve(min(row.t, times[-1]))).max() for row in recorder.rows)
    assert plain > 5*piper_backend.tracking_limit_deg    # the plain gap alone would have stopped the swing


def test_a_joint_left_behind_during_a_swing_is_still_refused(piper_backend):
    """At 1% controller speed the joint barely follows; it is refused once the whole lag window is out of reach."""
    rt = piper_backend.runtime
    times, knots = swing(160.)
    plan = joint_plan(piper_backend, times, knots, stops=(1,), controller_percent=1)
    began = time.monotonic()
    with pytest.raises(RuntimeError, match=r'joint tracking error \d+\.\d deg on j4 exceeded 3 deg .*0\.5 s behind'):
        run(piper_backend, plan)
    assert .5 <= time.monotonic()-began <= 1.
    assert drift_deg(rt) < .05


def test_the_tracking_limit_at_rest_is_the_plain_joint_gap(piper_backend):
    rt = piper_backend.runtime
    start = rt.joints_deg('left')
    plan = joint_plan(piper_backend, [0., .2], single_joint(piper_backend, [61., 61.], joint=1))
    began = time.monotonic()
    with pytest.raises(RuntimeError, match='joint tracking error 61.0 deg on j2'):
        run(piper_backend, plan)
    assert time.monotonic()-began < .1                 # refused on the first cycle, before any reference was sent
    np.testing.assert_array_equal(rt.joints_deg('left'), start)
