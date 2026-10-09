"""Shape of the joint reference: C2 continuity, rest endpoints, bounded knots and retimed derivative limits."""
import numpy as np
import pytest

from urai.backend.retiming import joint_curve, retiming_scale
from urai.settings import motion_limits


@pytest.mark.parametrize('profile', ['fine', 'normal'])
def test_retiming_checks_both_sides_of_cubic_knots(profile):
    times = np.array([0, .64486993, .74418625, 1.8338107, 2.93174761, 4.0819498])
    q = np.array([15.41068426, 32.08395792, 34.18785381, 33.8287513, 50.42467696, 37.37771411])[:, None]
    limits = motion_limits(profile)
    scale = retiming_scale(joint_curve(times, q), limits)
    assert scale > 1.02
    limited = joint_curve(times*scale, q)
    grid = np.unique(np.r_[np.linspace(0, times[-1]*scale, 10000), times[1:]*scale-1e-10])
    assert np.abs(limited(grid, 1)).max() <= limits['joint_speed_deg_s']+1e-8
    assert np.abs(limited(grid, 2)).max() <= limits['joint_acceleration_deg_s2']+1e-8
    assert np.abs(limited(grid, 3)).max() <= limits['joint_jerk_deg_s3']+1e-8


def test_joint_reference_has_continuous_acceleration_and_rest_endpoints():
    t = np.array([0., 1., 1.4, 3.])
    q = np.array([0., 4., 3., 5.])[:, None]
    curve = joint_curve(t, q)
    np.testing.assert_allclose(curve(t[[0, -1]], 1), 0., atol=1e-8)
    np.testing.assert_allclose(curve(t[[0, -1]], 2), 0., atol=1e-8)
    np.testing.assert_allclose(curve(t[1:-1]-1e-9, 2), curve(t[1:-1]+1e-9, 2), atol=1e-5)


def test_short_dual_arm_wait_has_exact_zero_derivatives():
    t = np.array([0., 1., 1.+1e-7, 2.])
    q = np.array([30.015, 32.015, 32.015, 33.015])[:, None]
    c = joint_curve(t, q, (1, 2))
    for order in (1, 2, 3):
        np.testing.assert_allclose(c((t[1]+t[2])/2, order), 0., atol=1e-8)
    # A 0.1 us wait between two stops must not inflate the retiming factor.
    assert retiming_scale(c, motion_limits('fine')) < 2


def test_dense_smooth_motion_does_not_stop_acceleration_at_every_sample():
    t = np.linspace(0, 4, 201)
    q = (2*t*t)[:, None]
    c = joint_curve(t, q)
    np.testing.assert_allclose(c(t[30:-30], 2), 4., atol=1e-5)


def test_smoothing_keeps_joint_bounds_and_continuous_acceleration():
    t = np.array([0, .1, .3, .6, 1., 1.4])
    q = np.array([-.2, -.01, -.005, -.001, -.01, -.2])[:, None]
    c = joint_curve(t, q, joint_limits=np.array([[-1., 0.]]))
    grid = np.linspace(0, t[-1], 4000)
    assert c(grid).max() <= 1e-10 and c(grid).min() >= -1-1e-10
    np.testing.assert_allclose(c(t[1:-1]-1e-9, 2), c(t[1:-1]+1e-9, 2), atol=1e-3)


def test_out_of_bounds_knots_are_refused():
    t = np.array([0., 1., 2.])
    q = np.array([0., .5, 1.5])[:, None]
    with pytest.raises(ValueError, match='out-of-bounds knot'):
        joint_curve(t, q, joint_limits=np.array([[-1., 1.]]))


def test_uneven_sample_times_cannot_turn_a_short_monotone_path_into_a_swing():
    t = np.array([0., 1., 1.01, 2.01])
    q = np.arange(4.)[:, None]
    c = joint_curve(t, q, joint_limits=np.array([[-150., 150.]]))
    for i in range(3):
        actual = c(np.linspace(t[i], t[i+1], 500))
        assert actual.min() >= q[i, 0]-1e-9
        assert actual.max() <= q[i+1, 0]+1e-9
