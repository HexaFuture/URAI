"""The vectorized retiming checks reproduce the per-interval np.roots and per-sample Jacobian references bit for bit."""
import numpy as np
import pytest

from urai.backend.retiming import (
    angular_interval_scales,
    interval_retiming_scales,
    joint_axes,
    joint_curve,
    polynomial_peaks,
)
from urai.settings import motion_limits
from urai.trajectory import MAX_ANGULAR_SPEED


def reference_interval_scales(curve, limits):
    """Per-interval loop: np.roots for every interval, joint and derivative order."""
    scales = np.ones(len(curve.x) - 1)
    for order, key in [(1, 'joint_speed_deg_s'), (2, 'joint_acceleration_deg_s2'), (3, 'joint_jerk_deg_s3')]:
        derivative = curve.derivative(order)
        for i, h in enumerate(np.diff(curve.x)):
            peak = 0.
            for j in range(derivative.c.shape[-1]):
                coefficients = derivative.c[:, i, j]
                roots = np.roots(np.polyder(coefficients))
                roots = roots.real[(np.abs(roots.imag) < 1e-8) & (roots.real > 0) & (roots.real < h)]
                peak = max(peak, float(np.max(np.abs(np.polyval(coefficients, np.r_[0., h, roots])))))
            scales[i] = max(scales[i], (peak / limits[key]) ** (1 / order))
    return scales


def reference_angular_scales(model, curve):
    """Per-sample loop: one spatial Jacobian evaluation per interior sample."""
    times = (curve.x[:-1, None] + np.diff(curve.x)[:, None] * np.linspace(0, 1, 9)).ravel()
    speeds = []
    for q, v in zip(curve(times), curve(times, 1)):
        axes = model.kin.fk_with_spatial_jacobian(q)[2]
        speeds.append(float(np.linalg.norm(axes @ v)))
    return np.maximum(1., np.asarray(speeds).reshape(-1, 9).max(axis=1) / MAX_ANGULAR_SPEED)


def random_curve(seed, knots=120, joints=6, amplitude=3.):
    rng = np.random.default_rng(seed)
    times = np.r_[0., np.cumsum(rng.uniform(.02, .3, knots - 1))]
    qs = np.cumsum(rng.normal(0., amplitude, (knots, joints)), axis=0)
    qs[40:44] = qs[40]  # a held pose: constant intervals carry all-zero derivative polynomials
    qs[:, 2] = 0.  # a joint that never moves
    return joint_curve(times, qs, stop_indices=[60])


@pytest.mark.parametrize('seed', range(4))
def test_vectorized_interval_scales_match_the_per_interval_reference_bitwise(seed):
    curve = random_curve(seed)
    for profile in ('fine', 'normal', 'fast', 'throw'):
        limits = motion_limits(profile)
        np.testing.assert_array_equal(interval_retiming_scales(curve, limits), reference_interval_scales(curve, limits))


def test_polynomial_peaks_follow_np_roots_for_degenerate_derivatives():
    coefficients = np.zeros((5, 4, 2))  # degree-4 polynomials on 4 intervals for 2 joints
    coefficients[:, 0, 0] = [1., -2., 0., 3., 1.]  # generic derivative
    coefficients[:, 1, 0] = [0., 0., 1., 0., 0.]  # leading zeros in the derivative
    coefficients[:, 2, 0] = [1., 0., 0., 0., 0.]  # trailing zeros in the derivative
    coefficients[:, 3, 1] = [0., 0., 0., 0., 2.]  # constant polynomial
    spans = np.array([1., .5, 2., .1])
    expected = np.zeros((4, 2))
    for i in range(4):
        for j in range(2):
            polynomial = coefficients[:, i, j]
            roots = np.roots(np.polyder(polynomial))
            roots = roots.real[(np.abs(roots.imag) < 1e-8) & (roots.real > 0) & (roots.real < spans[i])]
            expected[i, j] = np.max(np.abs(np.polyval(polynomial, np.r_[0., spans[i], roots])))
    np.testing.assert_array_equal(polynomial_peaks(coefficients, spans), expected)


def test_batched_joint_axes_match_the_spatial_jacobian_bitwise(arm_models):
    model = arm_models['left']
    lower = np.array([model.kin.limits_deg[i][0] for i in range(1, 7)])
    upper = np.array([model.kin.limits_deg[i][1] for i in range(1, 7)])
    qs = np.random.default_rng(3).uniform(lower, upper, size=(400, 6))
    expected = np.stack([model.kin.fk_with_spatial_jacobian(q)[2] for q in qs])
    np.testing.assert_array_equal(joint_axes(model.kin, qs), expected)


@pytest.mark.parametrize('seed', range(3))
def test_batched_angular_scales_match_the_per_sample_reference_bitwise(seed, arm_models):
    model = arm_models['left']
    curve = random_curve(seed, knots=80, amplitude=1.)
    scales = angular_interval_scales(model, curve)
    np.testing.assert_array_equal(scales, reference_angular_scales(model, curve))
    assert scales.max() > 1.  # the random walk does turn the tool faster than the limit somewhere
