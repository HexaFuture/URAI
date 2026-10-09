"""Joint-space interpolation and time scaling shared by planning and execution.

Every function here is a pure computation on numpy arrays and ``scipy`` piecewise polynomials.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BPoly, CubicSpline, PPoly
from scipy.spatial.transform import Rotation
from ..trajectory import MAX_ANGULAR_SPEED


def joint_curve(times, qs, stop_indices=(), joint_limits=None):
    """C2 joint interpolant through ``qs`` at ``times``, at rest at the stop indices and bounded between knots."""
    stops = sorted(set([0, len(times)-1, *stop_indices]))
    slopes = np.zeros_like(qs); accelerations = np.zeros_like(qs)
    for lo,hi in zip(stops[:-1],stops[1:]):
        spline = CubicSpline(times[lo:hi+1],qs[lo:hi+1],bc_type=((1,np.zeros(qs.shape[1])),(1,np.zeros(qs.shape[1]))))
        slopes[lo:hi+1] = spline(times[lo:hi+1],1)
        accelerations[lo:hi+1] = spline(times[lo:hi+1],2)
    flat = qs[1:] == qs[:-1]
    fixed = np.zeros_like(qs,dtype=bool);fixed[:-1] |= flat;fixed[1:] |= flat;fixed[stops] = True
    slopes[fixed] = 0.;accelerations[fixed] = 0.
    # A smooth interpolant must also stay near its input path. Keep each
    # interval's Bernstein control points inside its endpoint component box;
    # the convex hull property prevents large inter-knot excursions.
    low=np.minimum(qs[:-1],qs[1:]);high=np.maximum(qs[:-1],qs[1:])
    if joint_limits is not None:
        lower,upper=np.asarray(joint_limits).T
        if np.any(qs<lower-1e-9) or np.any(qs>upper+1e-9):
            raise ValueError('Joint trajectory contains an out-of-bounds knot')
        low=np.maximum(low,lower);high=np.minimum(high,upper)
    h=np.diff(times)[:,None]
    vlo=np.full_like(qs,-np.inf);vhi=np.full_like(qs,np.inf)
    vlo[:-1]=np.maximum(vlo[:-1],(low-qs[:-1])*2.5/h);vhi[:-1]=np.minimum(vhi[:-1],(high-qs[:-1])*2.5/h)
    vlo[1:]=np.maximum(vlo[1:],(qs[1:]-high)*2.5/h);vhi[1:]=np.minimum(vhi[1:],(qs[1:]-low)*2.5/h)
    slopes=np.clip(slopes,vlo,vhi);slopes[fixed]=0.
    alo=np.full_like(qs,-np.inf);ahi=np.full_like(qs,np.inf)
    for part,base in [(slice(None,-1),qs[:-1]+2*h*slopes[:-1]/5),(slice(1,None),qs[1:]-2*h*slopes[1:]/5)]:
        alo[part]=np.maximum(alo[part],20*(low-base)/h**2)
        ahi[part]=np.minimum(ahi[part],20*(high-base)/h**2)
    accelerations=np.clip(accelerations,alo,ahi);accelerations[fixed]=0.
    derivatives = [[q,v,a] for q,v,a in zip(qs,slopes,accelerations)]
    curve = PPoly.from_bernstein_basis(BPoly.from_derivatives(times, derivatives))
    # Equal poses must hold exactly. Basis conversion on extremely short wait
    # intervals otherwise amplifies floating-point cancellation into spurious jerk.
    constant = (qs[1:] == qs[:-1]) & (slopes[1:] == 0) & (slopes[:-1] == 0)
    curve.c[:-1] = np.where(constant[None,:,:], 0., curve.c[:-1])
    curve.c[-1] = np.where(constant, qs[:-1], curve.c[-1])
    return curve


def interval_retiming_scales(curve, limits):
    """Polynomial derivative extrema per interval, including both sides of knots."""
    spans = np.diff(curve.x)
    scales = np.ones(len(spans))
    for order, key in [(1, 'joint_speed_deg_s'), (2, 'joint_acceleration_deg_s2'), (3, 'joint_jerk_deg_s3')]:
        peaks = polynomial_peaks(curve.derivative(order).c, spans).max(axis=1)
        # Python float arithmetic keeps these scales identical to the historical per-interval loop.
        scales = np.maximum(scales, [(float(peak) / limits[key]) ** (1 / order) for peak in peaks])
    return scales


def polynomial_peaks(coefficients, spans):
    """Largest |p(t)| on [0, h] for every polynomial of a (degree+1, intervals, joints) PPoly block.

    Interior extrema are the real roots of p' inside (0, h). Derivatives with a non-zero leading
    and trailing coefficient take their companion-matrix eigenvalues in one batch, which is
    exactly the computation np.roots performs for a single polynomial; the remaining ones
    (all zero, leading or trailing zeros) go through np.roots itself. Every peak therefore
    equals the value a per-polynomial evaluation gives.
    """
    rows, intervals, joints = coefficients.shape
    flat = coefficients.reshape(rows, intervals * joints)
    h = np.repeat(spans, joints)

    def evaluate(points, columns=slice(None)):
        values = np.zeros_like(points)
        for row in flat[:, columns]:  # Horner's scheme, like np.polyval
            values = values * points + row[:, None]
        return values

    peaks = np.abs(evaluate(np.stack([np.zeros_like(h), h], axis=1))).max(axis=1)
    degree = rows - 2  # of p'
    if degree >= 1:
        derivative = (flat[:-1] * np.arange(rows - 1, 0, -1)[:, None]).T  # np.polyder order
        generic = (derivative[:, 0] != 0) & (derivative[:, -1] != 0)
        if generic.any():
            leading = derivative[generic]
            companion = np.zeros((len(leading), degree, degree))
            companion[:, 0, :] = -leading[:, 1:] / leading[:, :1]
            if degree > 1:
                companion[:, np.arange(1, degree), np.arange(degree - 1)] = 1.
            roots = np.linalg.eigvals(companion)
            inside = (np.abs(roots.imag) < 1e-8) & (roots.real > 0) & (roots.real < h[generic][:, None])
            values = np.where(inside, np.abs(evaluate(roots.real, generic)), 0.)
            peaks[generic] = np.maximum(peaks[generic], values.max(axis=1))
        for column in np.flatnonzero(~generic):
            roots = np.roots(derivative[column])
            roots = roots.real[(np.abs(roots.imag) < 1e-8) & (roots.real > 0) & (roots.real < h[column])]
            if len(roots):
                peaks[column] = max(peaks[column], float(np.max(np.abs(np.polyval(flat[:, column], roots)))))
    return peaks.reshape(intervals, joints)


def retiming_scale(curve, limits):
    return float(interval_retiming_scales(curve, limits).max())*1.02


def joint_axes(kinematics, qs):
    """Joint axes in the base frame for a batch of configurations, shape (samples, 3, 6).

    Walks the parsed URDF chain (``PiperXKinematics.joint_origins`` / ``joint_axes``) for all
    samples at once with the same products in the same order as
    ``fk_with_spatial_jacobian``, so the axes are bit for bit those of the per-sample call.
    """
    qs = np.asarray(qs, dtype=float)
    transform = np.tile(np.eye(4), (len(qs), 1, 1))
    axes = np.empty((len(qs), 6, 3))
    for i in range(6):
        pre = transform @ kinematics.joint_origins[i]
        axes[:, i] = pre[:, :3, :3] @ kinematics.joint_axes[i]
        step = np.tile(np.eye(4), (len(qs), 1, 1))
        step[:, :3, :3] = Rotation.from_rotvec(np.outer(np.deg2rad(qs[:, i]), kinematics.joint_axes[i])).as_matrix()
        transform = pre @ step
    return np.transpose(axes, (0, 2, 1))


def angular_interval_scales(model, curve):
    """Per-interval slow-down factor (>= 1) that keeps the solved TCP angular speed under the limit.

    Every interpolation interval is inspected, including its interior: with a free orientation the
    solved joint curve, not the requested RPY, determines how fast the tool turns.
    """
    times = (curve.x[:-1, None]+np.diff(curve.x)[:, None]*np.linspace(0, 1, 9)).ravel()
    qs = curve(times)
    velocities = curve(times, 1)
    axes = joint_axes(model.kin, qs)
    speeds = np.linalg.norm((axes @ velocities[:, :, None])[:, :, 0], axis=1)
    return np.maximum(1., speeds.reshape(-1, 9).max(axis=1)/MAX_ANGULAR_SPEED)


def angular_retiming_scale(model, curve):
    return float(angular_interval_scales(model, curve).max())*1.02


def tracking_gap(curve, actual, t, lag_s, step=.02):
    """Largest joint gap to the nearest reference point in [t-lag_s, t], with that joint's index."""
    taus = np.r_[np.arange(t, max(t-lag_s, 0.), -step), max(t-lag_s, 0.)]
    actual = np.asarray(actual, dtype=float)
    gaps = np.array([np.abs(np.asarray(curve(tau), dtype=float)-actual) for tau in taus])
    nearest = int(np.argmin(gaps.max(axis=1)))
    return float(gaps[nearest].max()), int(np.argmax(gaps[nearest]))


def measured_progress(curve, actual, lo, hi, step=.02):
    """Reference time in [lo, hi] whose joints lie nearest the measured joints; never moves backwards."""
    if hi <= lo:
        return float(lo)
    taus = np.r_[np.arange(lo, hi, step), hi]
    actual = np.asarray(actual, dtype=float)
    gaps = [float(np.max(np.abs(np.asarray(curve(tau), dtype=float)-actual))) for tau in taus]
    return float(taus[int(np.argmin(gaps))])
