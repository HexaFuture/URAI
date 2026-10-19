"""Pure observed-rim geometry for bounded, grasp-preserving pour corrections.

All coordinates are raw world metres. No segmentation, robot access, or motion is
performed here. Source normal signs must follow the bottle through 90 degrees.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

FIT_LIMIT_M = .003
MAX_TILT_DEG = 150.


def _vector(value, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(name + ' must contain three finite coordinates')
    return result.copy()


def _normal(value, name):
    result = _vector(value, name)
    length = np.linalg.norm(result)
    if length < 1e-8:
        raise ValueError(name + ' must be nonzero')
    return result / length


def fit_rim(points_world, normal_hint=(0., 0., 1.)):
    """Fit 4..16 rim landmarks; require at least 120 degrees of circular coverage."""
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= 16 or not np.isfinite(points).all():
        raise ValueError('A rim needs 4..16 finite 3D landmarks')
    if len(np.unique(np.round(points, 9), axis=0)) < 4:
        raise ValueError('A rim needs four distinct landmarks')
    hint = _normal(normal_hint, 'normal_hint')
    origin = points.mean(axis=0)
    _, singular, basis = np.linalg.svd(points-origin, full_matrices=False)
    if singular[1] < 1e-5:
        raise ValueError('Rim landmarks are degenerate or collinear')
    normal = basis[2].copy()
    if normal @ hint < 0:
        normal = -normal
    plane_rms = float(np.sqrt(np.mean(((points-origin) @ normal)**2)))
    xy = (points-origin) @ basis[:2].T
    design = np.c_[2*xy, np.ones(len(xy))]
    initial, _, rank, _ = np.linalg.lstsq(design, np.sum(xy**2, axis=1), rcond=None)
    if rank != 3:
        raise ValueError('Rim circle is degenerate')
    radius_sq = initial[2] + initial[:2] @ initial[:2]
    if radius_sq <= 0:
        raise ValueError('Rim radius is invalid')
    fitted = least_squares(
        lambda v: np.linalg.norm(xy-v[:2], axis=1)-v[2],
        np.r_[initial[:2], np.sqrt(radius_sq)], max_nfev=100)
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise ValueError('Rim circle fit did not converge')
    circle_xy, radius = fitted.x[:2], float(fitted.x[2])
    radial_rms = float(np.sqrt(np.mean(fitted.fun**2)))
    if plane_rms > FIT_LIMIT_M or radial_rms > FIT_LIMIT_M:
        raise ValueError('Rim plane or radial RMS exceeds 3 mm')
    if not .008-1e-10 <= radius <= .100+1e-10:
        raise ValueError('Rim radius must be 8..100 mm')
    offsets = xy-circle_xy
    angles = np.sort(np.mod(np.arctan2(offsets[:, 1], offsets[:, 0]), 2*np.pi))
    span = 2*np.pi-float(np.max(np.diff(np.r_[angles, angles[0]+2*np.pi])))
    if span < np.deg2rad(120)-1e-8:
        raise ValueError('Rim landmarks cover less than 120 degrees')
    return {'center': origin+circle_xy @ basis[:2], 'normal': normal,
            'radius': radius, 'residual_m': max(plane_rms, radial_rms)}


def lowest_lip(center, normal, radius):
    """Lowest point of a circular mouth; a horizontal rim uses its center."""
    center = _vector(center, 'center')
    normal = _normal(normal, 'normal')
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be finite and positive')
    down = np.array([0., 0., 1.])-normal*normal[2]
    length = np.linalg.norm(down)
    return center if length < 1e-8 else center-radius*down/length


def _rim(value, name):
    if not isinstance(value, dict):
        raise ValueError(name + ' must be a fitted rim')
    try:
        center = _vector(value['center'], name+'.center')
        normal = _normal(value['normal'], name+'.normal')
        radius = float(value['radius'])
        residual = float(value.get('residual_m', 0.))
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError(name + ' is incomplete or invalid') from exc
    if not np.isfinite([radius, residual]).all() or not .008-1e-10 <= radius <= .100+1e-10 or not 0 <= residual <= FIT_LIMIT_M+1e-10:
        raise ValueError(name + ' has invalid radius or residual')
    return center, normal, radius, residual


def plan_step(tcp_xyz, tcp_rpy_deg, source_rim, target_rim, *, mode='align',
              tilt_step_deg=5., clearance_m=.04, max_translation_m=.015,
              max_tilt_step_deg=5., tolerance_m=.003, bottle_length_m=.35,
              bottle_radius_m=.06, pour_direction_xy=None):
    """Return one bounded correction, including sampled rigid-mouth compensation.

    Align clamps translation only. Tilt requires alignment and reduces its angle
    until the complete compensated TCP path fits max_translation_m. The supplied
    pour direction is world XY; retain it across observations. Without it a tilted
    mouth supplies its own direction, then an upright mouth uses target minus TCP.
    At upright, that direction selects the prospective pouring edge among the
    equally low rim points; ALIGN establishes it before the first tilt. Standalone
    lowest_lip retains its center convention. aligned/error_m describe the CURRENT
    observation; planned_aligned/planned_error_m describe the emitted endpoint.
    """
    p0 = _vector(tcp_xyz, 'tcp_xyz')
    rpy0 = _vector(tcp_rpy_deg, 'tcp_rpy_deg')
    R0 = Rotation.from_euler('xyz', rpy0, degrees=True).as_matrix()
    c0, n0, radius, source_error = _rim(source_rim, 'source_rim')
    target, target_normal, target_radius, target_error = _rim(target_rim, 'target_rim')
    parameters = np.asarray([tilt_step_deg, clearance_m, max_translation_m,
                             max_tilt_step_deg, tolerance_m, bottle_length_m,
                             bottle_radius_m], dtype=float)
    if not np.isfinite(parameters).all() or np.any(parameters <= 0):
        raise ValueError('Step limits, clearance, tolerance and bottle dimensions must be positive and finite')
    if mode not in ('align', 'tilt'):
        raise ValueError('mode must be align or tilt')
    if max_tilt_step_deg > MAX_TILT_DEG:
        raise ValueError('max_tilt_step_deg cannot exceed 150 degrees')
    if np.rad2deg(np.arccos(np.clip(abs(target_normal[2]), 0., 1.))) >= 15.-1e-9:
        raise ValueError('Target rim must be less than 15 degrees from horizontal')
    if target_radius-target_error-source_error <= .010:
        raise ValueError('Target opening needs more than 10 mm of reliable interior radius')
    initial_tilt = float(np.rad2deg(np.arccos(np.clip(n0[2], -1., 1.))))
    if initial_tilt > MAX_TILT_DEG+1e-8:
        raise ValueError('Observed source tilt exceeds 150 degrees')
    direction = None
    if pour_direction_xy is not None:
        direction = np.asarray(pour_direction_xy, dtype=float)
        if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            raise ValueError('pour_direction_xy must be a finite nonzero 2D direction')
        direction = direction/np.linalg.norm(direction)
    elif np.linalg.norm(n0[:2]) > 1e-6:
        direction = n0[:2]/np.linalg.norm(n0[:2])
    elif np.linalg.norm(target[:2]-p0[:2]) > 1e-8:
        direction = (target[:2]-p0[:2])/np.linalg.norm(target[:2]-p0[:2])
    target_top = float(target[2]+target_radius*np.linalg.norm(target_normal[:2])+target_error)

    def selected_lip(center, normal):
        if normal[2] > 0 and np.linalg.norm(normal[:2]) < 1e-8 and direction is not None:
            return np.asarray(center)+np.r_[radius*direction, 0.]
        return lowest_lip(center, normal, radius)

    def desired_center(normal):
        lip_offset = selected_lip(np.zeros(3), normal)
        side = np.linalg.norm(normal[:2])
        lip_height = target_top+clearance_m+source_error
        body_height = target_top+.020+source_error+bottle_length_m*max(normal[2], 0.)+bottle_radius_m*side
        center = np.r_[target[:2]-lip_offset[:2], max(lip_height-lip_offset[2], body_height)]
        return center

    goal0 = desired_center(n0)
    error0 = float(np.linalg.norm(goal0-c0))
    if mode == 'align':
        shift = goal0-c0
        shift *= min(1., max_translation_m/max(error0, 1e-15))
        positions = np.stack([p0, p0+shift])
        rotations = [R0, R0]
        final_center, final_normal = c0+shift, n0
    else:
        if error0 > tolerance_m+1e-10:
            raise ValueError('Align the mouth over the target before tilting')
        if direction is None:
            raise ValueError('Upright on-axis geometry requires pour_direction_xy')
        if np.linalg.norm(n0[:2]) > 1e-6 and direction @ (n0[:2]/np.linalg.norm(n0[:2])) < np.cos(np.deg2rad(15.))-1e-10:
            raise ValueError('pour_direction_xy disagrees with the observed tilt direction')
        axis = np.r_[-direction[1], direction[0], 0.]
        lever = c0-p0

        def arc(angle_deg):
            # The body envelope changes derivative when its axis crosses horizontal.
            # Include that angle exactly so interpolation cannot round off its clearance.
            crossing = float(np.rad2deg(np.arctan2(n0[2], direction @ n0[:2])))
            boundaries = [0., crossing, angle_deg] if 1e-9 < crossing < angle_deg-1e-9 else [0., angle_deg]
            angles = np.unique(np.concatenate([
                np.linspace(a, b, max(1, int(np.ceil(b-a)))+1)
                for a, b in zip(boundaries, boundaries[1:])]))
            positions, rotations = [p0.copy()], [R0.copy()]
            for angle in angles:
                Q = Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
                normal = Q @ n0
                center = desired_center(normal)
                point = center-Q @ lever
                # Keep the actual measured start, including a small alignment residual.
                if angle == 0. and np.linalg.norm(point-p0) < 1e-12:
                    continue
                positions.append(point)
                rotations.append(Q @ R0)
            if len(positions) == 1:
                positions.append(p0.copy())
                rotations.append(R0.copy())
            positions = np.asarray(positions)
            distance = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            return positions, rotations, center, normal, distance

        maximum = min(float(tilt_step_deg), float(max_tilt_step_deg),
                      max(0., MAX_TILT_DEG-initial_tilt))
        if error0 > max_translation_m+1e-10:
            raise ValueError('Finish alignment before tilting within this translation limit')
        candidate = arc(maximum)
        if candidate[-1] > max_translation_m:
            low, high = 0., maximum
            for _ in range(45):
                middle = (low+high)/2
                if arc(middle)[-1] <= max_translation_m:
                    low = middle
                else:
                    high = middle
            candidate = arc(low)
        positions, rotations, final_center, final_normal, _ = candidate

    tilt = float(np.rad2deg(np.arccos(np.clip(final_normal[2], -1., 1.))))
    error = float(np.linalg.norm(desired_center(final_normal)-final_center))
    fractions = np.linspace(0., 1., len(positions))
    keyframes = [[float(s), *Rotation.from_matrix(R).as_euler('xyz', degrees=True).tolist()]
                 for s, R in zip(fractions, rotations)]
    return {'xyz': positions[-1].tolist(), 'rpy_deg': keyframes[-1][1:],
            'source_center': final_center.tolist(), 'source_normal': final_normal.tolist(),
            'lip_xyz': selected_lip(final_center, final_normal).tolist(),
            'aligned': bool(error0 <= tolerance_m+1e-10), 'error_m': error0,
            'planned_aligned': bool(error <= tolerance_m+1e-10), 'planned_error_m': error,
            'tilt_deg': tilt, 'delta_tilt_deg': max(0., tilt-initial_tilt),
            'path': {'mode': 'waypoints', 'points': positions.tolist()},
            'orientation': {'mode': 'keyframes', 'points': keyframes},
            'path_length_m': float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
            'pour_direction_xy': None if direction is None else direction.tolist()}
