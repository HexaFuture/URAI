"""Level-finger side approaches for measured rigid objects at long reach."""
import copy
import numpy as np
from .grasp_solver import solve_line_grasp, TILTS_DEG
from .trajectory import compile_arm


def grasp_options(ctx):
    if not ctx.has_kinematics:
        return None
    from .grasp import grasp_terms
    terms = grasp_terms(ctx.arm, ctx.model, ctx.backend.table_checks)
    return dict(probe=ctx.probe, seed_joints=ctx.current[ctx.arm]['joints_deg'],
                pose_tolerance=ctx.pose_tolerance,
                compensation_m=terms['pinch_compensation_m'], min_tcp_z_m=terms['min_tcp_z_m'],
                observed_table_frame=(ctx.arm == 'right' and getattr(
                    getattr(ctx.backend, 'runtime', None), 'calibration', None) is not None))


def refine_transfer(result, placement, route, yaw, base_xy, clearance_m, options):
    """Choose a level-finger pose whose complete interpolated carry is reachable.

    Reachable contact and approach endpoints alone do not establish that the
    intervening carry/descent can follow the same IK branch. Retry with a small
    tilt when a vertical candidate fails; the normal preview still checks collisions.
    """
    last_error = None
    for first in range(len(TILTS_DEG)):
        try:
            candidate = _transfer_candidate(copy.deepcopy(result), placement, route, yaw,
                                            base_xy, clearance_m, options, TILTS_DEG[first:])
            spec = candidate['arm']
            compiled = compile_arm(spec, {'xyz': spec['path']['points'][0],
                                          'rpy_deg': spec['orientation']['points'][0][1:]})
            q = np.asarray(options['seed_joints'], float)
            tolerance = np.minimum(options['pose_tolerance'], [10., 3.])
            for point, rotation in zip(compiled['xyz'], compiled['rotation_matrices']):
                pe, re, following = options['probe'](point, rotation, q)
                if following is None or not np.isfinite([pe, re]).all() or pe > tolerance[0] or re > tolerance[1]:
                    raise ValueError('连续搬运或落点接近路径不可达')
                q = following
            return candidate
        except ValueError as error:
            last_error = error
    raise ValueError(f'抓放路径不可达：{last_error}')


def _transfer_candidate(result, placement, route, yaw, base_xy, clearance_m, options, tilts):
    """Keep the contact anchors, solve both end poses, and approach along the tool axis.

    A base-radial lean can tilt the jaw closing axis out of the table plane: on a 14 mm
    block the old pose separated the two fingertip heights by 11 mm. Here the jaw axis
    stays horizontal. Full interpolated IK and collision checks still run in preview.
    """
    solve = lambda anchor: solve_line_grasp(anchor, yaw, base_xy, clearance_m,
        options['compensation_m'], options['probe'], options['seed_joints'],
        options['pose_tolerance'], options['min_tcp_z_m'], tilts_deg=tilts)
    source, dest = solve(placement['grasp']), solve(placement['place'])
    height = max(placement['hover'][2], source['lift'][2], dest['pre'][2])
    lift = source['target'].copy(); lift[2] = height
    # Keep the release approach on its tool axis as well. Its entry is connected by
    # a high waypoint, so the final tilted motion cannot drag a fingertip across a neighbour.
    entry = dest['pre'].copy(); entry[2] = max(height, entry[2])
    points = [source['pre'], source['target'], lift, *route, entry, dest['pre'], dest['target']]
    if placement['placement'] != 'drop':
        points.append(dest['pre'])
    n = len(points)-1
    spec = result['arm']
    spec['path']['points'] = np.asarray(points).tolist()
    spec['orientation']['points'] = [[0., *source['pose']], [2/n, *source['pose']],
                                     [3/n, *dest['pose']], [1., *dest['pose']]]
    events = spec['gripper_events']
    events[1]['s'] = 1/n
    events[-1]['s'] = 1. if placement['placement']=='drop' else (n-1)/n
    result.update(grasp_mode=source['mode'], grasp_tilt_deg=source['tilt_deg'],
                  release_tilt_deg=dest['tilt_deg'],
                  source_contact_xyz=np.asarray(placement['grasp']).tolist(),
                  target_contact_xyz=np.asarray(placement['place']).tolist(),
                  source_xyz=source['target'].tolist(), target_xyz=dest['target'].tolist(),
                  approach_axis=source['rotation'][:,2].tolist(),
                  fingertip_height_difference_mm=0.,
                  grasp_solver_candidates=source['candidates'])
    return result
