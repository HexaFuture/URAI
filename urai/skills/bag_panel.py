"""One-call upper-panel grasp and lift, with a reversible held-object trajectory."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .registry import register, number, integer
from .bag import checked_spec
from ..grasp import grasp_from_line, grasp_terms


@register(
    name='bag_panel_lift', label='开包 · 抓上层并提起', group='柔性物体',
    stroke='two_points',
    summary='两点指定上层布片的两指夹持方向，夹住后连续上提并保持；不自动回位。',
    hint='在上层布片可夹取的位置画短线，两端是两指位置；不是沿包口描边。',
    inputs={
        'inset_mm': number('夹入深度', 0, 24, 18, 1, ' mm'),
        'opening_mm': number('初始开口', 12, 60, 40, 1, ' mm'),
        'lift_mm': number('夹住后总上提高度', 40, 140, 100, 5, ' mm'),
        'grip_effort': integer('夹持力矩', 100, 1500, 1200),
        'speed_m_s': number('上提速度', .05, .22, .15, .01, ' m/s'),
        'approach_speed_m_s': number('接近速度', .05, .2, .12, .01, ' m/s'),
    })
def panel_lift(ctx, ink, inset_mm=18, opening_mm=40, lift_mm=100,
               grip_effort=1200, speed_m_s=.15, approach_speed_m_s=.12):
    line = np.asarray(ink, float)
    if line.shape != (2, 2) or not np.isfinite(line).all():
        raise ValueError('请画上层布片的两指夹持线')
    uv = line.mean(axis=0)
    samples = np.asarray([ctx.frame.pick(*(uv+d), mode='surface')
                          for d in ([0, 0], [3, 0], [-3, 0], [0, 3], [0, -3])])
    if not np.isfinite(samples).all() or np.ptp(samples[:, 2]) > .025:
        raise ValueError('布片夹持点周围深度不一致，请换到连续的上层布片区域')
    surface = np.median(samples, axis=0)
    result = grasp_from_line(
        ctx.frame, line, inset_mm=inset_mm, clearance_m=.06,
        speed_m_s=speed_m_s, preferred_yaw_deg=ctx.preferred_yaw_deg,
        base_xy=ctx.base_xy, grip_effort=grip_effort, surface_xyz=surface,
        top_down=not ctx.has_kinematics,
        reach_probe=ctx.probe if ctx.has_kinematics else None,
        seed_joints=ctx.current[ctx.arm]['joints_deg'], pose_tolerance=(3., 3.),
        **grasp_terms(ctx.arm, ctx.model, ctx.backend.table_checks))
    original = result['arm']
    approach, contact = np.asarray(original['path']['points'])
    R = Rotation.from_euler('xyz', original['orientation']['points'][-1][1:], degrees=True).as_matrix()
    # First lift 40 mm without twisting the pinch. Small rotations around the jaw axis
    # keep both fingertips level while searching a reachable higher holding posture.
    heights = np.linspace(.04, lift_mm/1000, 6)
    points = [approach, contact] + [contact + [0, 0, z] for z in heights]
    failures = []
    for tilt in (0., 10., -10., 20., -20.):
        rotations = [R, R] + [Rotation.from_rotvec(R[:, 0]*np.deg2rad(tilt)*t).as_matrix() @ R
                               for t in np.linspace(0, 1, len(heights))]
        events = copy.deepcopy(original['gripper_events'])
        for event in events:
            if event.get('pregrasp_open'):
                event['opening_mm'] = opening_mm
            else:
                event['s'] = 1/(len(points)-1)
        try:
            spec = checked_spec(ctx, points, rotations, events, speed=speed_m_s)
        except ValueError as error:
            failures.append(str(error))
            continue
        spec['approach']['speed_m_s'] = approach_speed_m_s
        # Preserve every executed post-contact pose for lowering and releasing later.
        poses = [[*p.tolist(), *Rotation.from_matrix(r).as_euler('xyz', degrees=True).tolist()]
                 for p, r in zip(points, rotations)]
        metadata = {'phase': 'panel_lift', 'arm': ctx.arm, 'observation_id': ctx.frame.id,
                    'bag_open_verified': False, 'contact_xyz': surface.tolist(),
                    'grasp_pose': poses[1], 'approach_pose': poses[0],
                    'return_poses': list(reversed(poses[1:])), 'lift_mm': lift_mm}
        spec['bag_flap'] = metadata
        return {'arm': spec, 'bag_flap': metadata, 'surface_xyz': surface.tolist(),
                'grasp_mode': result.get('grasp_mode'), 'grasp_tilt_deg': result.get('grasp_tilt_deg'),
                'lift_tilt_deg': tilt, 'actual_inset_mm': result['actual_inset_mm'],
                'summary': '上层布片已规划为夹持后连续上提并保持；执行后由 RGB 判断主仓是否打开。'}
    raise ValueError('上层抓提路径不可达，请换夹持线方向或位置：' + failures[-1])
