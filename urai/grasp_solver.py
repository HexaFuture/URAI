"""Reachability search that preserves the annotated horizontal jaw axis."""
import numpy as np
from scipy.spatial.transform import Rotation

# From vertical; the last candidate is a side grasp pitched 15 degrees down.
TILTS_DEG = (0., 20., 35., 50., 65., 75.)


def line_rotation(yaw_deg, tilt_deg, side):
    yaw = np.deg2rad(yaw_deg)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.])
    horizontal = side * np.array([-np.sin(yaw), np.cos(yaw), 0.])
    a = np.deg2rad(tilt_deg)
    z = horizontal*np.sin(a) + np.array([0., 0., -np.cos(a)])
    return np.column_stack([x, np.cross(z, x), z])


def solve_line_grasp(contact, yaw_deg, base_xy, clearance_m, compensation_m,
                     probe, seed_joints, pose_tolerance=(10., 3.), min_tcp_z_m=None):
    """Keep both fingertips level. Try vertical then minimal tilt, seeded continuously.

    Probe the grasp first to avoid spending time on impossible orientations, then sample
    the approach and lift with one IK branch. Full trajectory/collision validation follows
    in the ordinary preview pipeline; no command is sent from here.
    """
    contact=np.asarray(contact,dtype=float)
    initial=np.asarray(seed_joints,dtype=float)
    if contact.shape!=(3,) or initial.shape!=(6,) or not np.isfinite(np.r_[contact,initial]).all():
        raise ValueError('抓取求解需要有效目标和六个关节初值')
    yaw=np.deg2rad(yaw_deg)
    sideways=np.array([-np.sin(yaw),np.cos(yaw)])
    outward=contact[:2]-np.asarray(base_xy,dtype=float)
    preferred=1. if outward@sideways >= 0 else -1.
    # Small pose errors matter for keeping two fingertips on the annotated endpoints.
    tol=np.minimum(np.asarray(pose_tolerance,dtype=float),[10.,3.])
    seeds=[initial, np.zeros(6)]
    tried=0
    for tilt in TILTS_DEG:
        for side in ([preferred] if tilt==0 else [preferred,-preferred]):
            rotation=line_rotation(yaw_deg,tilt,side)
            target=contact+rotation[:,1]*compensation_m
            if min_tcp_z_m is not None and target[2] < min_tcp_z_m-1e-9:
                continue
            pre=target-rotation[:,2]*clearance_m
            lift=target+[0.,0.,.04]
            for seed in seeds:
                tried+=1
                pe,re,q=probe(target,rotation,seed)
                if q is None or not np.isfinite([pe,re]).all() or pe>tol[0] or re>tol[1]:
                    continue
                # Seed the approach from the solved contact; then traverse forward in order.
                pe,re,q=probe(pre,rotation,q)
                if q is None or not np.isfinite([pe,re]).all() or pe>tol[0] or re>tol[1]:
                    continue
                good=True
                for start,end in [(pre,target),(target,lift)]:
                    for point in np.linspace(start,end,7)[1:]:
                        pe,re,following=probe(point,rotation,q)
                        if following is None or not np.isfinite([pe,re]).all() or pe>tol[0] or re>tol[1] or np.max(np.abs(np.asarray(following)-q))>45.:
                            good=False;break
                        q=following
                    if not good:break
                if good:
                    return dict(pose=Rotation.from_matrix(rotation).as_euler('xyz',degrees=True).tolist(),
                                target=target,pre=pre,lift=lift,rotation=rotation,
                                tilt_deg=tilt,mode='top_down' if tilt==0 else 'tilted_side',candidates=tried)
    raise ValueError('竖直及倾斜侧抓均不可达：已保持画线两端的指尖方向，并检查接近与提起；请调整画线方向或位置')


def append_grasp_lift(result):
    """Append the validated lift only to a closing action, never to release/keep."""
    if result.get('lift_xyz') is None:return result
    arm=result['arm']
    arm['path']['points'].append(result['lift_xyz'])
    for key in arm['orientation']['points']:key[0]/=2
    arm['orientation']['points'].append([1.,*arm['orientation']['points'][-1][1:]])
    for event in arm['gripper_events']:event['s']/=2
    return result
