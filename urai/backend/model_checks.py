"""Per-arm kinematic checks on top of the PiPER-X arm model: joint bounds, IK residuals and the table floor."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from ..settings import normalize_pose_tolerance

#: Only actual interpenetration of the gripper with the arm's own base or upper arm is refused. The
#: capsule model's own 15 mm margin would reject the overhand wind-up, which folds the tool 11 mm
#: from the base column.
SELF_COLLISION_MIN_CLEARANCE_MM = 0.


def model_joint_limit_reason(model, q, *, bounds=None, tolerance_deg=1e-9):
    """Why planned joints ``q`` leave the model's (or the given) bounds beyond roundoff; None when inside."""
    q = np.asarray(q,dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        return 'joint feedback must contain six finite angles'
    for i,value in enumerate(q,1):
        lo,hi = model.kin.limits_deg[i] if bounds is None else bounds[i-1]
        if value < lo-tolerance_deg or value > hi+tolerance_deg:
            return f'planned joint limit exceeded (j{i}={value:+.3f} deg; allowed {lo:g}..{hi:g})'
    return None


def measured_joint_bounds(model, q):
    """An initial measured pose is admissible without changing the model."""
    q=np.asarray(q,dtype=float)
    if q.shape!=(6,) or not np.isfinite(q).all():
        raise ValueError('joint feedback must contain six finite angles')
    bounds=np.asarray([model.kin.limits_deg[i+1] for i in range(6)],dtype=float)
    return np.c_[np.minimum(bounds[:,0],q),np.maximum(bounds[:,1],q)]


def recovery_joint_bounds(model, qs, task_index):
    qs=np.asarray(qs,dtype=float)
    if qs.ndim!=2 or qs.shape[1]!=6 or not len(qs) or not np.isfinite(qs).all():
        raise ValueError('Joint trajectory must contain finite six-joint poses')
    nominal=np.asarray([model.kin.limits_deg[i+1] for i in range(6)])
    outside=np.maximum(nominal[:,0]-qs,0)+np.maximum(qs-nominal[:,1],0)
    # Only the initial approach may return from a taught pose. Once inside,
    # no subsequent knot may leave; task targets always use nominal bounds.
    if np.any(np.diff(outside,axis=0)>1e-9) or np.any(outside[task_index:]>1e-9):
        raise ValueError('Joint recovery must move inward and enter nominal bounds before the task')
    return measured_joint_bounds(model,qs[0])


def canonical_planned_joints(model, q):
    # Polynomial evaluation may be one ULP outside a nominal joint bound.
    # Only canonicalize arithmetic roundoff; substantive violations stay
    # unchanged so the strict geometry validator rejects them.
    q=np.asarray(q,dtype=float)
    bounds=np.asarray([model.kin.limits_deg[i+1] for i in range(6)]).T
    clipped=np.clip(q,*bounds)
    return np.where(np.abs(q-clipped)<=1e-9,clipped,q)


def self_collision_reason(model, q):
    """Reason the gripper interpenetrates the arm's own base or upper arm at ``q``; None when it clears."""
    clearance, mover, target = model.self_collision_clearance(np.asarray(q, dtype=float))
    if clearance < SELF_COLLISION_MIN_CLEARANCE_MM:
        return f'{model.name} self-collision: {mover} vs {target} (clearance {clearance:.0f} mm)'
    return None


def ik_errors(model,target,q):
    q=np.asarray(q,dtype=float)
    if q.shape!=(6,) or not np.isfinite(q).all():
        raise ValueError('IK returned an invalid joint solution')
    for i,value in enumerate(q):
        lo,hi=model.kin.limits_deg[i+1]
        if not lo<=value<=hi:
            raise ValueError(f'IK joint limit exceeded: j{i+1}={value:.4f} degrees')
    # No facing rules (|j1| <= 90, wrist ahead of the base plane, j1 toward the target) are applied:
    # the overhand throw winds up above the base.
    actual=model.fk_tcp_world(q)
    pe=float(np.linalg.norm(actual[:3,3]-target[:3,3])*1000.)
    re=float(np.rad2deg((Rotation.from_matrix(actual[:3,:3]).inv()*Rotation.from_matrix(target[:3,:3])).magnitude()))
    return q,pe,re


def tcp_position_jacobian(model,q):
    pose,jac,axes=model.kin.fk_with_spatial_jacobian(q)
    offset=pose[:3,:3] @ model.t_link6_tcp[:3,3]
    tcp_jac=jac+np.cross(axes.T,offset).T*np.deg2rad(1.)
    return model.t_world_base[:3,:3] @ tcp_jac


def solve_ik(model, target, seed, pose_tolerance=None, orientation_free=False):
    if orientation_free:
        bounds=np.asarray([model.kin.limits_deg[i+1] for i in range(6)]).T
        seed=np.clip(np.asarray(seed,dtype=float),*bounds)
        # Optimize the actual tool center, including its rotating TCP offset.
        # A small joint-space preference selects nearby solutions without an RPY target.
        weight=.02
        def residual(q):
            return np.r_[(model.fk_tcp_world(q)[:3,3]-target[:3,3])*1000.,weight*(q-seed)]
        def jac(q):
            return np.vstack([tcp_position_jacobian(model, q), weight*np.eye(6)])
        q=least_squares(residual,seed,jac=jac,bounds=bounds,max_nfev=100,
                        ftol=1e-10,xtol=1e-10,gtol=1e-10).x
        return ik_errors(model,target,q)
    # The underlying optimizer already enforces the URDF joint bounds. ArmModel.ik_tcp_world
    # additionally pushes every solution 1 degree away from those bounds, which would invalidate
    # otherwise exact poses near home, so the kinematics' best-effort solver is called directly.
    base = model.t_base_world @ np.asarray(target, dtype=float)
    base[:3,3] *= 1000.
    tolerance = normalize_pose_tolerance(pose_tolerance)
    # First continue from the previous joint solution. The default search keeps restarting
    # for 0.5 mm / 0.1 deg even when the TCP tolerance is already met, which costs time and can
    # jump between IK branches.
    for restarts in (0, 6):
        q, _, _ = model.kin.ik_best_effort(base @ model.t_tcp_link6, seed_deg=seed, max_restarts=restarts)
        q = np.asarray(q, dtype=float)
        try:
            q,position_error,rotation_error = ik_errors(model,target,q)
        except ValueError:
            if restarts == 0:
                continue
            raise
        if restarts or (position_error <= tolerance['position_mm'] and rotation_error <= tolerance['orientation_deg']):
            return q, position_error, rotation_error


def floor_world_mm(model):
    """Lowest real fingertip height the model's table check accepts, in world millimetres."""
    t = np.asarray(model.t_world_base, dtype=float)
    if abs(float(t[2, 2])-1.) > 1e-9:
        raise ValueError(f'{model.name}: base is not upright; the table floor has no single world height')
    return float(t[2, 3])*1000.+float(model.table_z_mm)-float(model.fingertip_below_table_mm)
