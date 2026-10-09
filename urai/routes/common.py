"""Request helpers shared by the route modules."""
from __future__ import annotations

from ..backend import ARMS


def require_fields(data, *names):
    """Refuse a request body that lacks any of ``names``."""
    missing = [name for name in names if name not in data]
    if missing:
        raise ValueError('Missing field(s): ' + ', '.join(missing))


def require_arm(data, default='left'):
    """The arm a drafting request names; ``left`` when it names none."""
    arm = data.get('arm', default)
    if arm not in ARMS:
        raise ValueError('Choose left or right arm')
    return arm


def require_current_observation(service, data):
    """Refuse a request built on anything but the current observation. Call under ``service.lock``."""
    if not service.frame or data.get('observation_id') != service.frame.id:
        raise ValueError('Stale observation')


def require_unmoved(service, current, what):
    """Refuse drafting when the robot moved since the observation and automatic refresh is off."""
    if service._observation_stale(current) and not service.auto_prepare['refresh_observation']:
        raise ValueError(f'机器人已移动，请更新图像后再{what}')


def arm_geometry(service, arm, current):
    """``(model, base_xy, table_z_m, fingertip_bias_mm)`` a stroke on ``arm`` is planned with.

    The calibrated base decides how a grasp leans and which side the wrist takes; the simulation has no
    model, so the arm's current TCP plays that role.
    """
    model = service.backend.models.get(arm)
    base_xy = model.base_xy if model is not None else current[arm]['xyz'][:2]
    table_z_m, bias_mm = service.arm_calibration(arm)
    return model, base_xy, table_z_m, bias_mm


def reach_probe(service, arm, current):
    """IK-backed reachability probe for draft generators, or ``{}`` without a calibrated model."""
    if service.backend.models.get(arm) is None:
        return {}
    seed = list(current[arm]['joints_deg'])
    return {'reach_probe': lambda xyz, rotation, joints: service.backend.pose_error(
                arm, xyz, rotation, seed if joints is None else joints, service.pose_tolerance),
            'pose_tolerance': (service.pose_tolerance['position_mm'], service.pose_tolerance['orientation_deg'])}
