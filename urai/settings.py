"""Shared, explicit planner settings: pose tolerances, motion profiles and the safety defaults."""
import math

DEFAULT_POSE_TOLERANCE = {'position_mm': 10., 'orientation_deg': 3.}
MOTION_PROFILES = {
    'fine': {'label': '精细', 'joint_speed_deg_s': 10., 'joint_acceleration_deg_s2': 30., 'joint_jerk_deg_s3': 300.,
             'controller_speed_percent': 10},
    'normal': {'label': '正常', 'joint_speed_deg_s': 30., 'joint_acceleration_deg_s2': 120., 'joint_jerk_deg_s3': 3000.,
               'controller_speed_percent': 40},
    'fast': {'label': '快速', 'joint_speed_deg_s': 45., 'joint_acceleration_deg_s2': 240., 'joint_jerk_deg_s3': 6000.,
             'controller_speed_percent': 60},
    # Tossing needs the swing to reach about 1 m/s at the fingertips; only use it with a clear workspace.
    'throw': {'label': '抛掷', 'joint_speed_deg_s': 150., 'joint_acceleration_deg_s2': 2500., 'joint_jerk_deg_s3': 80000.,
              'controller_speed_percent': 100},
}

#: Conservative defaults of a fresh service. ``configs/paper-settings.json`` holds the settings the paper's
#: real-robot trials ran with, which relax several of these; see docs/safety.md before using it.
DEFAULT_MOTION_PROFILE = 'normal'
#: Largest joint gap (degrees) between the streamed reference and the measured joints before the arm is held.
DEFAULT_TRACKING_LIMIT_DEG = 3.
#: Table clearance and table collision checks in planning and in the 50 Hz feedback guard.
DEFAULT_TABLE_CHECKS = True


def motion_limits(profile=DEFAULT_MOTION_PROFILE):
    if not isinstance(profile, str) or profile not in MOTION_PROFILES:
        raise ValueError('运动档位必须是 ' + '、'.join(MOTION_PROFILES) + ' 之一')
    return {'profile': profile, **MOTION_PROFILES[profile]}


def motion_profiles():
    """Every selectable profile with its limits, in display order."""
    return [motion_limits(name) for name in MOTION_PROFILES]


def normalize_pose_tolerance(value=None):
    if value is None:
        return dict(DEFAULT_POSE_TOLERANCE)
    if not isinstance(value, dict) or set(value) != set(DEFAULT_POSE_TOLERANCE):
        raise ValueError('位姿容差需要 position_mm 和 orientation_deg 两个数值')
    result = {}
    for key, number in value.items():
        if type(number) not in (int, float):
            raise ValueError('位姿容差必须是非负有限数值')
        try:
            number = float(number)
        except (ValueError, OverflowError) as exc:
            raise ValueError('位姿容差必须是非负有限数值') from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError('位姿容差必须是非负有限数值')
        result[key] = number
    if result['orientation_deg'] > 180:
        raise ValueError('姿态容差范围为 0–180 度')
    return result
