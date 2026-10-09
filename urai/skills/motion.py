"""Route pieces every skill shares: tool frames, gripper events, the draft envelope and the candidate search.

Nothing here is specific to one skill. A skill decides *where* the tool goes; this module keeps *how* a route is
written down the same as the transfer and pour skills already write it, so the planner, the preview and the GUI
see one shape.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from ..pour import EXACT_DEG, EXACT_MM, bearing, heading, object_extent, probe_chain, rpy_deg, side_grasp_rotation
from ..transfer import GRIP_RAMP_S, MAX_JAW_WIDTH_M, MIN_JAW_WIDTH_M, grasp_tilt_deg

OPEN_MM = 70
OPEN_HOLD_S = .8
GRIP_HOLD_S = 1.
APPROACH_SPEED_M_S = .8
TOOL_CLOSE_HOLD_S = .8       # jaws shut into a tool settle this long before the action starts
#: 0.001 N*m. The default closes firmly on a block; a paper cup needs about 300.
BLOCK_GRIP_EFFORT = 1000
SOFT_GRIP_EFFORT = 300

__all__ = ['OPEN_MM', 'OPEN_HOLD_S', 'GRIP_HOLD_S', 'APPROACH_SPEED_M_S', 'BLOCK_GRIP_EFFORT', 'SOFT_GRIP_EFFORT',
           'EXACT_MM', 'EXACT_DEG', 'GRIP_RAMP_S', 'TOOL_CLOSE_HOLD_S', 'MIN_JAW_WIDTH_M', 'MAX_JAW_WIDTH_M',
           'bearing', 'heading', 'object_extent', 'probe_chain', 'rpy_deg', 'side_grasp_rotation', 'grasp_tilt_deg',
           'down_rotation', 'leaning_rotation', 'jaw_yaw_near', 'contact_point', 'open_event', 'close_event',
           'dwell_event', 'tool_event', 'draft', 'search_candidates', 'unreachable_error']


def down_rotation(jaw_yaw_deg):
    """Tool axis straight down, jaws closing along jaw_yaw_deg in the world XY plane."""
    return Rotation.from_euler('xyz', [180., 0., float(jaw_yaw_deg)], degrees=True).as_matrix()


def leaning_rotation(point, jaw_yaw_deg, tilt_deg, base_xy):
    """Tool leaning tilt_deg back toward base_xy at point: wrist pulled in, fingertips out.

    Upright top-down poses only reach about 0.33 m from a PiPER base; leaning the wrist in extends that to the
    calibrated 0.55 m (see transfer.grasp_tilt_deg).
    """
    from ..robot.arm_model import approach_pose_dir
    direction = np.asarray(point[:2], dtype=float)-np.asarray(base_xy, dtype=float)[:2]
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError('该点与机械臂基座重合，无法确定后仰方向')
    return approach_pose_dir(np.asarray(point, dtype=float), float(jaw_yaw_deg), float(tilt_deg), direction)[:3, :3]


def jaw_yaw_near(axis_deg, preferred_yaw_deg):
    """The 180-degree-equivalent jaw yaw closest to the arm's current wrist yaw: a closing axis has no sign."""
    return float(axis_deg+180*round((preferred_yaw_deg-axis_deg)/180))


def contact_point(found, direction_xy, margin_m=0.):
    """Where a tool moving along direction_xy first touches found: its outline on the upwind side.

    margin_m moves the point further upwind (a stand-off) or into the object (a negative value).
    """
    direction = np.asarray(direction_xy, dtype=float)[:2]
    direction = direction/np.linalg.norm(direction)
    points = np.asarray(found['points'], dtype=float)[:, :2]
    centre = np.asarray(found['center'], dtype=float)[:2]
    behind = float(np.quantile((points-centre) @ direction, .005))
    return centre+direction*(behind-margin_m)


def open_event(s, opening_mm=OPEN_MM, hold_s=OPEN_HOLD_S, ramp_s=None):
    """Open the jaws and wait for the arm to be there. The opening itself is not verified: fully open jaws
    report 62-69 mm depending on what sits between the fingers, and a 65 mm gate stopped a carry
    mid-way over a 2 mm difference."""
    event = {'s': float(s), 'opening_mm': int(opening_mm), 'hold_s': float(hold_s), 'wait_for_arrival': True}
    if ramp_s is not None:
        event['ramp_s'] = float(ramp_s)
    return event


def close_event(s, opening_mm=0, hold_s=GRIP_HOLD_S, effort=None, ramp_s=GRIP_RAMP_S):
    event = {'s': float(s), 'opening_mm': int(opening_mm), 'hold_s': float(hold_s), 'wait_for_arrival': True,
             'verify': 'holding', 'ramp_s': float(ramp_s)}
    if effort is not None:
        event['effort'] = int(effort)
    return event


def tool_event(s, hold_s=TOOL_CLOSE_HOLD_S, effort=BLOCK_GRIP_EFFORT):
    """Shut the empty jaws into one rigid tool - a pusher, a press head, a blade, a stirrer.

    Deliberately not :func:`close_event`: that one carries ``verify='holding'``, and the executor passes a
    holding gate only while the measured opening stays between 1.5 and 67 mm (the real-robot executor). Jaws closed
    on air report 0 mm, so a verified event would stop the whole run with "夹爪疑似空抓或未闭合" three seconds
    after the tool is formed, before it has touched anything. Nothing is held here, so there is nothing to
    verify; the arrival wait, the ramp and the torque limit still apply. The hold also switches compile_arm's
    waypoint refinement on, which keeps long reciprocating paths inside its sample step.
    """
    return {'s': float(s), 'opening_mm': 0, 'hold_s': float(hold_s), 'wait_for_arrival': True,
            'ramp_s': GRIP_RAMP_S, 'effort': int(effort)}


def dwell_event(s, hold_s):
    """Stop at s for hold_s without moving the jaws (a settle, a soak, a contact dwell)."""
    return {'s': float(s), 'hold_s': float(hold_s), 'wait_for_arrival': True}


def draft(points, keyframes, events, speed_m_s, clearance_m, approach_speed_m_s=APPROACH_SPEED_M_S,
          angular_speed_deg_s=None, start_hold_s=None):
    """The arm spec every draft carries: waypoints, orientation keyframes, gripper events and the approach."""
    spec = {'path': {'mode': 'waypoints', 'points': np.asarray(points, dtype=float).tolist()},
            'orientation': {'mode': 'keyframes', 'points': [[float(v) for v in row] for row in keyframes]},
            'speed': float(speed_m_s), 'gripper_events': list(events),
            'approach': {'speed_m_s': float(approach_speed_m_s), 'clearance_m': float(clearance_m)}}
    if angular_speed_deg_s is not None:
        spec['angular_speed_deg_s'] = float(angular_speed_deg_s)
    if start_hold_s is not None:
        spec['start_hold_s'] = float(start_hold_s)
    return spec


def search_candidates(probe, candidates, poses, tolerance, seed=None, exact=(EXACT_MM, EXACT_DEG)):
    """Pick the candidate whose key poses this arm solves best, in the order the arm will fly them.

    candidates is any iterable; poses(candidate) returns [(xyz, rotation matrix), ...]. Each candidate is
    solved as a chain seeded from the previous pose and abandoned at the first pose outside tolerance (a pose
    the solver cannot reach costs seven solves). The first candidate inside exact ends the search.

    Returns (candidate, (position_mm, rotation_deg), joints, ok). Without a probe (the simulation has no
    calibrated model) the first candidate is returned with zero error and ok True.
    """
    listed = list(candidates)
    if not listed:
        raise ValueError('没有可尝试的候选姿态')
    if probe is None:
        return listed[0], (0., 0.), None, True
    best = None
    for candidate in listed:
        errors, joints = probe_chain(probe, poses(candidate), seed, tolerance)
        score = max(errors[0]/tolerance[0], errors[1]/tolerance[1])
        if best is None or score < best[0]:
            best = (score, candidate, errors, joints)
        if errors[0] <= exact[0] and errors[1] <= exact[1]:
            break
    score, candidate, errors, joints = best
    return candidate, errors, joints, score <= 1.


def unreachable_error(what, errors, tolerance, advice='请把目标挪近这只机械臂，或换另一只臂'):
    """The refusal every skill raises when its best candidate misses: what was tried, by how much, what to do."""
    return ValueError(f'{what}不可达：最好的候选误差 {errors[0]:.0f} mm / {errors[1]:.0f}°，'
                      f'超过容差 {tolerance[0]:g} mm / {tolerance[1]:g}°；{advice}')
