"""Tosses of a grasped tabletop object toward a target point: overhand (default) and sidearm.

Overhand: the arm grasps top-down, carries the
object in joint space to a wind-up low in front of its own base with the wrist folded back, settles, then
thrusts forward: its three pitch joints (j2, j3, j4) move together along a straight joint-space line at the
throw profile's rates (``OVERHAND_LINES``: a fast far-reaching line and a nearer, lower one). The whole line
executes at full speed; the jaws open at the instant along it where the tool axis faces the fingertip velocity
and the free flight lands on the target (later releases land farther), triggered by the *measured* arm
reaching that pose; the draft keeps whichever line lands at the higher speed. The lines were found offline
by a linear programme over the reachable planar configurations (see ``OVERHAND_LINES``).

Sidearm: the jaws close along the radial direction from the base, the object is carried onto
a circle of ``SWING_RADIUS_M`` around the base, the arm settles at a wind-up behind the release angle
and swings about its base joint through the release point into a follow-through. The release point is
where the circle's tangent aims at the target, so the jaw opening faces the target; rising along the arc
gives the launch its elevation and the wrist leans back toward the base throughout.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .transfer import (GRIP_RAMP_S, grasp_orientation, leveled_cloud, level_point, locate_object,
                       place_transfer, release_orientation, validate_parameters)

TOSS_STYLES = ('overhand', 'sidearm')
STYLE_LABELS = {'overhand': '伸臂投掷', 'sidearm': '侧摆'}

SWING_RADIUS_M = .45       # circle around the base joint the swing follows
SWING_RADII_M = (.50, .45, .55)   # widest lever first for fingertip speed; the queue falls back to the next when planning fails
MIN_SWING_RADIUS_M = .30
WINDUP_DEG = 30.           # swing angle before the release point
FOLLOW_DEG = 20.           # swing angle after it
ARC_STEP_DEG = 5.
ARC_SAMPLES = int(round((WINDUP_DEG+FOLLOW_DEG)/ARC_STEP_DEG))
RELEASE_INDEX = int(round(WINDUP_DEG/ARC_STEP_DEG))   # arc sample that is the release point
WINDUP_HEIGHT_M = .10
FOLLOW_HEIGHT_M = .20      # the arc rises linearly from the wind-up to the follow-through
SWING_TILT_DEG = 30.       # lean back toward the base through the swing
DEFAULT_TOSS_SPEED_M_S = 1.5
DEFAULT_LEAD_S = .08
SWING_ACCEL_M_S2 = 8.      # Cartesian acceleration envelope of the draft
SWING_ANGULAR_SPEED_DEG_S = 360.   # the jaw yaw turns with the base joint during the swing
MIN_THROW_DISTANCE_M = .12
MIN_TOSS_SPEED_M_S = .3
MAX_TOSS_SPEED_M_S = 5.
MAX_LEAD_S = .5
WINDUP_SPEED_M_S = .05
WINDUP_HOLD_S = .2         # settle at the wind-up so the swing starts from a measured stop
SWING_MAX_BEARING_DEG = 85.   # the whole arc stays in front of the base (which faces +X), clear of the head camera mast

# Overhand throw lines: q(t) = release_joints + rates * t for t in [-windup_s, +follow_s], j1 = target bearing.
# Found offline on the PiPER-X kinematics: a linear programme per reachable planar
# configuration maximises the fingertip speed along a forward direction 5 deg up (velocity within 8 deg of it,
# tool axis within 30 deg) with j2/j3/j4 capped at 150 deg/s, 5 deg inside their limits, and the whole line
# feasible (limits, table, self collision). j2/j3/j4 are the three parallel pitch joints (j5 yaws, j6 rolls).
# "fast": fingertip at most 0.45 m ahead / 0.12-0.30 m up at the release configuration; the fingertip runs from
#         0.13 m ahead / 0.10 m up (wind-up, wrist folded back) to 0.56 m / 0.30 m at 0.85-0.95 m/s. With the jaws
#         facing the flight (OVERHAND_TOOL_CONE_DEG) it lands 0.62-0.77 m ahead of the base at full speed.
# "near": fingertip at most 0.40 m ahead / 0.10-0.25 m up and the wrist snapping forward at >= 100 deg/s; slower
#         (0.7-0.8 m/s) and lower (0.26 m at most) but its facing releases land 0.52-0.65 m ahead.
# The draft tries every line and keeps the one that lands on the target at the highest release speed.
# Both lines stay feasible to follow_s past the release configuration; the stroke itself stops
# OVERHAND_FOLLOW_AFTER_RELEASE_S after the chosen release instant (see there).
OVERHAND_LINES = ({'name': 'fast', 'release_joints_deg': np.array([0., 70., -55., 10., 0., 0.]),
                   'rates_deg_s': np.array([0., 150., -84., -150., 0., 0.]), 'windup_s': .40, 'follow_s': .46},
                  {'name': 'near', 'release_joints_deg': np.array([0., 60., -40., 5., 0., 0.]),
                   'rates_deg_s': np.array([0., 150., -67., -150., 0., 0.]), 'windup_s': .34, 'follow_s': .46})
OVERHAND_STEP_S = .02
OVERHAND_ACCELERATION_S = .10             # earliest release: this long after the wind-up
# The firmware follows a joint reference at about 150 deg/s at most (measured: throws commanded at 148 deg/s
# reached 145-153 deg/s on j2/j4; throws commanded at 228 deg/s were no faster, and because j2/j4 saturated
# while j3 did not the arm left the joint-space line: 15 deg off the release pose instead of 8, the fingertip
# leaving at 40 deg up instead of 22, and the throws were visibly worse). Both lines already drive
# j2/j4 at that ceiling, so toss_speed_m_s can only slow a stroke down, never speed it up.
OVERHAND_JOINT_SPEED_CEILING_DEG_S = 150.
OVERHAND_RATE_SCALE = (.3, 1.0)           # toss_speed_m_s scales the joint rates within this range of the design
# The reference keeps running this long past the release instant, then stops: the arm trails it by about 0.26 s
# at full rate and must still find it moving when it reaches the release pose (with 0.2 s the arm arrived while
# already braking), but running on to the end of the line (0.34 s more, j2 to 129 deg and the fingertip 0.64 m
# ahead at 0.16 m) read as the arm pushing forward after the object had left rather than flicking it.
# The arm trails the 50 Hz reference by a quarter second at full rate (measured: j2/j4
# 37-41 deg behind while moving at the reference's 150 deg/s), so it passes the release pose that long after the
# reference does. The reference therefore keeps running at full speed for the lag plus a launch window past the
# release instant, and the release instant is chosen no later than that before the end of the line: when the
# target was out of reach the release sat 0.20 s before the line's end, the reference stopped while the arm
# was still short of the release pose, the jaws opened as the arm braked and the block dropped at its feet.
OVERHAND_REFERENCE_LAG_S = .25
OVERHAND_LAUNCH_S = .10
OVERHAND_FOLLOW_AFTER_RELEASE_S = OVERHAND_REFERENCE_LAG_S+OVERHAND_LAUNCH_S
OVERHAND_CARRY_SPEED_M_S = .40            # hover -> wind-up carry (a joint-space line, object held)
TOSS_APPROACH_SPEED_M_S = .12             # automatic approach from the current pose to the hover
OVERHAND_LANDING_TOLERANCE_M = .02
OVERHAND_MIN_FLIGHT_M = .03
OVERHAND_TOOL_CONE_DEG = 45.              # release only where the tool axis is this close to the fingertip velocity
OVERHAND_CARRY_STEP_DEG = 2.5        # joint-space sampling of the carry from the hover to the wind-up
OVERHAND_CARRY_FLOOR_M = .03         # the carried object stays at least this far above where it was grasped
G_M_S2 = 9.81


def width_along(points_xy, angle_deg):
    """Visible extent of an object across the closing axis at ``angle_deg``, padded like object_geometry."""
    axis = np.array([np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))])
    low, high = np.quantile(np.asarray(points_xy, dtype=float) @ axis, [.005, .995])
    return float(high-low+.003)


def swing_geometry(base_xy, target_xy, block_xy, radii=SWING_RADII_M):
    """Release angle on the swing circle whose tangent aims at the target, the swing sense and the radius.

    Both senses (counter-clockwise +1, clockwise -1) have such a point. The whole arc, wind-up to
    follow-through, must stay within ``SWING_MAX_BEARING_DEG`` of the base's forward (+X) direction,
    which keeps the base joint inside ±90 degrees. The smallest radius in ``radii``
    that admits an arc wins (a wider circle moves the release angle toward the target bearing);
    between its senses, the wind-up closer to the object's own bearing needs the shorter carry.
    """
    base = np.asarray(base_xy, dtype=float); target = np.asarray(target_xy, dtype=float)
    block = np.asarray(block_xy, dtype=float)
    d = target-base
    distance = float(np.linalg.norm(d))
    if min(radii[0], distance-.05) < MIN_SWING_RADIUS_M:
        raise ValueError(f'抛掷目标离机械臂基座只有 {distance*100:.0f} cm，横向甩臂半径不足')
    bearing = float(np.arctan2(d[1], d[0]))
    block_bearing = float(np.arctan2(block[1]-base[1], block[0]-base[0]))
    for radius in radii:
        r = min(radius, distance-.05)
        gap = float(np.arccos(r/distance))
        best = None
        for sense in (1., -1.):
            release = bearing-sense*gap
            windup = release-sense*np.deg2rad(WINDUP_DEG)
            follow = release+sense*np.deg2rad(FOLLOW_DEG)
            swept = max(abs(float(np.degrees(np.angle(np.exp(1j*angle))))) for angle in (windup, release, follow))
            if swept > SWING_MAX_BEARING_DEG:
                continue
            carry = abs(float(np.angle(np.exp(1j*(windup-block_bearing)))))
            if best is None or carry < best[0]:
                best = (carry, release, sense)
        if best is not None:
            return best[1], best[2], r
    raise ValueError(f'抛掷弧线会让基座关节转过 ±{SWING_MAX_BEARING_DEG:.0f}°，请把落点或物体放到机械臂正前方')


def ballistic_speed(distance_m, drop_m, elevation_deg):
    """Launch speed that lands ``distance_m`` ahead after descending ``drop_m`` (positive = landing lower) at ``elevation_deg``."""
    elevation = np.deg2rad(elevation_deg)
    denominator = 2*np.cos(elevation)**2*(drop_m+distance_m*np.tan(elevation))
    if distance_m <= 0 or denominator <= 0:
        raise ValueError('落点比这条抛物线能到的位置还高，请把落点放低或放远')
    return float(distance_m*np.sqrt(G_M_S2/denominator))


def flight_distance(tip, velocity, landing_z):
    """Horizontal distance an object released at ``tip`` with ``velocity`` travels before reaching ``landing_z``."""
    drop = max(float(tip[2]-landing_z), 0.)
    vertical = float(velocity[2])
    duration = (vertical+np.sqrt(vertical**2+2*G_M_S2*drop))/G_M_S2
    return float(np.linalg.norm(velocity[:2])*duration)


def line_joints(line, times):
    """Joints along ``line`` at its design rates: q(t) = release + rates * max(t, -windup_s), t = 0 at the
    release configuration."""
    times = np.asarray(times, dtype=float)[:, None]
    return line['release_joints_deg']+line['rates_deg_s']*np.maximum(times, -line['windup_s'])


def overhand_line(fk, base_joint_deg, times, line=OVERHAND_LINES[0]):
    """World tool poses and joints along ``line`` at its design rates for the given ``times``."""
    joints = line_joints(line, times)
    joints[:, 0] = base_joint_deg
    return np.asarray([np.asarray(fk(q), dtype=float) for q in joints]).reshape(-1, 4, 4), joints


def overhand_samples(fk, base_joint_deg, release_time_s=0., line=OVERHAND_LINES[0]):
    """Throw samples every ``OVERHAND_STEP_S`` from the wind-up to ``OVERHAND_FOLLOW_AFTER_RELEASE_S`` past the
    release instant (never beyond the line's ``follow_s``), plus the release instant itself; returns poses,
    joints, times, release index."""
    end = min(line['follow_s'], release_time_s+OVERHAND_FOLLOW_AFTER_RELEASE_S)   # equal for any instant in the window
    grid = np.arange(-line['windup_s'], end+1e-9, OVERHAND_STEP_S)
    times = np.unique(np.round(np.r_[grid, release_time_s], 6))
    release = int(np.argmin(np.abs(times-release_time_s)))
    poses, joints = overhand_line(fk, base_joint_deg, times, line)
    return poses, joints, times, release


def release_window_s(line):
    """Release instants: after the acceleration and early enough for the lagging arm to pass the release pose at
    full speed before the line ends."""
    return -line['windup_s']+OVERHAND_ACCELERATION_S, line['follow_s']-OVERHAND_FOLLOW_AFTER_RELEASE_S


def rpy_deg(pose):
    return Rotation.from_matrix(np.asarray(pose)[:3, :3]).as_euler('xyz', degrees=True).tolist()


def overhand_release(fk, base_joint_deg, base, target, toss_speed_m_s, line=OVERHAND_LINES[0]):
    """Pick the release instant on ``line`` whose full-speed flight lands on ``target``.

    The joint rates are scaled so the fastest fingertip speed along the stroke is ``toss_speed_m_s`` (within
    ``OVERHAND_RATE_SCALE`` of the design). Only instants where the tool axis is within
    ``OVERHAND_TOOL_CONE_DEG`` of the fingertip velocity count (the jaw opening faces the flight). Later releases
    land farther; a target beyond the latest release's reach is thrown at from there (the shortfall is
    reported), one nearer than the earliest release's reach is thrown at from there with the rates scaled down
    to the ballistic speed. Returns ``(release_time_s, rate_scale, shortfall_m, release_speed_m_s)``.
    """
    start, end = release_window_s(line)
    times = np.arange(start, end+1e-9, .005)
    poses, _ = overhand_line(fk, base_joint_deg, times, line)
    tips = poses[:, :3, 3]
    velocities = np.gradient(tips, times, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    scale = float(np.clip(toss_speed_m_s/speeds.max(), *OVERHAND_RATE_SCALE))
    tool_off = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', poses[:, :3, 2], velocities)/speeds, -1., 1.)))
    facing = tool_off <= OVERHAND_TOOL_CONE_DEG
    if not facing.any():
        raise ValueError(f'抛掷线 {line["name"]} 上没有夹爪开口朝向飞行方向的放手瞬间')
    times, tips, velocities, speeds = times[facing], tips[facing], velocities[facing], speeds[facing]
    distance = float(np.linalg.norm(target[:2]-base))
    reach = np.linalg.norm(tips[:, :2]-base, axis=1)
    landing = reach+np.array([flight_distance(tip, v*scale, target[2]) for tip, v in zip(tips, velocities)])
    i = int(np.argmin(np.abs(landing-distance)))
    if landing[i] > distance+OVERHAND_LANDING_TOLERANCE_M:
        i = int(np.argmin(reach))          # the earliest facing instant: nearest release point, then slow the stroke down
        flight = distance-float(reach[i])
        if flight < OVERHAND_MIN_FLIGHT_M:
            raise ValueError(f'落点离机械臂基座只有 {distance*100:.0f} cm，伸臂投掷最早在基座前 {reach[i]*100:.0f} cm 处放手也会扔过头，'
                             f'请把落点放远一些或改用侧摆')
        velocity = velocities[i]
        elevation = float(np.degrees(np.arctan2(velocity[2], np.linalg.norm(velocity[:2]))))
        needed = ballistic_speed(flight, float(tips[i][2]-target[2]), elevation)
        scale = needed/float(speeds[i])
        if scale < OVERHAND_RATE_SCALE[0]:
            raise ValueError(f'落点离机械臂基座只有 {distance*100:.0f} cm，伸臂投掷最早在基座前 {reach[i]*100:.0f} cm 处放手也会扔过头，'
                             f'请把落点放远一些或改用侧摆')
        return float(times[i]), float(scale), 0., float(speeds[i]*scale)
    # Within the landing tolerance the throw is on target: a 1 mm miss must not outrank a faster line.
    shortfall = 0. if abs(float(landing[i])-distance) <= OVERHAND_LANDING_TOLERANCE_M else max(0., distance-float(landing[i]))
    return float(times[i]), scale, float(shortfall), float(speeds[i]*scale)


def solve_hover_approach(grasp, hover, grasping, yaw, base, seed, windup, probe, tolerance):
    """Keep the exact grasp pose; search hover tilt and initial IK branches when needed."""
    seeds = [np.asarray(seed, dtype=float), np.asarray(windup, dtype=float), np.zeros(6)]
    poses = [grasping] + [release_orientation(hover, yaw, tilt, base) for tilt in (15., 30., 45.)]
    best = (float('inf'), float('inf'))
    for pose_index, pose in enumerate(poses):
        rotation = Rotation.from_euler('xyz', pose, degrees=True)
        for seed_index, initial in enumerate(seeds):
            mm, deg, q = probe(hover, rotation.as_matrix(), initial)
            if mm/max(tolerance[0],1e-9)+deg/max(tolerance[1], 1e-9) < best[0]/max(tolerance[0],1e-9)+best[1]/max(tolerance[1],1e-9):
                best = (mm, deg)
            if q is None or not np.isfinite([mm, deg]).all() or mm > tolerance[0] or deg > tolerance[1]:
                continue
            if pose_index or seed_index:
                # Candidate hover poses must also connect to the unchanged grasp pose.
                interpolator = Slerp([0., 1.], Rotation.from_euler('xyz', [pose, grasping], degrees=True))
                previous = np.asarray(q, dtype=float)
                valid = True
                for fraction in np.linspace(0., 1., 6)[1:]:
                    xyz = np.asarray(hover)*(1-fraction)+np.asarray(grasp)*fraction
                    pe, re, next_q = probe(xyz, interpolator(fraction).as_matrix(), previous)
                    if next_q is None or not np.isfinite([pe,re]).all() or pe > tolerance[0] or re > tolerance[1]:
                        valid = False; break
                    previous = np.asarray(next_q, dtype=float)
                if not valid:
                    continue
            return pose, np.asarray(q, dtype=float)
    raise ValueError(f'抓取接近姿态搜索未通过（最佳悬停误差 {best[0]:.0f} mm / {best[1]:.0f}°）；'
                     '已尝试多个初始解和悬停倾角，请调整抓取线方向或位置')


def overhand_draft(found, placement, target, table_z, clearance_m, speed_m_s, preferred_yaw_deg,
                   toss_speed_m_s, lead_s, base, fk, reach_probe, seed_joints, pose_tolerance=(30., 60.),
                   lines=OVERHAND_LINES, carry_speed_m_s=OVERHAND_CARRY_SPEED_M_S,
                   windup_speed_m_s=WINDUP_SPEED_M_S, grasp_hold_s=1.):
    """Overhand throw draft along the best of ``lines``.

    ``fk`` maps six joint angles (degrees) to the world TCP pose; ``reach_probe(xyz, rotation, seed)`` returns
    ``(position_mm, rotation_deg, joints)`` for one pose. From the hover above the grasp the carry to the
    wind-up is a joint-space line sampled every ``OVERHAND_CARRY_STEP_DEG``: holding the tool vertical while
    climbing pins j4 at its limit (a Cartesian climb failed 0.3 m up; even a 5 cm lift put j4 at 89 deg and
    the IK jumped branches on the way), and a joint-space line between two reachable configurations stays
    reachable by construction. The throw itself runs the whole line at full speed and lets go at the instant
    :func:`overhand_release` picks for the target distance.
    """
    grasp, hover = placement['grasp'], placement['hover']
    reach = target[:2]-base
    distance = float(np.linalg.norm(reach))
    bearing = float(np.degrees(np.arctan2(reach[1], reach[0])))
    forward = np.asarray(fk(lines[0]['release_joints_deg']), dtype=float)[:2, 3]-base   # base +X in world at j1 = 0
    base_joint = bearing-float(np.degrees(np.arctan2(forward[1], forward[0])))
    choices, refusals = [], []
    for line in lines:
        try:
            choices.append((line, *overhand_release(fk, base_joint, base, target, toss_speed_m_s, line)))
        except ValueError as exc:
            refusals.append(str(exc))
    if not choices:
        raise ValueError(refusals[0])
    # The line that lands on the target at the highest release speed; failing that, the smallest shortfall.
    line, release_time, scale, shortfall, _ = min(choices, key=lambda c: (c[3] > 0, c[3] if c[3] > 0 else -c[4]))
    poses, joints, times, release_index = overhand_samples(fk, base_joint, release_time, line)
    tips = poses[:, :3, 3]
    velocities = np.gradient(tips, times, axis=0)*scale
    speeds = np.linalg.norm(velocities, axis=1)
    release = tips[release_index]
    release_velocity = velocities[release_index]
    flight = flight_distance(release, release_velocity, target[2])
    tool_off = float(np.degrees(np.arccos(np.clip(poses[release_index][:3, 2] @ release_velocity/speeds[release_index], -1., 1.))))
    yaw = found['axis_deg']
    yaw += 180*round((preferred_yaw_deg-yaw)/180)
    grasping = grasp_orientation(grasp, yaw, base)
    hovering, hover_joints = solve_hover_approach(grasp, hover, grasping, yaw, base,
                                                 seed_joints, joints[0], reach_probe, pose_tolerance)
    delta = joints[0]-hover_joints
    steps = max(1, int(np.ceil(np.abs(delta).max()/OVERHAND_CARRY_STEP_DEG)))
    carry_joints = hover_joints+delta*np.linspace(0., 1., steps+1)[1:-1, None]
    carry_poses = np.asarray([np.asarray(fk(q), dtype=float) for q in carry_joints]).reshape(-1, 4, 4)
    carry_tips = carry_poses[:, :3, 3]
    if len(carry_tips) and carry_tips[:, 2].min() < grasp[2]+OVERHAND_CARRY_FLOOR_M:
        raise ValueError(f'从悬停点到蓄力点的关节空间搬运会把物体压回到抓取高度上方 {(carry_tips[:, 2].min()-grasp[2])*100:.0f} cm，请改用侧摆')
    k = len(carry_tips)
    points = np.vstack([hover, grasp, hover, carry_tips, tips])   # tips[0] is the wind-up
    n = len(points)-1
    s_hover, s_windup, s_release = 2/n, (3+k)/n, (3+k+release_index)/n
    approach_keys = ([[0., *grasping], [s_hover, *grasping]] if np.allclose(hovering, grasping) else
                     [[0., *hovering], [1/n, *grasping], [s_hover, *hovering]])
    keyframes = approach_keys+[[(3+i)/n, *rpy_deg(pose)] for i, pose in enumerate(carry_poses)]+ \
                [[(3+k+i)/n, *rpy_deg(pose)] for i, pose in enumerate(poses)]
    speed = [[0., speed_m_s], [s_hover, speed_m_s]]+([[3/n, carry_speed_m_s], [(2+k)/n, carry_speed_m_s]] if k else [])+ \
            [[s_windup, windup_speed_m_s]]+[[(3+k+i)/n, float(speeds[i])] for i in range(1, len(tips))]
    events = [{'s': 0., 'opening_mm': 70, 'hold_s': min(.8, grasp_hold_s), 'wait_for_arrival': True},
              {'s': 1/n, 'opening_mm': 0, 'hold_s': grasp_hold_s, 'wait_for_arrival': True, 'verify': 'holding', 'ramp_s': GRIP_RAMP_S},
              {'s': s_windup, 'hold_s': WINDUP_HOLD_S, 'wait_for_arrival': True},
              {'s': s_release, 'opening_mm': 70, 'lead_s': lead_s, 'on_measured': True, 'ramp_s': 0.}]
    return {'toss_style': 'overhand', 'source_xyz': grasp.tolist(), 'target_xyz': release.tolist(),
            'toss_target_xyz': target.tolist(), 'windup_xyz': tips[0].tolist(),
            'throw_direction_xy': (reach/distance).tolist(), 'throw_distance_m': flight, 'landing_shortfall_m': shortfall,
            'drop_height_m': float(release[2]-target[2]),
            'release_elevation_deg': float(np.degrees(np.arctan2(release_velocity[2], np.linalg.norm(release_velocity[:2])))),
            'tool_off_deg': tool_off, 'release_time_s': release_time, 'rate_scale': scale, 'throw_line': line['name'],
            'toss_speed_m_s': float(speeds[release_index]), 'lead_s': lead_s, 'throw_times_s': times.tolist(),
            'release_joints_deg': joints[release_index].tolist(), 'hover_joints_deg': hover_joints.tolist(), 'carry_samples': k,
            'arm': {'path': {'mode': 'waypoints', 'points': points.tolist()},
                    'orientation': {'mode': 'keyframes', 'points': keyframes},
                    'speed': speed, 'accel_m_s2': SWING_ACCEL_M_S2, 'angular_speed_deg_s': SWING_ANGULAR_SPEED_DEG_S,
                    'gripper_events': events, 'approach': {'speed_m_s': TOSS_APPROACH_SPEED_M_S, 'clearance_m': clearance_m}}}


def toss_between(frame, object_pixel, target_xyz, table_z=0., clearance_m=.12, speed_m_s=.10,
                 preferred_yaw_deg=0., fingertip_bias_mm=0., toss_speed_m_s=DEFAULT_TOSS_SPEED_M_S, lead_s=DEFAULT_LEAD_S,
                 base_xy=None, radii=SWING_RADII_M, style='overhand', fk=None, reach_probe=None, seed_joints=None,
                 pose_tolerance=(30., 60.), grasp_pixels=None, grasp_options=None,
                 carry_speed_m_s=OVERHAND_CARRY_SPEED_M_S, windup_speed_m_s=WINDUP_SPEED_M_S, grasp_hold_s=1.):
    """Draft that grasps the object under ``object_pixel`` and tosses it toward ``target_xyz``.

    ``style`` is ``overhand`` (needs ``fk``, ``reach_probe`` and ``seed_joints``: the arm's
    kinematics, see :func:`overhand_draft`) or ``sidearm``.
    """
    for value,lo,hi in [(carry_speed_m_s,.01,1.),(windup_speed_m_s,.01,.3),(grasp_hold_s,.5,2.)]:
        if not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError('阶段速度或夹持停留时间超出范围')
    if style not in TOSS_STYLES:
        raise ValueError('抛法只接受 overhand（伸臂投掷）或 sidearm（侧摆）')
    if style == 'overhand' and not (callable(fk) and callable(reach_probe) and seed_joints is not None):
        raise ValueError(f'{STYLE_LABELS[style]}需要这只机械臂的运动学模型；仿真后端只能侧摆')
    pixel = np.asarray(object_pixel, dtype=float)
    h, w = frame.depth.shape
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or np.any(pixel < 0) or pixel[0] >= w or pixel[1] >= h:
        raise ValueError('物体像素需要落在相机画面内')
    target = np.asarray(target_xyz, dtype=float) if isinstance(target_xyz, (list, tuple, np.ndarray)) else None
    if target is None or target.shape != (3,) or not np.isfinite(target).all() or np.any(np.abs(target) > 2):
        raise ValueError('抛掷目标需要三个有限的世界坐标（米），绝对值不超过 2 m')
    validate_parameters(table_z, clearance_m, speed_m_s, preferred_yaw_deg, fingertip_bias_mm)
    if not np.isfinite([toss_speed_m_s, lead_s]).all() or not MIN_TOSS_SPEED_M_S <= toss_speed_m_s <= MAX_TOSS_SPEED_M_S \
            or not 0 <= lead_s <= MAX_LEAD_S:
        raise ValueError(f'抛掷速度需为 {MIN_TOSS_SPEED_M_S}–{MAX_TOSS_SPEED_M_S} m/s，提前量 0–{MAX_LEAD_S} s')
    if base_xy is None:
        raise ValueError('抛掷需要机械臂基座位置')
    base = np.asarray(base_xy, dtype=float)[:2]
    cloud, plane = leveled_cloud(frame, table_z)
    if grasp_pixels is None:
        found = locate_object(frame, pixel, table_z, preferred_yaw_deg, cloud)
    else:
        from .grasp import line_object
        found = line_object(frame, grasp_pixels, clearance_m=clearance_m, speed_m_s=speed_m_s,
                            preferred_yaw_deg=preferred_yaw_deg, **(grasp_options or {}))
    target = level_point(plane, table_z, target)
    placement = place_transfer(found, target, table_z, clearance_m, fingertip_bias_mm, drop=True)
    if grasp_pixels is not None:
        placement.update(grasp=found['manual_grasp'], hover=found['manual_hover'])
    grasp, hover = placement['grasp'], placement['hover']
    if np.linalg.norm(target[:2]-grasp[:2]) < MIN_THROW_DISTANCE_M:
        raise ValueError(f'抛掷目标离物体只有 {np.linalg.norm(target[:2]-grasp[:2])*100:.0f} cm，请选更远的目标')
    common = {'observation_id': frame.id, 'object_width_mm': found['width']*1000,
              'object_height_mm': (found['top']-table_z)*1000, 'placement': 'toss'}
    if style == 'overhand':
        return {**common, **overhand_draft(found, placement, target, table_z, clearance_m, speed_m_s, preferred_yaw_deg,
                                           toss_speed_m_s, lead_s, base, fk, reach_probe, seed_joints, pose_tolerance,
                                           carry_speed_m_s=carry_speed_m_s,windup_speed_m_s=windup_speed_m_s,grasp_hold_s=grasp_hold_s)}
    # The jaws close along the base's radial direction so the opening faces the tangent, i.e. the target.
    block_bearing = float(np.degrees(np.arctan2(grasp[1]-base[1], grasp[0]-base[0])))
    jaw_width = found['width'] if grasp_pixels is not None else width_along(found['points'][:, :2], block_bearing)   # reported, not a gate
    release_angle, sense, radius = swing_geometry(base, target[:2], grasp[:2], radii)
    closing_axis = found['axis_deg'] if grasp_pixels is not None else block_bearing
    grasp_yaw = closing_axis+180*round((preferred_yaw_deg-closing_axis)/180)
    fold = 180*round((preferred_yaw_deg-block_bearing)/180)
    windup_angle = release_angle-sense*np.deg2rad(WINDUP_DEG)
    # Unwrap the swing angles next to the object's bearing so the jaw yaw stays continuous.
    windup_angle = np.deg2rad(block_bearing)+float(np.angle(np.exp(1j*(windup_angle-np.deg2rad(block_bearing)))))
    angles = windup_angle+sense*np.deg2rad(ARC_STEP_DEG)*np.arange(0, ARC_SAMPLES+1)
    heights = table_z+WINDUP_HEIGHT_M+(FOLLOW_HEIGHT_M-WINDUP_HEIGHT_M)*np.arange(0, ARC_SAMPLES+1)/ARC_SAMPLES
    arc = np.c_[base[0]+radius*np.cos(angles), base[1]+radius*np.sin(angles), heights]
    points = np.vstack([hover, grasp, hover, arc])          # arc[0] is the wind-up
    n = len(points)-1
    s_windup, s_release = 3/n, (3+RELEASE_INDEX)/n
    release = arc[RELEASE_INDEX]
    tangent = sense*np.array([-np.sin(angles[RELEASE_INDEX]), np.cos(angles[RELEASE_INDEX])])
    grasping = grasp_orientation(grasp, grasp_yaw, base)
    keyframes = [[0, *grasping], [2/n, *grasping]]
    for i, (point, angle) in enumerate(zip(arc, angles)):
        yaw = float(np.degrees(angle))+fold
        keyframes.append([(3+i)/n, *release_orientation(point, yaw, SWING_TILT_DEG, base)])
    speed = [[0., speed_m_s], [1/n, speed_m_s], [2/n, carry_speed_m_s],
             [s_windup, windup_speed_m_s], [s_release, toss_speed_m_s], [1., toss_speed_m_s/2]]
    events = [{'s': 0., 'opening_mm': 70, 'hold_s': min(.8, grasp_hold_s), 'wait_for_arrival': True},
              {'s': 1/n, 'opening_mm': 0, 'hold_s': grasp_hold_s, 'wait_for_arrival': True, 'verify': 'holding', 'ramp_s': GRIP_RAMP_S},
              {'s': s_windup, 'hold_s': WINDUP_HOLD_S, 'wait_for_arrival': True},
              {'s': s_release, 'opening_mm': 70, 'lead_s': lead_s, 'on_measured': True, 'ramp_s': 0.}]
    return {**common, 'toss_style': 'sidearm', 'object_width_mm': jaw_width*1000, 'source_xyz': grasp.tolist(), 'target_xyz': release.tolist(), 'toss_target_xyz': target.tolist(),
            'throw_direction_xy': tangent.tolist(), 'throw_distance_m': float(np.linalg.norm(target[:2]-release[:2])),
            'toss_speed_m_s': toss_speed_m_s, 'lead_s': lead_s, 'swing_tilt_deg': SWING_TILT_DEG,
            'swing_radius_m': radius, 'swing_sense': sense,
            'arm': {'path': {'mode': 'waypoints', 'points': points.tolist()},
                    'orientation': {'mode': 'keyframes', 'points': keyframes},
                    'speed': speed, 'accel_m_s2': SWING_ACCEL_M_S2, 'angular_speed_deg_s': SWING_ANGULAR_SPEED_DEG_S,
                    'gripper_events': events, 'approach': {'speed_m_s': TOSS_APPROACH_SPEED_M_S, 'clearance_m': clearance_m}}}
