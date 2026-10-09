"""Side grasp of a cup and a pour into a target container from one screen-space intent stroke."""
import numpy as np
from scipy.spatial.transform import Rotation
from .transfer import GRIP_RAMP_S, leveled_cloud, locate_container, locate_object, rim_circle

MIN_CUP_HEIGHT_M = .050
PREGRASP_STANDOFF_M = .060
LIFT_M = .080
POUR_MOUTH_CLEARANCE_M = .100   # lowest point of the tilted cup's rim above the target rim (6 cm read as too low)
CUP_GRIP_EFFORT = 300           # 0.001 N*m: the block-grasping default (1000) flattens a paper cup
MIN_POUR_DISTANCE_M = .030
OPEN_MM = 70
OPEN_HOLD_S = .8
GRIP_HOLD_S = 1.
#: Approach directions to try, as offsets from the base→cup radial, straightest first: the operator reads a
#: sideways approach as a strange grasp, so ±30° is only taken when the radial approach is not exact.
APPROACH_OFFSETS_DEG = (0., -30., 30., -60., 60.)
#: Tool pitch candidates for the side grasp (nose down, about the closing axis), level first. With the tool
#: level no direction solved exactly for cups 0.29-0.44 m from the base (radial 30-51 mm, ±30° 10-24 mm; the
#: 10-20 mm residuals pushed the cup out of the 2 mm jaw clearance and every grasp closed on air);
#: 15° down made the radial approach exact at 0.44 m and ±30° exact at 0.29 m.
SIDE_GRASP_PITCH_DEG = (0., 15., 30.)
#: A round cup's rim circle (radius within this range) gives its centre and diameter; the visible extent across
#: the jaws is widened 1-2 cm by the depth halo around a white cup.
CUP_RIM_RADIUS_M = (.025, .060)
#: Directions the cup mouth may tip toward, as offsets from the base→target radial. Tipping sideways (±90°) is a
#: roll about the horizontal tool axis that j6 can make; straight outward (0°) needs a 100° wrist pitch that
#: j5 (±89°) cannot quite reach; back toward the base is never solvable, so it is not offered.
POUR_OFFSETS_DEG = (90., -90., 45., -45., 0.)
#: Key-pose errors at or below these count as exact and end the candidate search early.
EXACT_MM = 3.
EXACT_DEG = 3.


def stroke_pixels(frame, pixels):
    """Validate the intent stroke: 2–4096 finite pixels inside the frame with distinct ends."""
    ink = np.asarray(pixels, dtype=float)
    if ink.ndim != 2 or ink.shape[1] != 2 or not 2 <= len(ink) <= 4096 or not np.isfinite(ink).all():
        raise ValueError('请从杯子画到目标容器：需要 2–4096 个有效像素点')
    h, w = frame.depth.shape
    if np.any(ink < 0) or np.any(ink[:, 0] >= w) or np.any(ink[:, 1] >= h):
        raise ValueError('笔迹超出相机画面')
    if np.linalg.norm(ink[-1] - ink[0]) < 12:
        raise ValueError('请从杯子画到另一个容器')
    return ink


def object_extent(points_xy, axis):
    """Visible extent of ``points_xy`` along the unit ``axis``, with the same 3 mm rim margin as locate_object."""
    projected = points_xy @ axis
    low, high = np.quantile(projected, [.005, .995])
    return float(high - low + .003)


def side_grasp_rotation(direction_xy, pitch_deg=0.):
    """Tool frame for a side grasp along ``direction_xy``: Z along +dir, X closing across it, Y down, the tool
    axis pitched ``pitch_deg`` nose-down about the closing axis (the jaws stay horizontal, the cup stays upright).

    Y down is the wrist's natural roll for a horizontal tool (FK at j2=90°, j3=−90°, others 0: Z=+x, X=−y,
    Y=−z). The mirrored frame with Y up needs a 180° roll about the tool axis that j6 (±120°) cannot make:
    on the real arm every reach solved it exactly 60° short, which is why the jaw yaw here is the approach
    bearing minus 90°.
    """
    from .robot.arm_model import approach_pose_dir
    jaw_yaw_deg = float(np.rad2deg(np.arctan2(direction_xy[1], direction_xy[0]))) - 90.
    pose = approach_pose_dir(np.zeros(3), jaw_yaw_deg, 90., np.asarray(direction_xy, dtype=float))
    rotation = pose[:3, :3]
    if pitch_deg:
        rotation = rotation @ Rotation.from_euler('x', -float(pitch_deg), degrees=True).as_matrix()
    return rotation


def tilted_rotation(upright, pour_direction_xy, tilt_deg):
    """Tip the cup mouth ``tilt_deg`` toward ``pour_direction_xy`` about the world-horizontal axis through the TCP."""
    axis = np.cross([0., 0., 1.], [*pour_direction_xy, 0.])
    return Rotation.from_rotvec(axis * np.deg2rad(tilt_deg)).as_matrix() @ upright


def heading(bearing_deg):
    return np.array([np.cos(np.deg2rad(bearing_deg)), np.sin(np.deg2rad(bearing_deg))])


def bearing(xy):
    return float(np.rad2deg(np.arctan2(xy[1], xy[0])))


def approach_candidates(cup_xy, base_xy):
    """Approach directions as ``(offset_deg, unit_xy)`` in preference order around the base→cup radial."""
    radial = bearing(np.asarray(cup_xy, dtype=float) - np.asarray(base_xy, dtype=float))
    return [(offset, heading(radial + offset)) for offset in APPROACH_OFFSETS_DEG]


def pour_height(target_top, cup_height, grasp_fraction, radius, tilt_deg, clearance_m):
    """TCP height at which the lowest point of the tilted cup's rim clears ``target_top`` by ``clearance_m``.

    The mouth centre sits ``cup_height * (1 - grasp_fraction)`` above the grasp point along the cup axis; after
    the tilt that is ``cos(tilt)`` of it higher (negative past 90 degrees) and the low side of the rim hangs
    ``radius * sin(tilt)`` below the mouth centre. A fixed "0.6 cup heights + 3 cm" left only 2 cm over a real tin.
    """
    above = cup_height*(1.-grasp_fraction)
    tilt = np.deg2rad(tilt_deg)
    return target_top+clearance_m-above*np.cos(tilt)+radius*np.sin(tilt)


def pour_candidates(cup_xy, target_xy, base_xy):
    """Mouth directions as ``(offset_deg, unit_xy)`` around the base→target radial, closest to the drawn cup→target first."""
    drawn = np.asarray(target_xy, dtype=float) - np.asarray(cup_xy, dtype=float)
    distance = float(np.linalg.norm(drawn))
    if distance < MIN_POUR_DISTANCE_M:
        raise ValueError('目标容器离杯子太近，请画到另一个容器')
    drawn /= distance
    radial = bearing(np.asarray(target_xy, dtype=float) - np.asarray(base_xy, dtype=float))
    candidates = [(offset, heading(radial + offset)) for offset in POUR_OFFSETS_DEG]
    return sorted(candidates, key=lambda c: round(float(np.rad2deg(np.arccos(np.clip(c[1] @ drawn, -1., 1.)))), 6))


def rpy_deg(matrix):
    return Rotation.from_matrix(matrix).as_euler('xyz', degrees=True).tolist()


def probe_chain(reach_probe, poses, seed, limits):
    """Solve ``poses`` in order, each seeded from the previous solution; returns the worst (mm, deg) and last joints.

    The chain stops at the first pose outside ``limits``: the candidate is out of tolerance whatever the rest
    would give, and a pose the solver cannot reach costs seven solves (the planner's restarts).
    """
    worst = (0., 0.)
    for xyz, rotation in poses:
        position_error, rotation_error, joints = reach_probe(xyz, rotation, seed)
        worst = (max(worst[0], float(position_error)), max(worst[1], float(rotation_error)))
        if joints is not None:
            seed = joints
        if position_error > limits[0] or rotation_error > limits[1]:
            break
    return worst, seed


def pour_route(cup, target, table_z, fingertip_bias_mm, base_xy, tilt_deg, hold_s, grasp_fraction, clearance_m,
               speed_m_s, reach_probe=None, pose_tolerance=(30., 60.), mouth_clearance_m=POUR_MOUTH_CLEARANCE_M,
               grip_effort=CUP_GRIP_EFFORT):
    """Route for side-grasping ``cup`` and pouring it into ``target`` (both ``locate_object`` results).

    The approach direction and the direction the mouth tips toward are chosen from APPROACH_OFFSETS_DEG and
    POUR_OFFSETS_DEG. With ``reach_probe(xyz, rotation, seed_joints) -> (position_mm, rotation_deg, joints)``
    (``seed_joints`` is None at the start of each candidate chain) the five key poses of every candidate are
    solved with the arm's kinematics: the first candidate within EXACT_MM / EXACT_DEG wins, otherwise the one
    within ``pose_tolerance`` with the smallest normalised error. Without a probe the first candidates are used.
    The route ends at the pre-grasp point after the cup is put back; the service's automatic joint-space return
    home takes the arm back from there.
    """
    cup_height = cup['top'] - table_z
    if cup_height < MIN_CUP_HEIGHT_M:
        raise ValueError(f'杯子高约 {cup_height * 1000:.0f} mm，侧抓需要至少 50 mm 高')
    base_xy = np.asarray(base_xy, dtype=float)[:2]
    bias = fingertip_bias_mm / 1000
    grasp = np.r_[cup['center'], table_z + cup_height * grasp_fraction - bias]
    pours = pour_candidates(cup['center'], target['center'], base_xy)
    position_limit, rotation_limit = pose_tolerance

    def with_pour(candidate, pour_offset, pour_direction):
        # The cup was upright at grasp even if the wrist itself was pitched.
        # Rotate its measured TCP-to-mouth offset and keep its lowest rim point
        # just inside the target's near rim during tipping and returning upright.
        angles = np.linspace(0., tilt_deg, int(np.ceil(tilt_deg / 5.)) + 1)
        theta = np.deg2rad(angles)
        above = cup_height * (1. - grasp_fraction) + bias
        radius = candidate['radius']
        offsets = (above * np.sin(theta) + radius * np.cos(theta))[:, None] * pour_direction
        lip_z = target['top'] + np.maximum(mouth_clearance_m, cup_height * np.maximum(np.cos(theta), 0.) + .02)
        tcp_z = lip_z - above * np.cos(theta) + radius * np.sin(theta)
        target_radius = float(target.get('radius', np.median(np.linalg.norm(target['points'][:, :2]-target['center'], axis=1))))
        inset = min(.010, target_radius * .25)
        target_rim = np.asarray(target['center']) - pour_direction * target_radius
        lip_xy = target_rim + pour_direction * inset
        pouring = np.c_[lip_xy - offsets, tcp_z]
        rotations = [tilted_rotation(candidate['upright'], pour_direction, angle) for angle in angles]
        return dict(candidate, pour_offset_deg=pour_offset, pour_direction=pour_direction,
                    pour=pouring[-1], pouring=pouring, pour_rotations=rotations,
                    lip_xy=lip_xy, target_rim=target_rim, target_radius=target_radius, rim_inset=inset,
                    pour_z=float(tcp_z[-1]), tilted=rotations[-1])

    def score(errors):
        return max(errors[0] / position_limit, errors[1] / rotation_limit)

    best = None  # (normalised error, candidate); a candidate without pour_offset_deg failed at its grasp poses
    candidates = [(offset, approach, pitch) for offset, approach in approach_candidates(cup['center'], base_xy)
                  for pitch in SIDE_GRASP_PITCH_DEG]
    for approach_offset, approach, pitch_deg in candidates:
        upright = side_grasp_rotation(approach, pitch_deg)
        # The diameter sizes the stand-off and the pour offset; whether the jaws can hold the cup is the
        # operator's call. The rim circle is preferred: the visible extent is widened by the depth halo.
        diameter = cup.get('rim_diameter') or object_extent(cup['points'][:, :2], upright[:2, 0])
        pregrasp = grasp - np.r_[approach * (diameter / 2 + PREGRASP_STANDOFF_M), 0.]
        pour_z = pour_height(target['top'], cup_height, grasp_fraction, diameter / 2, tilt_deg, mouth_clearance_m) - bias
        lift = np.r_[cup['center'], max(grasp[2] + LIFT_M, pour_z)]
        candidate = {'approach_offset_deg': approach_offset, 'pitch_deg': pitch_deg, 'approach': approach, 'upright': upright,
                     'diameter': diameter,
                     'radius': diameter / 2, 'pregrasp': pregrasp, 'pour_z': pour_z, 'lift': lift, 'pour_offset_deg': None,
                     'errors': (0., 0.)}
        if reach_probe is None:
            best = (0., with_pour(candidate, *pours[0]))
            break
        grasp_errors, seed = probe_chain(reach_probe, [(pregrasp, upright), (grasp, upright), (lift, upright)], None, pose_tolerance)
        candidate['errors'] = grasp_errors
        if score(grasp_errors) > 1.:
            # Every pour direction shares these grasp poses; the candidate only stays for the error report.
            if best is None or score(grasp_errors) < best[0]:
                best = (score(grasp_errors), candidate)
            continue
        for pour_offset, pour_direction in pours:
            candidate = with_pour(candidate, pour_offset, pour_direction)
            pour_errors, _ = probe_chain(reach_probe, list(zip(candidate['pouring'], candidate['pour_rotations'])), seed, pose_tolerance)
            candidate['errors'] = (max(grasp_errors[0], pour_errors[0]), max(grasp_errors[1], pour_errors[1]))
            if best is None or score(candidate['errors']) < best[0]:
                best = (score(candidate['errors']), candidate)
            if candidate['errors'][0] <= EXACT_MM and candidate['errors'][1] <= EXACT_DEG:
                break
        else:
            continue
        break  # an exact candidate ends the search
    error_score, chosen = best
    if error_score > 1.:
        where = (f'接近方向偏离径向 {chosen["approach_offset_deg"]:+.0f}°、工具俯角 {chosen["pitch_deg"]:.0f}°' if chosen['pour_offset_deg'] is None else
                 f'接近方向偏离径向 {chosen["approach_offset_deg"]:+.0f}°、工具俯角 {chosen["pitch_deg"]:.0f}°、杯口倒向偏离目标径向 {chosen["pour_offset_deg"]:+.0f}°')
        stage = '抓取姿态' if chosen['pour_offset_deg'] is None else '关键位姿'
        raise ValueError(f'倒水路线不可达：最好的候选（{where}）{stage}误差 {chosen["errors"][0]:.0f} mm / '
                         f'{chosen["errors"][1]:.0f}°，超过容差 {position_limit:g} mm / {rotation_limit:g}°；'
                         '请把杯子或容器挪近这只机械臂，或换另一只臂')
    approach, pour_direction, upright = chosen['approach'], chosen['pour_direction'], chosen['upright']
    pregrasp, pour, lift = chosen['pregrasp'], chosen['pour'], chosen['lift']
    # Sampled compensated arc: no fixed-TCP rotation, including the way back.
    pouring = chosen['pouring']
    lift = lift.copy(); lift[2] = max(lift[2], float(pouring[:, 2].max()))
    points = [pregrasp, grasp, lift, *pouring, *pouring[-2::-1], lift, grasp, pregrasp]
    rotations = [upright] * 3 + chosen['pour_rotations'] + chosen['pour_rotations'][-2::-1] + [upright] * 3
    tilt_index = 3 + len(pouring) - 1
    upright_index = 3 + 2 * (len(pouring) - 1)
    release_index = len(points) - 2
    points = np.array(points)
    n = len(points) - 1
    keyframes = [[i / n, *rpy_deg(rotation)] for i, rotation in enumerate(rotations)]
    events = [{'s': 0., 'opening_mm': OPEN_MM, 'hold_s': OPEN_HOLD_S, 'wait_for_arrival': True},
              {'s': 1 / n, 'opening_mm': 0, 'hold_s': GRIP_HOLD_S, 'wait_for_arrival': True, 'verify': 'holding', 'ramp_s': GRIP_RAMP_S,
               'effort': int(grip_effort)},
              {'s': tilt_index / n, 'hold_s': float(hold_s), 'wait_for_arrival': True},
              {'s': release_index / n, 'opening_mm': OPEN_MM, 'hold_s': OPEN_HOLD_S, 'wait_for_arrival': True}]
    errors = None if reach_probe is None else chosen['errors']
    return {'route_mode': 'pour', 'pour_phase': [3 / n, upright_index / n], 'pour_hold_index': tilt_index,
            'cup_diameter_mm': chosen['diameter'] * 1000, 'cup_height_mm': cup_height * 1000,
            'pour_lip_xy': chosen['lip_xy'].tolist(), 'target_rim_xyz': [*chosen['target_rim'].tolist(), float(target['top'])],
            'target_radius_mm': chosen['target_radius'] * 1000, 'pour_rim_inset_mm': chosen['rim_inset'] * 1000,
            'target_height_mm': (target['top'] - table_z) * 1000, 'target_center_xy': np.asarray(target['center']).tolist(),
            'tilt_deg': float(tilt_deg), 'hold_s': float(hold_s),
            'mouth_clearance_m': float(mouth_clearance_m), 'grip_effort': int(grip_effort),
            'pour_height_mm': (chosen['pour_z'] + bias - table_z) * 1000,
            'approach_offset_deg': float(chosen['approach_offset_deg']), 'pour_offset_deg': float(chosen['pour_offset_deg']),
            'tool_pitch_deg': float(chosen['pitch_deg']),
            'reach_error_mm': None if errors is None else float(errors[0]),
            'reach_error_deg': None if errors is None else float(errors[1]),
            'pour_direction_xy': pour_direction.tolist(), 'approach_direction_xy': approach.tolist(),
            'source_xyz': grasp.tolist(), 'target_xyz': pour.tolist(),
            'arm': {'path': {'mode': 'waypoints', 'points': points.tolist()},
                    'orientation': {'mode': 'keyframes', 'points': keyframes},
                    'speed': float(speed_m_s), 'gripper_events': events,
                    'approach': {'speed_m_s': .8, 'clearance_m': float(clearance_m)}}}


def pour_from_stroke(frame, pixels, table_z=0., clearance_m=.12, speed_m_s=.8, preferred_yaw_deg=0.,
                     fingertip_bias_mm=0., base_xy=(0., 0.), tilt_deg=130., hold_s=2.5, grasp_fraction=.30,
                     reach_probe=None, pose_tolerance=(30., 60.), mouth_clearance_m=POUR_MOUTH_CLEARANCE_M,
                     grip_effort=CUP_GRIP_EFFORT):
    """Side-grasp the cup under the stroke start, pour it into the container under the stroke end and put it back."""
    ink = stroke_pixels(frame, pixels)
    base_xy = np.asarray(base_xy, dtype=float)[:2]
    parameters = [table_z, clearance_m, speed_m_s, preferred_yaw_deg, fingertip_bias_mm, tilt_deg, hold_s,
                  grasp_fraction, *base_xy]
    if not np.isfinite(parameters).all() or not .02 <= clearance_m <= .30 or not .001 <= speed_m_s <= 1.:
        raise ValueError('接近余量需为 0.02–0.30 m，速度需为 0.001–1.0 m/s')
    if not 60 <= tilt_deg <= 170:
        raise ValueError('倾倒角需为 60–170°')
    if not 0 < hold_s <= 3:
        raise ValueError('倾倒停留需为 0–3 秒')
    if not .2 <= grasp_fraction <= .7:
        raise ValueError('抓取高度比例需为 0.2–0.7')
    if not np.isfinite(mouth_clearance_m) or not .02 <= mouth_clearance_m <= .20:
        raise ValueError('杯口离沿高度需为 0.02–0.20 m')
    if not float(grip_effort).is_integer() or not 50 <= grip_effort <= 5000:
        raise ValueError('夹持力矩需为 50–5000 的整数（单位 0.001 N·m）')
    cloud, _ = leveled_cloud(frame, table_z)
    cup = locate_object(frame, ink[0], table_z, preferred_yaw_deg, cloud)
    circle = rim_circle(cup['points'], frame.t[:2, 3])
    if circle is not None and CUP_RIM_RADIUS_M[0] <= circle[1] <= CUP_RIM_RADIUS_M[1]:
        # A round cup: its rim circle is unbiased by the depth halo that shifts the visible centroid and widens
        # the extent (107 mm for an 86 mm cup); the halo-shifted centre helped push the cup out of the jaws.
        cup = {**cup, 'center': circle[0], 'rim_diameter': 2*circle[1]}
    try:
        target = locate_container(frame, ink[-1], table_z, preferred_yaw_deg, cloud)
    except ValueError as exc:
        raise ValueError(f'目标容器：{str(exc).replace("起点", "终点")}') from exc
    if np.array_equal(cup['mask'], target['mask']):
        raise ValueError('起点和终点落在同一个物体上，请从杯子画到另一个容器')
    route = pour_route(cup, target, table_z, fingertip_bias_mm, base_xy, tilt_deg, hold_s, grasp_fraction,
                       clearance_m, speed_m_s, reach_probe, pose_tolerance, mouth_clearance_m, int(grip_effort))
    return {'observation_id': frame.id, **route}
