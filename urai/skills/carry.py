"""One-stroke carry: the grasp line, the flight path and the drop point written as a single draft.

Moving one bread half from the middle plate to a side plate used to cost three separate executions - draw a
grasp line and close, switch to the brush and fly at a hand-typed height, draw a descent and open. A
profile of a two-hot-dog demo measured 11.5 s of console overhead (observe, pick tool, draw, answer the
endpoint dialog, settle) against 42 s of actual motion for that one transfer, paid three times.
None of the geometry needed the split: the grasp line already fixes the closing axis and the grasp point, the
rest of the same stroke already fixes the route, and where the pen stops already fixes where the jaws open.

Two numbers the operator used to type are measured here instead:

* the carry height. A 45 mm carry dragged the source plate along with the bread, because the
  bread hung below the fingertips further than the plate rim was tall. The corridor under the drawn route is
  read out of the levelled point cloud, the object's hang below the fingertips is bounded by the table it was
  picked off (it cannot hang lower than that), and the two are added before the margin.
* the release height. The jaws open one object-thickness plus a small gap above whatever the pen stopped on,
  so dropping into a plate and dropping onto bare table are the same instruction.

The grasp itself is not re-derived: this skill calls the registered ``grasp_line`` planner, so the line-width
limits, the surface pick, the below-table refusal, the wrist lean and the soft closing ramp are literally the
ones that were validated on the robot.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..transfer import MAX_OBJECT_HEIGHT_M, grasp_orientation, plane_height, table_plane
from .motion import (BLOCK_GRIP_EFFORT, GRIP_RAMP_S, close_event, draft, open_event, search_candidates,
                     unreachable_error)
from .registry import get, integer, number, register

# A grasp line is cut out of the drawn pixels, so a stroke has to carry more than its two ends to have one.
MIN_GRASP_STRETCH_POINTS = 3

#: Half the swept width of the closed jaws plus a held object: what the carry has to fly clear of.
CORRIDOR_RADIUS_M = .07
CORRIDOR_STEP_M = .005          # the route is sampled this finely before the corridor is queried
CLOUD_STRIDE = 2                # every second row and column of the cloud: 0.9 M points is far more than needed
#: A height only counts as an obstacle when this many strided cloud samples reach it; one depth spike is noise.
MIN_OBSTACLE_SAMPLES = 20
OBSTACLE_RISE_M = .004          # below this the corridor is bare table
MAX_CARRY_Z_M = .40             # above the table; a PiPER wrist runs out of reach carrying higher than this
MIN_ROUTE_M = .04               # the drop point has to be at least this far past the end of the grasp line
ROUTE_STEP_M = .02              # drawn route points closer than this to the previous one are dropped
MAX_ROUTE_POINTS = 24
#: Tried in order when the nominal carry height does not solve; every step only adds clearance, never removes it.
CARRY_LIFT_STEPS_M = (0., .02, .04, .07)
LANDING_BELOW_TABLE_M = .005    # a landing measured lower than this is a depth failure, not a low support


def _dense(polyline, step_m):
    """Resample a polyline so consecutive samples are at most ``step_m`` apart."""
    out = [polyline[0]]
    for start, end in zip(polyline[:-1], polyline[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(end-start)/step_m)))
        out.extend(start+(end-start)*np.arange(1, count+1)[:, None]/count)
    return np.asarray(out, dtype=float)


def levelled_heights(frame, table_z):
    """Every depth pixel as (x, y, height above the calibrated table), with only the table plane removed.

    Deliberately not ``ctx.cloud``: that one additionally subtracts ``transfer.table_background``, a 20th
    percentile over a 200 mm window. Segmentation wants that - it is what makes an object stand out from the
    tabletop it lies on - and a carry must not have it, because the window swallows a dinner plate whole. The
    plate then reads as table and the bread on it reads only its own thickness, which is precisely how a 45 mm
    carry came to drag the source plate along. Clearing a rim is an absolute question.
    """
    points = frame.world_points()
    heights = table_z+(points[:, :, 2]-plane_height(table_plane(points, table_z), points[:, :, :2]))
    return np.dstack([points[:, :, :2], heights])


def corridor_top(cloud, table_z, polyline_xy, radius_m=CORRIDOR_RADIUS_M):
    """Highest real surface within ``radius_m`` of the drawn route, read out of the levelled cloud.

    Anything taller than a tabletop object is the arms, the camera bracket or the room rather than something
    to fly over, and a height backed by fewer than ``MIN_OBSTACLE_SAMPLES`` samples is depth noise; both would
    otherwise push the carry to the ceiling.
    """
    samples = _dense(np.asarray(polyline_xy, dtype=float), CORRIDOR_STEP_M)
    points = cloud[::CLOUD_STRIDE, ::CLOUD_STRIDE].reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[(points[:, 2] > table_z+OBSTACLE_RISE_M) & (points[:, 2] < table_z+MAX_OBJECT_HEIGHT_M)]
    if len(points) < MIN_OBSTACLE_SAMPLES:
        return float(table_z)
    low, high = samples.min(axis=0)-radius_m, samples.max(axis=0)+radius_m
    points = points[np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)]
    if len(points) < MIN_OBSTACLE_SAMPLES:
        return float(table_z)
    distance, _ = cKDTree(samples).query(points[:, :2])
    near = np.sort(points[distance < radius_m, 2])
    if len(near) < MIN_OBSTACLE_SAMPLES:
        return float(table_z)
    return float(near[-MIN_OBSTACLE_SAMPLES])


def fingertip_floor_m(ctx):
    """Lowest TCP height this arm may command, or None without a calibrated model.

    The same rule ``grasp.grasp_terms`` gives ``plan_grasp_line`` before it descends; the carry repeats the
    arithmetic for its release rather than letting the release sink through the table.
    """
    model = ctx.model
    if not getattr(ctx.backend, 'table_checks', False) or model is None:
        return None
    if abs(float(model.t_world_base[2, 2])-1.) > 1e-9:
        return None
    return float(model.t_world_base[2, 3])+(model.table_z_mm-model.fingertip_below_table_mm
                                            -model.fingertip_bias_mm+1.)/1000


def _stroke_plane(frame, ink, z):
    """The drawn pixels as world points on the horizontal plane at ``z``."""
    return np.array([frame.pick(u, v, mode='plane', z=z) for u, v in ink], dtype=float)


def _start_height(frame, ink, table_z):
    """Height of the surface the stroke starts on: the grasp line is measured on the object, not on the table."""
    try:
        height = float(frame.pick(*ink[0], mode='surface')[2])
    except ValueError:
        return float(table_z)
    return height if table_z-LANDING_BELOW_TABLE_M <= height <= table_z+MAX_OBJECT_HEIGHT_M else float(table_z)


def _split(points, span_m):
    """Where the leading grasp line ends: the first ink point ``span_m`` of arc length from the start."""
    if len(points) < 3:
        raise ValueError('这一笔只有两个点：请连续画——先横跨物体画一小段抓取线，再顺势画到要放的位置')
    steps = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    arc = np.r_[0., np.cumsum(steps)]
    if arc[-1] < span_m+MIN_ROUTE_M:
        raise ValueError(f'这一笔太短：抓取线 {span_m*1000:.0f} mm 之后还需要至少 {MIN_ROUTE_M*1000:.0f} mm 的'
                         f'搬运路线，整笔现在只有 {arc[-1]*1000:.0f} mm')
    return min(max(int(np.searchsorted(arc, span_m)), 1), len(points)-2), arc


def _jaw_yaw(frame, first, last, z, preferred_yaw_deg):
    """Closing axis of the grasp line, folded exactly the way ``grasp_from_line`` folds it.

    Both that planner and ``motion.jaw_yaw_near`` take the half-turn equivalent nearest the current wrist,
    but they break the tie at exactly +-90 degrees the other way round: jaw_yaw_near answers +90 where
    grasp_from_line answers -90. The release keyframe would then turn the wrist half a revolution away from
    the grasp the line planner had just produced, so the grasp planner's fold is the one that wins here.
    """
    ends = np.array([frame.pick(u, v, mode='plane', z=z) for u, v in (first, last)], dtype=float)
    delta = ends[1]-ends[0]
    axis = float(np.rad2deg(np.arctan2(delta[1], delta[0])))
    return float(preferred_yaw_deg)+(axis-float(preferred_yaw_deg)+90.) % 180.-90.


def _landing(ctx, cloud, pixel):
    """The point the pen stopped on: its height from the levelled cloud, its position from the pixel ray."""
    height, width = cloud.shape[:2]
    u = int(min(max(round(float(pixel[0])), 0), width-1))
    v = int(min(max(round(float(pixel[1])), 0), height-1))
    patch = cloud[max(0, v-2):v+3, max(0, u-2):u+3, 2]
    patch = patch[np.isfinite(patch)]
    if len(patch) < 3:
        raise ValueError('落点处没有可用深度：请把笔尖停在看得见的桌面或盘子上，或更新图像后重画')
    surface = float(np.median(patch))
    if not ctx.table_z-LANDING_BELOW_TABLE_M <= surface <= ctx.table_z+MAX_OBJECT_HEIGHT_M:
        raise ValueError(f'落点测得高出桌面 {(surface-ctx.table_z)*1000:.0f} mm，'
                         f'不在 0–{MAX_OBJECT_HEIGHT_M*1000:.0f} mm 的可放置范围；请换个落点或更新图像')
    return ctx.frame.pick(u, v, mode='plane', z=surface)


def _thin(points, step_m, limit):
    """Keep the first and last point and drop the ones closer than ``step_m`` to the one before them."""
    kept = [points[0]]
    for point in points[1:-1]:
        if np.linalg.norm(point-kept[-1]) >= step_m:
            kept.append(point)
    while len(kept) > 1 and np.linalg.norm(points[-1]-kept[-1]) < step_m:
        kept.pop()
    kept.append(points[-1])
    if len(kept) > limit:
        chosen = np.unique(np.rint(np.linspace(0, len(kept)-1, limit)).astype(int))
        kept = [kept[index] for index in chosen]
    return np.asarray(kept, dtype=float)


@register(name='carry_line', label='一笔搬运', group='抓取搬运', stroke='stroke', arms='single',
          summary='一笔从抓到放：开头一小段横跨物体决定两指闭合方向与抓取点，其余笔迹是搬运路线，笔尖停处放下',
          inputs={'grasp_span_mm': number('抓取线长度', 3, 70, 40, step=1, unit=' mm'),
                  'inset_mm': number('下探物体表面', 0, 50, 10, step=1, unit=' mm'),
                  'carry_margin_m': number('越障余量', .02, .20, .05, step=.01, unit=' m'),
                  'release_gap_mm': number('松手离落点', 0, 60, 10, step=1, unit=' mm'),
                  'approach_speed_m_s': number('接近速度', .01, 1., .8, step=.01, unit=' m/s'),
                  'grip_effort': integer('夹持力矩', 50, 5000, BLOCK_GRIP_EFFORT, unit=' ‰N·m')},
          hint='不要抬笔：先横跨物体画一小段（默认前 40 mm，就是两指要夹的那条线），接着顺势画出搬运路线，'
               '笔尖停在要放下的地方。软的东西（面包、纸杯）把力矩调到 300 左右。',
          limits='抓取线对应宽度需为 3–70 mm；搬运全程走同一高度，该高度按沿途点云最高物体自动算，'
                 '只保证飞过看得见的东西；不判断是否抓住，落点也不检查是否已被占用。')
def plan_carry_line(ctx, ink, grasp_span_mm, inset_mm, carry_margin_m, release_gap_mm, approach_speed_m_s,
                    grip_effort):
    frame = ctx.frame
    ink = np.asarray(ink, dtype=float)
    cloud = levelled_heights(frame, ctx.table_z)
    cut, arc = _split(_stroke_plane(frame, ink, _start_height(frame, ink, ctx.table_z)), grasp_span_mm/1000)
    # The grasp is the registered grasp_line skill, numbers and refusals included; only the route is new here.
    grasp = get('grasp_line').plan(ctx, np.array([ink[0], ink[cut]]), inset_mm=inset_mm,
                                   endpoint_action='grasp', grip_effort=grip_effort)
    hover, target = np.asarray(grasp['arm']['path']['points'], dtype=float)
    top = float(grasp['surface_xyz'][2])
    yaw = _jaw_yaw(frame, ink[0], ink[cut], top, ctx.preferred_yaw_deg)
    landing = _landing(ctx, cloud, ink[-1])
    drawn = np.array([frame.pick(u, v, mode='plane', z=ctx.table_z)[:2] for u, v in ink[cut:-1]], dtype=float)
    route = _thin(np.vstack([target[:2], drawn, landing[:2]]), ROUTE_STEP_M, MAX_ROUTE_POINTS+2)[1:-1]
    if np.linalg.norm(landing[:2]-target[:2]) < MIN_ROUTE_M:
        raise ValueError(f'落点离抓取点只有 {np.linalg.norm(landing[:2]-target[:2])*1000:.0f} mm，'
                         f'请画到至少 {MIN_ROUTE_M*1000:.0f} mm 以外')

    # The object cannot hang below the fingertips further than the table it was picked off, so this bound needs
    # no thickness estimate and is never optimistic.
    hang = max(0., float(target[2])-ctx.table_z)
    obstacle = corridor_top(cloud, ctx.table_z, np.vstack([target[:2], route, landing[:2]]))
    floor = fingertip_floor_m(ctx)
    release_z = float(landing[2])+hang+release_gap_mm/1000
    if floor is not None:
        release_z = max(release_z, floor)
    nominal = max(obstacle+hang, float(target[2]), release_z, top)+float(carry_margin_m)
    nominal = max(nominal, float(hover[2]))
    if nominal > ctx.table_z+MAX_CARRY_Z_M:
        raise ValueError(f'沿途最高处高出桌面 {(obstacle-ctx.table_z)*1000:.0f} mm，加上物体下垂 '
                         f'{hang*1000:.0f} mm 与余量后需要 {(nominal-ctx.table_z)*1000:.0f} mm 的搬运高度，'
                         f'超过 {MAX_CARRY_Z_M*1000:.0f} mm；请改画绕开高处的路线或降低余量')

    grasping = grasp_orientation(target, yaw, ctx.base_xy)
    placing = grasp_orientation(np.r_[landing[:2], release_z], yaw, ctx.base_xy)
    rotations = [Rotation.from_euler('xyz', pose, degrees=True).as_matrix() for pose in (grasping, placing)]

    def build(carry_z):
        flight = np.c_[route, np.full(len(route), carry_z)] if len(route) else np.empty((0, 3))
        lift, arrival = np.r_[target[:2], carry_z], np.r_[landing[:2], carry_z]
        return np.vstack([hover, target, lift, flight, arrival, np.r_[landing[:2], release_z], arrival])

    def poses(carry_z):
        points = build(carry_z)
        keys = [(points[1], rotations[0]), (points[2], rotations[0])]
        flight = points[3:-3]
        if len(flight):
            # The farthest waypoint from the base is where a carry runs out of reach, if anywhere.
            keys.append((flight[int(np.argmax(np.linalg.norm(flight[:, :2]-ctx.base_xy, axis=1)))], rotations[1]))
        return keys+[(points[-3], rotations[1]), (points[-2], rotations[1])]

    heights = [nominal+step for step in CARRY_LIFT_STEPS_M if nominal+step <= ctx.table_z+MAX_CARRY_Z_M]
    carry_z, errors, _, ok = search_candidates(ctx.probe if ctx.has_kinematics else None, heights, poses,
                                               ctx.pose_tolerance)
    if not ok:
        raise unreachable_error('一笔搬运的关键位姿', errors, ctx.pose_tolerance)
    points = build(carry_z)
    n = len(points)-1
    # Grasp orientation through the grasp and the lift, the placing lean from the carry onwards - the shape
    # transfer_draft uses, so a leaning release looks the same whichever skill produced it.
    keyframes = ([[0, *grasping], [1, *grasping]] if placing == grasping else
                 [[0, *grasping], [2/n, *grasping], [3/n, *placing], [1, *placing]])
    events = [open_event(0.), close_event(1/n, effort=grip_effort), open_event((n-1)/n, ramp_s=GRIP_RAMP_S)]
    spec = draft(points, keyframes, events, ctx.speed_m_s, ctx.clearance_m,
                 approach_speed_m_s=float(approach_speed_m_s))
    return {'summary': f'一笔搬运 · 抓取线 {grasp["width_mm"]:.0f} mm · 搬运高 {(carry_z-ctx.table_z)*1000:.0f} mm'
                       f' · 松手高 {(release_z-ctx.table_z)*1000:.0f} mm',
            'arm': spec, 'grasp_width_mm': float(grasp['width_mm']), 'grasp_line_mm': float(arc[cut]*1000),
            'actual_inset_mm': float(grasp['actual_inset_mm']), 'table_limited': bool(grasp['table_limited']),
            'object_hang_mm': float(hang*1000), 'obstacle_top_mm': float((obstacle-ctx.table_z)*1000),
            'carry_height_mm': float((carry_z-ctx.table_z)*1000),
            'release_height_mm': float((release_z-ctx.table_z)*1000),
            'source_xyz': target.tolist(), 'landing_xyz': landing.tolist(), 'route_points': int(len(route)),
            'reach_error_mm': float(errors[0]), 'reach_error_deg': float(errors[1])}
