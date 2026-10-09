"""Geometric top-down tabletop transfer from one screen-space intent stroke."""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, percentile_filter, zoom

# Heights are measured against the table plane fitted to the depth cloud, then re-expressed
# above the calibrated table height, so a tilted or offset camera calibration neither hides
# low objects nor turns bare table into phantom objects (on the development rig the raw depth of
# one table read -5 mm on one side and +14 mm on the other).
MIN_OBJECT_HEIGHT_M = .003
MAX_OBJECT_HEIGHT_M = .15
# Small blocks read only a few millimetres tall in the depth image while the bare table is
# noisy by about the same amount, so colour is the primary cue for coloured objects: they only
# need to clear the depth noise. Everything else (white, grey) must rise clearly above the table,
# and black pixels are never candidates because the arms are black.
COLOUR_SATURATION = .25
COLOUR_VALUE = .2   # dark purple blocks sit near 0.3; the black arms stay below 0.15
COLOUR_RISE_M = .002
DEPTH_RISE_M = .03   # depth alone: under bright room light the head camera's table noise reaches 2 cm
# Coloured pixels without depth (the arms shadow the camera's projector near their bases) are
# still objects resting on the table; their points come from the pixel ray at this height.
FILL_HEIGHT_M = .02
# When enough of the region does have depth, its own measured top is the fill height instead: near the bases
# the shadow covers about half a block, and the fixed 20 mm pulled a 12 mm block toward the camera and a
# 28 mm block away from it by 2-3 cm, so the jaws missed both.
MIN_MEASURED_FILL_POINTS = 30
TABLE_FIT_BAND_M = .04
TABLE_FIT_INLIER_M = .008
TABLE_FIT_STRIDE = 4
TABLE_FIT_MIN_POINTS = 2000
# After the plane fit, residual depth curvature of a few millimetres remains. A low-percentile
# filter over windows wider than any tabletop object recovers the local table background.
BACKGROUND_STRIDE = 8
BACKGROUND_WINDOW = 15
BACKGROUND_PERCENTILE = 20
SEGMENT_RADIUS_M = .18
MIN_OBJECT_POINTS = 150
AXIS_STEP_DEG = 5
MIN_JAW_WIDTH_M = .004
MAX_JAW_WIDTH_M = .067
# Dropping releases the object this far above the chosen support point, so a
# container that already holds earlier items does not count as an occupied landing.
DROP_RELEASE_HEIGHT_M = .06
PLACE_RELEASE_HEIGHT_M = .03   # a placement lets go this far above the support: lowering all the way hit bowls and the table
# A pinch grasps something the depth camera cannot segment: a shirt, a towel, a sheet of paper. Flat material
# never rises above the table, so nothing about it can be measured - the fingers go to the table itself at the
# drawn pixel and let go just above it (for example when folding a T-shirt with two arms).
PINCH_WIDTH_M = .02
# How far below the table plane the fingertips are driven to catch a single layer. Measured on the development
# platform: fingertips commanded 1 mm above the table closed on air and shoved the shirt aside every time;
# 5 mm of press is what actually pinches cloth. The table collision check is off by default and the calibration
# already allows the tips below the table (fingertip_below_table_mm), so this is a depth, not a violation.
PINCH_DEPTH_M = .005
MAX_PINCH_DEPTH_M = .02
PINCH_RELEASE_HEIGHT_M = .012
GRIP_RAMP_S = .5               # jaws are walked to their target over half a second instead of snapping
GRASP_TILT_START_RADIUS_M = .33   # upright top grasps reach about this far from the base
GRASP_TILT_FULL_RADIUS_M = .55    # beyond this the grasp leans the full MAX_GRASP_TILT_DEG toward the base
MAX_GRASP_TILT_DEG = 60.
MAX_POLYGON_POINTS = 4096
# Far drop points lie outside the vertical top-down reach of the arms; leaning the release pose
# back toward the base extends the reach the same way a tilted grasp does (see grasp_tilt_deg).
MAX_RELEASE_TILT_DEG = 80.
# A re-projected object centre may fall on a letter or highlight inside the object.
SNAP_RADIUS_PX = 12


def table_plane(cloud, table_z):
    """Fit z = a*x + b*y + c to the depth points near the calibrated table height.

    A coarse pass takes every point within ``TABLE_FIT_BAND_M`` of ``table_z``; a second pass
    keeps only the inliers of that fit so objects lying on the table do not pull the plane up.
    """
    sample = cloud[::TABLE_FIT_STRIDE, ::TABLE_FIT_STRIDE].reshape(-1, 3)
    sample = sample[np.isfinite(sample).all(axis=1)]
    selected = sample[np.abs(sample[:, 2]-table_z) < TABLE_FIT_BAND_M]
    if len(selected) < TABLE_FIT_MIN_POINTS:
        raise ValueError('可见的空闲桌面太少，无法估计桌面高度')
    coefficients = None
    for band in (TABLE_FIT_BAND_M, TABLE_FIT_INLIER_M):
        design = np.c_[selected[:, 0], selected[:, 1], np.ones(len(selected))]
        coefficients, *_ = np.linalg.lstsq(design, selected[:, 2], rcond=None)
        residual = selected[:, 2]-design @ coefficients
        selected = selected[np.abs(residual) < band]
        if len(selected) < TABLE_FIT_MIN_POINTS:
            raise ValueError('可见的空闲桌面太少，无法估计桌面高度')
    return coefficients


def plane_height(plane, xy):
    """Depth-measured table height at horizontal position(s) ``xy`` (…, 2)."""
    xy = np.asarray(xy, dtype=float)
    return plane[0]*xy[..., 0]+plane[1]*xy[..., 1]+plane[2]


def level_point(plane, table_z, xyz):
    """Re-express one depth point so its height sits above the calibrated table."""
    xyz = np.asarray(xyz, dtype=float).copy()
    xyz[2] = table_z+(xyz[2]-plane_height(plane, xyz[:2]))
    return xyz


def table_background(leveled_z, table_z):
    """Local table height map: a low percentile of the plane-leveled depth over wide windows.

    Objects are narrower than the window, so the percentile sees the table around them; the
    filter runs on a strided grid and is interpolated back to full resolution.
    """
    coarse = leveled_z[::BACKGROUND_STRIDE, ::BACKGROUND_STRIDE]
    filled = np.where(np.isfinite(coarse), coarse, table_z)
    background = percentile_filter(filled, BACKGROUND_PERCENTILE, size=BACKGROUND_WINDOW, mode='nearest')
    scale = (leveled_z.shape[0]/background.shape[0], leveled_z.shape[1]/background.shape[1])
    return zoom(background, scale, order=1)


def leveled_cloud(frame, table_z, plane=None):
    """World point cloud whose Z is the height above the local depth table plus ``table_z``.

    The table plane removes the camera tilt and offset, the background filter removes the
    remaining curvature, so segmentation thresholds, object heights and grasp heights all refer
    to the calibrated table the planner uses.
    """
    cloud = frame.world_points()
    if plane is None:
        plane = table_plane(cloud, table_z)
    leveled = cloud.copy()
    leveled[:, :, 2] = table_z+(cloud[:, :, 2]-plane_height(plane, cloud[:, :, :2]))
    leveled[:, :, 2] = table_z+(leveled[:, :, 2]-table_background(leveled[:, :, 2], table_z))
    return leveled, plane


def colourful_pixels(rgb):
    """Saturated and reasonably bright pixels: coloured objects, never the black arms or grey shadows."""
    values = np.asarray(rgb, dtype=float)/255.
    high = values.max(axis=2)
    low = values.min(axis=2)
    saturation = np.where(high > 0, (high-low)/np.maximum(high, 1e-6), 0.)
    return (saturation > COLOUR_SATURATION) & (high > COLOUR_VALUE)


def object_candidates(frame, cloud, table_z, within):
    """Label connected pixels that can belong to a tabletop object inside ``within``.

    ``cloud`` is the leveled cloud: coloured pixels qualify once they clear the depth noise,
    other non-black pixels must rise ``DEPTH_RISE_M`` above the table.
    """
    height = cloud[:, :, 2]-table_z
    finite = np.isfinite(cloud).all(axis=2)
    usable = finite & (height < MAX_OBJECT_HEIGHT_M) & within
    bright = np.asarray(frame.rgb).max(axis=2) > COLOUR_VALUE*255
    coloured = colourful_pixels(frame.rgb)
    candidates = usable & ((coloured & (height > COLOUR_RISE_M)) | (bright & (height > DEPTH_RISE_M)))
    candidates |= coloured & ~finite & within
    return label(candidates)


def plane_points(frame, columns, rows, z):
    """World points where the pixel rays through (columns, rows) meet the horizontal plane at ``z``."""
    pixels = np.c_[columns, rows, np.ones(len(columns))].T
    rays = frame.t[:3, :3] @ np.linalg.solve(frame.k, pixels)
    origin = frame.t[:3, 3]
    distance = (z-origin[2])/rays[2]
    return (origin[:, None]+rays*distance).T


def region_points(frame, cloud, mask, table_z):
    """Leveled points of one region; pixels without depth are placed at the region's measured top height when at
    least ``MIN_MEASURED_FILL_POINTS`` of its pixels have depth, otherwise ``FILL_HEIGHT_M`` above the table."""
    points = cloud[mask]
    missing = ~np.isfinite(points).all(axis=1)
    if missing.any():
        measured = points[~missing, 2]
        fill_z = float(np.quantile(measured, .90)) if len(measured) >= MIN_MEASURED_FILL_POINTS else table_z+FILL_HEIGHT_M
        rows, columns = np.nonzero(mask)
        points[missing] = plane_points(frame, columns[missing], rows[missing], fill_z)
    return points, float(missing.mean())


def object_geometry(points, preferred_yaw_deg=0.):
    """Closing axis, visible width, grasp centre, top height and footprint of one segmented object."""
    xy = points[:, :2]
    angles = np.deg2rad(np.arange(0, 180, AXIS_STEP_DEG))
    axes = np.c_[np.cos(angles), np.sin(angles)]
    low, high = np.quantile(xy @ axes.T, [.005, .995], axis=0)
    widths = high-low+.003
    candidates = np.where(widths <= float(widths.min())+.004)[0]
    # Parallel jaws are symmetric under 180 degrees; doubling the angle folds that
    # symmetry so the narrow axis closest to the current wrist yaw wins the tie.
    preferred = np.deg2rad(preferred_yaw_deg)
    offsets = np.abs(np.angle(np.exp(2j*(angles[candidates]-preferred))))
    at = int(candidates[np.argmin(offsets)])
    axis = axes[at]
    side = np.array([-axis[1], axis[0]])
    center = axis*((low[at]+high[at])/2)+side*np.median(xy @ side)
    top = float(np.quantile(points[:, 2], .90))
    radius = max(.025, float(np.quantile(np.linalg.norm(xy-center, axis=1), .95)))
    return {'center': center, 'axis': axis, 'axis_deg': float(np.rad2deg(angles[at])),
            'width': float(widths[at]), 'top': top, 'radius': radius}


def nearest_region(components, u, v):
    """Label at pixel (u, v), or the closest labelled pixel within ``SNAP_RADIUS_PX``.

    A re-projected object centre or a click can land on a letter, a highlight or a depth hole
    inside the object where no pixel qualified as a candidate; snapping to the nearest region
    keeps the object without accepting anything farther than a few pixels away.
    """
    h, w = components.shape
    r = SNAP_RADIUS_PX
    window = components[max(0, v-r):min(h, v+r+1), max(0, u-r):min(w, u+r+1)]
    labels = np.unique(window[window > 0])
    if not len(labels):
        return 0
    # A printed letter can split an object into a large body and slivers; take the largest
    # region touching the window rather than the sliver that happens to be closest.
    sizes = np.bincount(components.ravel())
    hit = int(components[v, u])
    if hit and sizes[hit] >= MIN_OBJECT_POINTS:
        return hit
    return int(labels[np.argmax(sizes[labels])])


CONTAINER_SEARCH_RADIUS_M = .25   # a pour target's rim may be this far from the point drawn inside it
CONTAINER_RIM_BAND_M = .01        # rim points: within this of the region's top height
CONTAINER_MAX_RADIUS_M = .25      # a wider rim circle is a straight edge, not a container
CONTAINER_MARGIN_M = .03          # a drawn point this far outside the rim circle still counts as inside
MIN_RIM_POINTS = 30


def rim_circle(points, camera_xy):
    """Circle ``(centre_xy, radius)`` of a region's rim from the top band of its leveled points, or None.

    Depth smoothing across the rim's edge moves the near side of the rim along the viewing ray, toward the
    centre, so a least-squares circle through the visible half of a tin lands too near the camera and too
    small (measured: 47 mm for a 56 mm rim). Rays are tangential at the sides of the rim, so the extent along
    the rim's principal axis is the diameter and the centre lies one radius behind the nearest rim point. A
    round container seen from one side, its far rim hidden or outside the picture, still shows that widest
    chord; a box fits its circumscribed circle. None when the band is too small or too wide to be a rim.
    """
    top = float(np.quantile(points[:, 2], .90))
    rim = points[points[:, 2] >= top-CONTAINER_RIM_BAND_M, :2]
    if len(rim) < MIN_RIM_POINTS:
        return None
    centroid = rim.mean(axis=0)
    offsets = rim-centroid
    axes = np.linalg.eigh(offsets.T @ offsets)[1]         # columns: minor axis, major axis
    minor, major = axes[:, 0], axes[:, 1]
    if (centroid-np.asarray(camera_xy, dtype=float)) @ minor < 0:
        minor = -minor                                     # the minor axis points away from the camera
    low, high = np.quantile(offsets @ major, [.005, .995])
    radius = float(high-low)/2
    if radius > CONTAINER_MAX_RADIUS_M:
        return None
    near = float(np.quantile(offsets @ minor, .005))
    return centroid+major*(low+high)/2+minor*(near+radius), radius


def containing_region(frame, cloud, components, count, xy, table_z):
    """Label of the candidate region whose rim circle contains ``xy`` with the smallest radius, else 0."""
    best = None
    sizes = np.bincount(components.ravel())
    for region in range(1, count+1):
        if sizes[region] < MIN_OBJECT_POINTS:
            continue
        points, _ = region_points(frame, cloud, components == region, table_z)
        circle = rim_circle(points, frame.t[:2, 3])
        if circle is None:
            continue
        centre, radius = circle
        if np.linalg.norm(xy-centre) <= radius+CONTAINER_MARGIN_M and (best is None or radius < best[0]):
            best = (radius, region)
    return best[1] if best else 0


def locate_container(frame, pixel, table_z=0., preferred_yaw_deg=0., cloud=None):
    """Like :func:`locate_object`, but a pixel drawn inside a hollow container (a bowl, a tin) resolves to the
    rim around it: the raised region whose rim circle contains the drawn point in world coordinates, so a
    container cut by the picture edge or hiding its far rim behind its near wall still counts. The centre is
    the rim circle's centre and the top its height, which is what a pour needs."""
    h, w = frame.depth.shape
    if cloud is None:
        cloud, _ = leveled_cloud(frame, table_z)
    u, v = np.rint(pixel).astype(int)
    u, v = min(u, w-1), min(v, h-1)
    # The floor drawn inside a container has depth; a hole is placed where the pixel ray meets the table.
    drawn = cloud[v, u]
    xy = drawn[:2] if np.isfinite(drawn).all() else frame.pick(*pixel, mode='plane', z=table_z)[:2]
    within = (np.linalg.norm(cloud[:, :, :2]-xy, axis=2) < CONTAINER_SEARCH_RADIUS_M) | ~np.isfinite(cloud).all(axis=2)
    components, count = object_candidates(frame, cloud, table_z, within)
    region = nearest_region(components, u, v) or containing_region(frame, cloud, components, count, xy, table_z)
    if not region:
        patch = cloud[max(0, v-2):v+3, max(0, u-2):u+3, 2]
        patch_z = float(np.nanmedian(patch)) if np.isfinite(patch).any() else float('nan')
        raise ValueError(f'终点（像素 {u},{v}）处没有可分离的物体，也不在任何容器的边沿之内（此处高出桌面 {(patch_z-table_z)*1000:.0f} mm），请画在容器里')
    mask = components == region
    points, filled = region_points(frame, cloud, mask, table_z)
    if len(points) < MIN_OBJECT_POINTS:
        raise ValueError('容器有效深度太少，无法自动定位')
    geometry = object_geometry(points, preferred_yaw_deg)
    circle = rim_circle(points, frame.t[:2, 3])
    if circle is not None:
        # Pour at the rim circle's centre: the centroid of the visible half sits toward the camera.
        geometry['center'], geometry['radius'] = circle
    height = geometry['top']-table_z
    if not MIN_OBJECT_HEIGHT_M < height < MAX_OBJECT_HEIGHT_M:
        raise ValueError(f'容器高约 {height*1000:.0f} mm，不在 {MIN_OBJECT_HEIGHT_M*1000:.0f}–{MAX_OBJECT_HEIGHT_M*1000:.0f} mm 的桌面物体范围')
    return {'points': points, 'mask': mask, 'surface': np.r_[xy, table_z], 'depth_filled': filled, **geometry}


class Unsegmented(ValueError):
    """Nothing rising above the table under the drawn pixel: bare table, or flat material lying on it."""


def surface_pinch(frame, pixel, table_z=0., preferred_yaw_deg=0., axis_deg=None):
    """Grasp description for a pixel with nothing segmentable under it: close the jaws on the surface there.

    Flat material has no height to segment, so none of its geometry can be measured. The position is the
    pixel ray at table height, the closing axis is the caller's (the stroke direction, else the wrist's
    current yaw), the width is nominal and the top is the table. The shape matches :func:`locate_object`
    so every transfer generator handles it unchanged, with ``pinch`` set: the fingers go to the table and
    the executor is not asked to verify a grip that one layer of cloth cannot report.
    """
    anchor = frame.pick(*pixel, mode='plane', z=table_z)
    angle = float(preferred_yaw_deg if axis_deg is None else axis_deg)
    radians = np.deg2rad(angle)
    return {'points': np.empty((0, 3)), 'mask': np.zeros(frame.depth.shape, dtype=bool),
            'surface': np.r_[anchor[:2], table_z], 'depth_filled': 0., 'center': np.asarray(anchor[:2]),
            'axis': np.r_[np.cos(radians), np.sin(radians)], 'axis_deg': angle,
            'width': PINCH_WIDTH_M, 'top': table_z, 'radius': .05, 'pinch': True}


def pinch_or_object(frame, pixel, table_z, preferred_yaw_deg, cloud, pinch=False, axis_deg=None):
    """The segmented object under ``pixel``, or a pinch of the surface there.

    ``pinch`` forces the pinch: folding cloth needs the fingers exactly where the operator drew, and a shirt
    that does segment would otherwise send them to its centroid. Without it the pinch is the answer only when
    segmentation finds nothing - an operator drawing on bare cloth means that spot, not a mistake.
    """
    if pinch:
        return surface_pinch(frame, pixel, table_z, preferred_yaw_deg, axis_deg)
    try:
        return locate_object(frame, pixel, table_z, preferred_yaw_deg, cloud)
    except Unsegmented:
        return surface_pinch(frame, pixel, table_z, preferred_yaw_deg, axis_deg)


def locate_object(frame, pixel, table_z=0., preferred_yaw_deg=0., cloud=None):
    """Segment the raised tabletop object under ``pixel`` and choose a horizontal closing axis.

    Returns the object's world points, the closing axis with its angle, the visible width
    across that axis, the grasp centre, the top height and a footprint radius. The width is
    returned unchecked so callers can apply their own jaw limits. Raises ``ValueError`` with an
    operator-facing message when nothing graspable is found.
    """
    h, w = frame.depth.shape
    if cloud is None:
        cloud, _ = leveled_cloud(frame, table_z)
    u, v = np.rint(pixel).astype(int)
    u, v = min(u, w-1), min(v, h-1)
    # The click position comes from the pixel ray at table height, so a depth hole on a printed
    # letter or a highlight cannot stop the lookup; the search radius tolerates the small offset.
    anchor = frame.pick(*pixel, mode='plane', z=table_z)
    # Pixels without depth have no distance; leave them to the colour rule so a shadowed block
    # under the click is still reachable.
    within = (np.linalg.norm(cloud[:, :, :2]-anchor[:2], axis=2) < SEGMENT_RADIUS_M) | ~np.isfinite(cloud).all(axis=2)
    components, _ = object_candidates(frame, cloud, table_z, within)
    region = nearest_region(components, u, v)
    patch = cloud[max(0, v-2):v+3, max(0, u-2):u+3, 2]
    patch_z = float(np.nanmedian(patch)) if np.isfinite(patch).any() else float('nan')
    if not region:
        raise Unsegmented(f'起点处没有可分离的物体（此处高出桌面 {(patch_z-table_z)*1000:.0f} mm），请画在物体内部')
    mask = components == region
    points, filled = region_points(frame, cloud, mask, table_z)
    if len(points) < MIN_OBJECT_POINTS:
        raise Unsegmented('物体有效深度太少，无法自动定位抓取')
    geometry = object_geometry(points, preferred_yaw_deg)
    surface = np.r_[anchor[:2], patch_z if np.isfinite(patch_z) else geometry['top']]
    height = geometry['top']-table_z
    if not MIN_OBJECT_HEIGHT_M < height < MAX_OBJECT_HEIGHT_M:
        raise ValueError(f'物体高约 {height*1000:.0f} mm，不在 {MIN_OBJECT_HEIGHT_M*1000:.0f}–{MAX_OBJECT_HEIGHT_M*1000:.0f} mm 的桌面物体范围')
    return {'points': points, 'mask': mask, 'surface': surface, 'depth_filled': filled, **geometry}


def polygon_mask(polygon, shape):
    """Rasterize a closed pixel polygon into a boolean mask of the given (height, width)."""
    image = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(image).polygon([tuple(point) for point in polygon.tolist()], fill=1)
    return np.array(image, dtype=bool)


def objects_in_polygon(frame, polygon_pixels, table_z=0., preferred_yaw_deg=0.):
    """List every raised object inside a lasso polygon, nearest to the robot first.

    Each entry carries the mask pixel closest to the region's centroid, the world grasp
    centre, the visible width across the chosen closing axis, the top height and whether
    the object fits the parallel jaws. The order is by world X in 1 cm bands (near to far),
    then from the robot's left to its right, so repeated calls on one frame agree.
    """
    polygon = np.asarray(polygon_pixels, dtype=float)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= MAX_POLYGON_POINTS \
            or not np.isfinite(polygon).all():
        raise ValueError(f'圈选需要 3–{MAX_POLYGON_POINTS} 个有效像素点')
    h, w = frame.depth.shape
    if np.any(polygon < 0) or np.any(polygon[:, 0] >= w) or np.any(polygon[:, 1] >= h):
        raise ValueError('圈选范围超出相机画面')
    cloud, _ = leveled_cloud(frame, table_z)
    components, count = object_candidates(frame, cloud, table_z, polygon_mask(polygon, (h, w)))
    objects = []
    for region in range(1, count+1):
        mask = components == region
        points, filled = region_points(frame, cloud, mask, table_z)
        if len(points) < MIN_OBJECT_POINTS:
            continue
        geometry = object_geometry(points, preferred_yaw_deg)
        rows, columns = np.nonzero(mask)
        nearest = int(np.argmin(np.hypot(columns-columns.mean(), rows-rows.mean())))
        height = geometry['top']-table_z
        width = geometry['width']
        reason = None
        if not MIN_OBJECT_HEIGHT_M < height < MAX_OBJECT_HEIGHT_M:
            reason = f'物体高约 {height*1000:.0f} mm，不在 {MIN_OBJECT_HEIGHT_M*1000:.0f}–{MAX_OBJECT_HEIGHT_M*1000:.0f} mm 的桌面物体范围'
        elif not MIN_JAW_WIDTH_M <= width <= MAX_JAW_WIDTH_M:
            reason = f'物体可见宽度约 {width*1000:.0f} mm，不能可靠放进 70 mm 夹口'
        objects.append({'pixel': [int(columns[nearest]), int(rows[nearest])],
                        'center_xy': geometry['center'].tolist(), 'top_z': geometry['top'],
                        'width_mm': width*1000, 'height_mm': height*1000,
                        'axis_deg': geometry['axis_deg'], 'point_count': int(len(points)), 'depth_filled': filled,
                        'graspable': reason is None, 'reason': reason})
    objects.sort(key=lambda item: (round(item['center_xy'][0], 2), -item['center_xy'][1]))
    for index, item in enumerate(objects):
        item['index'] = index
    return objects


def validate_parameters(table_z, clearance_m, speed_m_s, preferred_yaw_deg, fingertip_bias_mm):
    parameters = [table_z, clearance_m, speed_m_s, preferred_yaw_deg, fingertip_bias_mm]
    if not np.isfinite(parameters).all() or not .02 <= clearance_m <= .30 or not .001 <= speed_m_s <= 1.:
        raise ValueError('搬运高度需为 0.02–0.30 m，速度需为 0.001–1.0 m/s')


def place_transfer(found, landing, table_z, clearance_m, fingertip_bias_mm, drop=False,
                   pinch_depth_m=PINCH_DEPTH_M):
    """Grasp height, placement point and hover anchors.

    Every transfer generator shares this so stroke transfers, direct transfers and drops
    agree on where the fingers close and how high they carry. ``drop`` releases
    ``DROP_RELEASE_HEIGHT_M`` above the support instead of lowering onto it. The landing is
    not checked for other objects: the operator chooses it, and the head camera's table
    depth noise alone reaches the height of a small block, so such a check refuses free landings.
    """
    landing = np.asarray(landing, dtype=float)
    if drop:
        if not table_z-.035 <= landing[2] <= table_z+MAX_OBJECT_HEIGHT_M:
            raise ValueError('投放点高度超出范围，请点在桌面或筐内')
        support_z = landing[2]+DROP_RELEASE_HEIGHT_M
    else:
        if abs(landing[2]-table_z) > .035:
            raise ValueError('终点需要落在空闲桌面上；当前自动搬运支持桌面放置')
        support_z = landing[2]+(PINCH_RELEASE_HEIGHT_M if found.get('pinch') else PLACE_RELEASE_HEIGHT_M)
    # The measured width is reported, never used to refuse. Segmentation reads a handle, a neighbour or
    # a depth halo as part of the object and inflates it; the jaws open to their full 70 mm either way,
    # so whether a thing fits is something the operator can see and the estimate cannot tell them.
    width = found['width']
    center, top = found['center'], found['top']
    # A pinch has no object height to aim below: cloth is a millimetre thick, so the fingertips press into
    # the table and the 8 mm floor that keeps jaws off a block would close them on air above the fabric.
    finger_z = (table_z-pinch_depth_m if found.get('pinch') else
                max(table_z+.008, top-min(.025, (top-table_z)*.45)))
    grasp = np.r_[center, finger_z-fingertip_bias_mm/1000]
    place = np.r_[landing[:2], support_z+finger_z-table_z+.005-fingertip_bias_mm/1000]
    if np.linalg.norm(place[:2]-grasp[:2]) < max(.04, width):
        raise ValueError('放置位置离物体太近，请画到空闲桌面')
    hover_z = max(top, place[2])+clearance_m
    return {'grasp': grasp, 'place': place, 'hover': np.r_[grasp[:2], hover_z], 'finger_z': finger_z,
            'arrival': np.r_[place[:2], hover_z], 'placement': 'drop' if drop else 'place'}


def grasp_tilt_deg(grasp_xy, base_xy):
    """Lean of a top grasp toward the arm base, growing with the horizontal reach: upright top-down grasps
    only reach about 0.33 m from a PiPER base, leaning the wrist in extends that to the calibrated 0.55 m."""
    reach = float(np.linalg.norm(np.asarray(grasp_xy, dtype=float)[:2]-np.asarray(base_xy, dtype=float)[:2]))
    fraction = (reach-GRASP_TILT_START_RADIUS_M)/(GRASP_TILT_FULL_RADIUS_M-GRASP_TILT_START_RADIUS_M)
    return float(np.clip(fraction, 0., 1.))*MAX_GRASP_TILT_DEG


def grasp_orientation(grasp, jaw_yaw_deg, base_xy=None):
    """RPY for closing on ``grasp``: straight down near the base, leaning back toward it farther out."""
    if base_xy is None or grasp_tilt_deg(grasp, base_xy) <= 0.:
        return [180, 0, float(jaw_yaw_deg)]
    return release_orientation(grasp, jaw_yaw_deg, grasp_tilt_deg(grasp, base_xy), base_xy)


def release_orientation(place, jaw_yaw_deg, tilt_deg, base_xy):
    """RPY of a release pose leaning ``tilt_deg`` back toward the arm base: wrist pulled in, fingertips out."""
    from .robot.arm_model import approach_pose_dir
    from scipy.spatial.transform import Rotation
    direction = np.asarray(place[:2], dtype=float)-np.asarray(base_xy, dtype=float)
    if np.linalg.norm(direction) < 1e-6:
        raise ValueError('投放点与机械臂基座重合，无法确定后仰方向')
    pose = approach_pose_dir(np.asarray(place, dtype=float), float(jaw_yaw_deg), float(tilt_deg), direction)
    return Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist()


def transfer_draft(frame, found, placement, route, route_mode, table_z, clearance_m, speed_m_s,
                   preferred_yaw_deg, release_tilt_deg=0., base_xy=None, top_down=False):
    """Shared arm draft: hover, grasp, lift, carry, lower, release, retract with verified dwells."""
    hover, grasp, arrival, place = (placement[key] for key in ('hover', 'grasp', 'arrival', 'place'))
    # A drop releases in mid-air, so the motion ends at the release point: the final climb back to
    # hover height at full reach is where joint tracking lagged on the robot, and nothing below
    # needs clearing. Placing onto the table still retracts upward after letting go.
    dropping = placement['placement'] == 'drop'
    points = np.array([hover, grasp, hover, *route, arrival, place] if dropping else
                      [hover, grasp, hover, *route, arrival, place, arrival])
    n = len(points)-1
    yaw = found['axis_deg']
    yaw += 180*round((preferred_yaw_deg-yaw)/180)
    # Straight down near the base, leaning toward it farther out where a vertical wrist runs out of reach.
    # ``top_down`` keeps the wrist vertical the whole way instead: leaning reaches farther, but it also drags
    # the fingers sideways across whatever lies next to the grasp, which is wrong on a folded sleeve.
    lean_base = None if top_down else base_xy
    grasping = grasp_orientation(grasp, yaw, lean_base)
    if release_tilt_deg:
        if top_down:
            raise ValueError('全程竖直抓取与后仰释放不能同时使用')
        if not 0 < release_tilt_deg <= MAX_RELEASE_TILT_DEG:
            raise ValueError(f'释放后仰角需在 0–{MAX_RELEASE_TILT_DEG:.0f}°')
        if base_xy is None:
            raise ValueError('后仰释放需要机械臂基座位置')
        leaning = release_orientation(place, yaw, release_tilt_deg, base_xy)
        # Grasp orientation through the grasp and the lift; lean back from the carry onwards.
        keyframes = [[0, *grasping], [2/n, *grasping], [3/n, *leaning], [1, *leaning]]
    else:
        placing = grasp_orientation(place, yaw, lean_base)
        keyframes = ([[0, *grasping], [1, *grasping]] if placing == grasping else
                     [[0, *grasping], [2/n, *grasping], [3/n, *placing], [1, *placing]])
    close = {'s': 1/n, 'opening_mm': 0, 'hold_s': 1., 'wait_for_arrival': True, 'ramp_s': GRIP_RAMP_S}
    if not found.get('pinch'):
        # An object's width is real and worth verifying; one layer of cloth measures under the 1.5 mm the
        # holding gate needs, so verifying a successful pinch would abort the carry.
        close['verify'] = 'holding'
    events = [{'s': 0., 'opening_mm': 70, 'hold_s': .8, 'wait_for_arrival': True}, close,
              {'s': 1. if dropping else (n-1)/n, 'opening_mm': 70, 'hold_s': .8, 'wait_for_arrival': True,
               'ramp_s': GRIP_RAMP_S}]
    top = found['top']
    return {'observation_id': frame.id, 'object_width_mm': found['width']*1000,
            'object_height_mm': (top-table_z)*1000, 'pinch': bool(found.get('pinch')), 'top_down': bool(top_down),
            # Where the fingertips close, not where the wrist goes: the two differ by the arm's own
            # fingertip calibration, and the operator sets the first one.
            'finger_z_mm': (placement['finger_z']-table_z)*1000,
            'source_xyz': grasp.tolist(), 'target_xyz': place.tolist(),
            'route_mode': route_mode, 'placement': placement['placement'], 'release_tilt_deg': release_tilt_deg,
            'arm': {'path': {'mode': 'waypoints', 'points': points.tolist()},
                    'orientation': {'mode': 'keyframes', 'points': keyframes},
                    'speed': speed_m_s, 'gripper_events': events,
                    'approach': {'speed_m_s': .8, 'clearance_m': clearance_m}}}


def pick_place_from_stroke(frame, pixels, table_z=0., clearance_m=.12, speed_m_s=.8,
                           preferred_yaw_deg=0., fingertip_bias_mm=0., *, base_xy=None,
                           pinch=False, top_down=False, pinch_depth_mm=PINCH_DEPTH_M*1000):
    """Start selects an object, end selects its support surface; lift and carry are automatic.

    Interior ink is deliberately ignored. Release remains arrival-gated at the lowered placement pose.
    ``pinch=True`` grasps the drawn point itself instead of a segmented object, and ``top_down=True`` keeps
    the wrist vertical from grasp to release.
    """
    ink = np.asarray(pixels, dtype=float)
    if ink.ndim != 2 or ink.shape[1] != 2 or not 2 <= len(ink) <= 4096 or not np.isfinite(ink).all():
        raise ValueError('请从物体画到放置位置：需要 2–4096 个有效像素点')
    h, w = frame.depth.shape
    if np.any(ink < 0) or np.any(ink[:, 0] >= w) or np.any(ink[:, 1] >= h):
        raise ValueError('笔迹超出相机画面')
    if np.linalg.norm(ink[-1]-ink[0]) < 12:
        raise ValueError('请从物体画到不同的放置位置')
    validate_parameters(table_z, clearance_m, speed_m_s, preferred_yaw_deg, fingertip_bias_mm)
    if not 0 <= float(pinch_depth_mm) <= MAX_PINCH_DEPTH_M*1000:
        raise ValueError(f'捏取下探需为 0–{MAX_PINCH_DEPTH_M*1000:.0f} mm')
    cloud, plane = leveled_cloud(frame, table_z)
    # A pinch has no measured shape, so its jaws close along the stroke: the operator's own direction.
    # Both ends come from the pixel ray at table height, which needs no depth at either end.
    start, end = (frame.pick(*uv, mode='plane', z=table_z) for uv in (ink[0], ink[-1]))
    travel = end[:2]-start[:2]
    axis_deg = float(np.rad2deg(np.arctan2(travel[1], travel[0])))
    found = pinch_or_object(frame, ink[0], table_z, preferred_yaw_deg, cloud, pinch, axis_deg)
    # Black cloth reads +-20 mm and often has no depth at all, so a pinch lands on the table plane instead
    # of the depth surface: the drawn drop point of a fold is on fabric the camera cannot measure.
    landing = end if found.get('pinch') else level_point(plane, table_z, frame.pick(*ink[-1], mode='surface'))
    placement = place_transfer(found, landing, table_z, clearance_m, fingertip_bias_mm,
                               pinch_depth_m=float(pinch_depth_mm)/1000)
    return transfer_draft(frame, found, placement, [], 'two-point', table_z, clearance_m, speed_m_s,
                          preferred_yaw_deg, base_xy=base_xy, top_down=top_down)


def transfer_between(frame, object_pixel, landing_xyz, table_z=0., clearance_m=.12, speed_m_s=.8,
                     preferred_yaw_deg=0., fingertip_bias_mm=0., drop=False, release_tilt_deg=0., base_xy=None):
    """Direct transfer of the object under ``object_pixel`` to a world landing point.

    Produces the same draft format as ``pick_place_from_stroke``:
    the carry goes straight from the lift anchor to the arrival anchor at hover height.
    """
    pixel = np.asarray(object_pixel, dtype=float)
    h, w = frame.depth.shape
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or np.any(pixel < 0) \
            or pixel[0] >= w or pixel[1] >= h:
        raise ValueError('物体像素需要落在相机画面内')
    landing = np.asarray(landing_xyz, dtype=float) if isinstance(landing_xyz, (list, tuple, np.ndarray)) else None
    if landing is None or landing.shape != (3,) or not np.isfinite(landing).all() or np.any(np.abs(landing) > 2):
        raise ValueError('落点需要三个有限的世界坐标（米），绝对值不超过 2 m')
    validate_parameters(table_z, clearance_m, speed_m_s, preferred_yaw_deg, fingertip_bias_mm)
    cloud, plane = leveled_cloud(frame, table_z)
    found = locate_object(frame, pixel, table_z, preferred_yaw_deg, cloud)
    landing = level_point(plane, table_z, landing)
    placement = place_transfer(found, landing, table_z, clearance_m, fingertip_bias_mm, drop=drop)
    return transfer_draft(frame, found, placement, [], 'direct', table_z, clearance_m, speed_m_s,
                          preferred_yaw_deg, release_tilt_deg=release_tilt_deg, base_xy=base_xy)
