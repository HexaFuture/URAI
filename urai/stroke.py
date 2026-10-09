"""Lift a screen-space brush stroke and bound its editable control points."""
from __future__ import annotations

import heapq
import numpy as np


def lift_stroke(frame, pixels, mode='plane', z=.15, surface_clearance_m=0.):
    try:
        samples = np.asarray(pixels, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('笔迹需要有限的二维像素坐标') from exc
    if samples.ndim != 2 or samples.shape[1] != 2 or not 2 <= len(samples) <= 4096 or not np.isfinite(samples).all():
        raise ValueError('笔迹需要 2–4096 个有限的二维像素坐标')
    if np.max(np.linalg.norm(samples-samples[0], axis=1)) < 2:
        raise ValueError('请按住鼠标拖动画出一条线')

    count = len(samples)
    h, w = frame.depth.shape
    if not ((samples >= 0).all() and (samples[:, 0] < w).all() and (samples[:, 1] < h).all()):
        raise ValueError('Pixel outside the observation')
    if mode == 'surface':
        # Pointer events can be sparse. Check the pixels traversed between
        # events too, so fast gestures cannot jump across unknown depth.
        steps = np.maximum(1, np.ceil(np.linalg.norm(np.diff(samples, axis=0), axis=1)).astype(int))
        if int(steps.sum())+1 > 16384:
            raise ValueError('深度表面笔迹过长，请缩短这一笔')
        samples = np.vstack([samples[:1]] + [np.linspace(a, b, int(n)+1)[1:]
                            for a, b, n in zip(samples[:-1], samples[1:], steps)])

    # Validate every sampled pixel before simplification: never bridge a hole in
    # measured depth merely because that pixel would not become a control point.
    points = np.asarray([frame.pick(u, v, mode, z) for u, v in samples])
    if not np.isfinite(points).all():
        raise ValueError('笔迹无法投影成有限的三维坐标')
    surface_clearance_m=float(surface_clearance_m)
    if not np.isfinite(surface_clearance_m) or not 0 <= surface_clearance_m <= .30:
        raise ValueError('表面上方抬升高度需为 0–300 mm')
    # The trajectory brush is intentionally kept above the object.  This is
    # relative to each measured surface point, so a garment or uneven object
    # remains clear instead of being forced onto a fixed horizontal plane.
    if mode == 'surface' and surface_clearance_m:
        points[:,2] += surface_clearance_m
    if np.max(np.linalg.norm(points-points[0], axis=1)) < 1e-8:
        raise ValueError('笔迹需要两个不同的空间位置')

    heap = []

    def segment(lo, hi):
        if hi-lo <= 1:
            return
        delta = points[hi]-points[lo]
        length2 = float(delta @ delta)
        interior = points[lo+1:hi]-points[lo]
        t = np.clip(interior @ delta / length2, 0, 1) if length2 > 1e-20 else np.zeros(len(interior))
        errors = np.linalg.norm(interior-t[:, None]*delta, axis=1)
        at = int(np.argmax(errors))
        heapq.heappush(heap, (-float(errors[at]), lo, hi, lo+1+at))

    keep = {0, len(points)-1}
    segment(0, len(points)-1)
    while heap and -heap[0][0] > .001 and len(keep) < 64:
        _, lo, hi, at = heapq.heappop(heap)
        keep.add(at)
        segment(lo, at)
        segment(at, hi)
    result = points[sorted(keep)]
    return {'observation_id': frame.id, 'points': result.tolist(),
            # Trajectory brush paths stay at their requested height; the
            # planner must not add the UI's lift approach on top of them.
            'air_track': True,
            'source_sample_count': count, 'control_point_count': len(result),
            # Error to the simplified polyline, not to the later PCHIP curve.
            'simplification_error_mm': (-heap[0][0]*1000 if heap else 0.)}
