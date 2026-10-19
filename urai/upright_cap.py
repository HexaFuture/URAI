"""Measured-axis cap turning. Pure geometry; callers still preview and execute."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def cap_geometry(surface, holder_xyz, holder_rotation, body_tool_offset, inset_m):
    surface = np.asarray(surface, float)
    body = np.asarray(holder_xyz, float) - np.asarray(holder_rotation) @ np.asarray(body_tool_offset)
    delta = surface - body
    length = np.linalg.norm(delta)
    if not np.isfinite(delta).all() or not .12 <= length <= .25:
        raise ValueError('瓶盖与持瓶抓点不一致，请重新定位瓶盖')
    axis = delta / length
    if axis[2] < np.cos(np.deg2rad(15)):
        raise ValueError('瓶子倾斜超过 15°，请先调整持瓶姿态')
    if not .003 <= inset_m <= .015:
        raise ValueError('瓶盖侧面抓点深度需为 3–15 mm')
    return surface - axis * inset_m, axis


def measured_tool_offset(tcp, rotation, center):
    offset = np.asarray(rotation).T @ (np.asarray(tcp) - np.asarray(center))
    if not np.isfinite(offset).all() or np.linalg.norm(offset) > .04:
        raise ValueError('夹爪与新瓶盖点偏离超过 40 mm，请先重新夹盖')
    return offset


def aligned_turn_route(center, axis, offset, start_rotation, base_rotation,
                       angle_deg, cycles, opening_mm=50, effort=2500):
    """Rotate the measured contact center, including insertion depth, not just TCP Y.

    The final turn keeps the lid clamped. A new observation is required before
    another batch or a lift. Empty closes are checked by the ordinary executor.
    """
    center, axis, offset = map(lambda x: np.asarray(x, float), (center, axis, offset))
    if not np.isfinite(np.r_[center, axis, offset]).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError('Invalid cap geometry')
    axis = axis / np.linalg.norm(axis)
    if not 5 <= angle_deg <= 60 or not 1 <= cycles <= 6 or int(cycles) != cycles:
        raise ValueError('每批最多 6 轮，每轮 5–60°')
    start, base = np.asarray(start_rotation), np.asarray(base_rotation)
    points, rotations, indexed_events = [center + start @ offset], [start], []
    if np.linalg.norm(Rotation.from_matrix(base @ start.T).as_rotvec()) > np.deg2rad(.5):
        indexed_events.append((0, opening_mm))
        reset = Slerp([0, 1], Rotation.from_matrix([start, base]))
        for t in np.linspace(1/6, 1, 6):
            r = reset(t).as_matrix()
            points.append(center + r @ offset); rotations.append(r)
        indexed_events.append((len(points)-1, 0))
    else:
        base = start
    for cycle in range(int(cycles)):
        for angle in np.linspace(angle_deg/6, angle_deg, 6):
            r = Rotation.from_rotvec(axis * np.deg2rad(angle)).as_matrix() @ base
            points.append(center + r @ offset); rotations.append(r)
        if cycle == cycles-1:
            break
        indexed_events.append((len(points)-1, opening_mm))
        for angle in np.linspace(angle_deg*5/6, 0, 6):
            r = Rotation.from_rotvec(axis * np.deg2rad(angle)).as_matrix() @ base
            points.append(center + r @ offset); rotations.append(r)
        indexed_events.append((len(points)-1, 0))
    events = []
    for i, width in indexed_events:
        event = {'s': i/(len(points)-1), 'opening_mm': width, 'hold_s': .8,
                 'wait_for_arrival': True, 'ramp_s': .4}
        if width == 0:
            event.update(effort=int(effort), verify='holding')
        events.append(event)
    return points, rotations, events
