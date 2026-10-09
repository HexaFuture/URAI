"""Synthetic tabletop scene for the kinematic simulator: geometry, ray casting and kinematic grasping.

Objects are rigid assemblies of axis-aligned primitives in their own frame (box, solid cylinder and an
open-top hollow cylinder that models cups and bowls); the object frame sits at the centre of the
object's bottom face. The scene renders RGB-D pictures for any pinhole camera by vectorised ray casting
and answers the two questions the simulated gripper asks: which object lies between the fingers and
how wide it is along the closing axis, and where a released object comes to rest.

All lengths are metres and all transforms are 4x4 homogeneous matrices in the world frame.
"""
from __future__ import annotations

import dataclasses
import functools
import threading
from collections.abc import Callable, Sequence

import numpy as np

__all__ = [
    "Box",
    "Capsule",
    "Cup",
    "Cylinder",
    "Scene",
    "SceneObject",
    "Table",
    "render_rgbd",
]

_EPS = 1e-9
_LIGHT = np.array([0.35, 0.25, 1.0]) / np.linalg.norm([0.35, 0.25, 1.0])
_AMBIENT = 0.35
_FLOOR_DROP_M = 0.75
_FLOOR_RGB = (70, 72, 76)
_TABLE_GRID_M = 0.1
_CAVITY_RISE_M = 10.0


@dataclasses.dataclass(frozen=True)
class Box:
    """Axis-aligned box ``size`` (x, y, z) whose bottom face centre is at ``(x0, y0, z0)``."""

    size: tuple[float, float, float]
    rgb: tuple[int, int, int]
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        sx, sy, sz = self.size
        return (np.array([self.x0 - sx / 2, self.y0 - sy / 2, self.z0]),
                np.array([self.x0 + sx / 2, self.y0 + sy / 2, self.z0 + sz]))


@dataclasses.dataclass(frozen=True)
class Cylinder:
    """Solid cylinder along z with its bottom centre at ``(x0, y0, z0)``."""

    radius: float
    height: float
    rgb: tuple[int, int, int]
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.radius
        return (np.array([self.x0 - r, self.y0 - r, self.z0]), np.array([self.x0 + r, self.y0 + r, self.z0 + self.height]))


@dataclasses.dataclass(frozen=True)
class Cup:
    """Open-top hollow cylinder (cup or bowl): outer ``radius``, side ``wall`` and ``bottom`` thickness."""

    radius: float
    height: float
    wall: float
    bottom: float
    rgb: tuple[int, int, int]
    x0: float = 0.0
    y0: float = 0.0
    z0: float = 0.0

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.radius
        return (np.array([self.x0 - r, self.y0 - r, self.z0]), np.array([self.x0 + r, self.y0 + r, self.z0 + self.height]))


Part = Box | Cylinder | Cup


@dataclasses.dataclass(frozen=True)
class Table:
    """Rectangular table top at height ``z`` covering ``x_range`` x ``y_range``; a floor lies below it."""

    z: float
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    rgb: tuple[int, int, int] = (196, 186, 166)


@dataclasses.dataclass
class SceneObject:
    """A rigid object: ``parts`` in the object frame and the object's world pose ``t_world_obj``."""

    name: str
    parts: tuple[Part, ...]
    t_world_obj: np.ndarray

    def local_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lows, highs = zip(*(p.bounds() for p in self.parts))
        return np.min(lows, axis=0), np.max(highs, axis=0)


# ---------------------------------------------------------------------------- ray / primitive intervals


def _slab(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis entry/exit parameters of rays ``o + t d`` through the slabs ``lo <= x <= hi``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    return np.minimum(t1, t2), np.maximum(t1, t2), t1, t2


def _box_interval(o: np.ndarray, d: np.ndarray, part: Box) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t_near, t_far, outward normal at entry, hit) of rays against an axis-aligned box."""
    lo, hi = part.bounds()
    tmin, tmax, _, _ = _slab(o, d, lo, hi)
    t_near = tmin.max(axis=1)
    t_far = tmax.min(axis=1)
    axis = tmin.argmax(axis=1)
    normal = np.zeros_like(o)
    rows = np.arange(len(o))
    normal[rows, axis] = -np.sign(d[rows, axis])
    return t_near, t_far, normal, (t_near <= t_far) & (t_far > _EPS)


def _cylinder_interval(o: np.ndarray, d: np.ndarray, radius: float, z_lo: float, z_hi: float, cx: float, cy: float
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t_near, t_far, outward normal at entry, outward normal at exit, hit) against a z-axis cylinder."""
    ox, oy = o[:, 0] - cx, o[:, 1] - cy
    dx, dy = d[:, 0], d[:, 1]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    parallel = a < 1e-14
    disc = b * b - 4.0 * a * c
    root = np.sqrt(np.maximum(disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tl0 = np.where(parallel, -np.inf, (-b - root) / (2.0 * a))
        tl1 = np.where(parallel, np.inf, (-b + root) / (2.0 * a))
    lateral = np.where(parallel, c <= 0.0, disc >= 0.0)
    tz_min, tz_max, tz_lo, _ = _slab(o[:, 2:3], d[:, 2:3], np.array([z_lo]), np.array([z_hi]))
    tz_min, tz_max, tz_lo = tz_min[:, 0], tz_max[:, 0], tz_lo[:, 0]
    t_near = np.maximum(tl0, tz_min)
    t_far = np.minimum(tl1, tz_max)
    hit = lateral & (t_near <= t_far) & (t_far > _EPS)

    def normal_at(t: np.ndarray, cap: np.ndarray, cap_sign: np.ndarray) -> np.ndarray:
        p = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
        n = np.stack([(p[:, 0] - cx) / radius, (p[:, 1] - cy) / radius, np.zeros(len(o))], axis=1)
        n[cap] = 0.0
        n[cap, 2] = cap_sign[cap]
        return n

    entry_cap = tz_min >= tl0
    exit_cap = tz_max <= tl1
    # Entering through the lower plane means the ray travels upwards (normal -z) and vice versa.
    entry_sign = np.where(tz_min == tz_lo, -1.0, 1.0)
    exit_sign = -entry_sign
    return t_near, t_far, normal_at(t_near, entry_cap, entry_sign), normal_at(t_far, exit_cap, exit_sign), hit


def _part_hit(o: np.ndarray, d: np.ndarray, part: Part) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First visible surface of ``part``: (t, outward normal facing the ray origin, hit)."""
    if isinstance(part, Box):
        t, _, n, hit = _box_interval(o, d, part)
        return t, n, hit & (t > _EPS)
    if isinstance(part, Cylinder):
        t, _, n, _, hit = _cylinder_interval(o, d, part.radius, part.z0, part.z0 + part.height, part.x0, part.y0)
        return t, n, hit & (t > _EPS)
    outer_t, outer_far, outer_n, _, outer_hit = _cylinder_interval(
        o, d, part.radius, part.z0, part.z0 + part.height, part.x0, part.y0)
    cav_t, cav_far, _, cav_exit_n, cav_hit = _cylinder_interval(
        o, d, part.radius - part.wall, part.z0 + part.bottom, part.z0 + part.height + _CAVITY_RISE_M, part.x0, part.y0)
    in_cavity = cav_hit & (cav_t <= outer_t + 1e-12) & (outer_t <= cav_far)
    t = np.where(in_cavity, cav_far, outer_t)
    n = np.where(in_cavity[:, None], -cav_exit_n, outer_n)
    hit = outer_hit & (~in_cavity | (cav_far < outer_far)) & (t > _EPS)
    return t, n, hit


def _part_hull_interval(o: np.ndarray, d: np.ndarray, part: Part) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t_near, t_far, hit) of the part's convex hull: what a pair of fingers closing on it touches."""
    if isinstance(part, Box):
        t0, t1, _, hit = _box_interval(o, d, part)
        return t0, t1, hit
    t0, t1, _, _, hit = _cylinder_interval(o, d, part.radius, part.z0, part.z0 + part.height, part.x0, part.y0)
    return t0, t1, hit


@dataclasses.dataclass(frozen=True)
class Capsule:
    """Segment ``p``-``q`` (world, metres) swept by a sphere of ``radius``: how simulated arm links are drawn."""

    p: np.ndarray
    q: np.ndarray
    radius: float
    rgb: tuple[int, int, int]


def _capsule_hit(o: np.ndarray, d: np.ndarray, cap: Capsule) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First intersection (t, outward normal, hit) of rays ``o + t d`` with a capsule."""
    scale = np.linalg.norm(d, axis=1)
    rd = d / scale[:, None]
    ba = cap.q - cap.p
    oa = o - cap.p
    baba = float(ba @ ba)
    bard = rd @ ba
    baoa = oa @ ba
    rdoa = np.einsum("ij,ij->i", rd, oa)
    oaoa = np.einsum("ij,ij->i", oa, oa)
    a = baba - bard * bard
    b = baba * rdoa - baoa * bard
    c = baba * oaoa - baoa * baoa - cap.radius ** 2 * baba
    h = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        t_body = (-b - np.sqrt(np.maximum(h, 0.0))) / a
    y = baoa + t_body * bard
    body = (h >= 0.0) & (a > 1e-12) & (y > 0.0) & (y < baba)
    end = np.where(((y <= 0.0) | ~np.isfinite(y))[:, None], oa, o - cap.q)
    bc = np.einsum("ij,ij->i", rd, end)
    hc = bc * bc - (np.einsum("ij,ij->i", end, end) - cap.radius ** 2)
    t_cap = -bc - np.sqrt(np.maximum(hc, 0.0))
    t_unit = np.where(body, t_body, np.where(hc > 0.0, t_cap, np.nan))
    hit = np.isfinite(t_unit) & (t_unit > _EPS)
    pos = o + rd * np.where(hit, t_unit, 0.0)[:, None]
    pa = pos - cap.p
    along = np.clip((pa @ ba) / baba, 0.0, 1.0) if baba > 0.0 else np.zeros(len(o))
    normal = (pa - along[:, None] * ba) / cap.radius
    return np.where(hit, t_unit / scale, np.inf), normal, hit


def _capsule_window(cap: Capsule, k: np.ndarray, t_cam_world: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Flat pixel indices covered by the projection of the capsule's bounding box."""
    r = cap.radius
    lo, hi = np.minimum(cap.p, cap.q) - r, np.maximum(cap.p, cap.q) + r
    return _window(np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]),
                   k, t_cam_world, width, height)


# ---------------------------------------------------------------------------- rendering


@functools.lru_cache(maxsize=8)
def _camera_rays(fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> np.ndarray:
    """Camera-frame ray directions with unit z (the ray parameter is the depth along the optical axis)."""
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1).reshape(-1, 3)
    rays.flags.writeable = False
    return rays


def _pixel_rays(k: np.ndarray, width: int, height: int) -> np.ndarray:
    return _camera_rays(float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), int(width), int(height))


def _shade_planes(origin: np.ndarray, dirs: np.ndarray, table: Table, depth: np.ndarray, normal: np.ndarray,
                  albedo: np.ndarray) -> None:
    """Fill the buffers with the floor and the table top (upward normals; grid lines on the table)."""
    dz = dirs[:, 2]
    for plane_z, top in ((table.z - _FLOOR_DROP_M, None), (table.z, table)):
        if origin[2] <= plane_z:
            continue
        with np.errstate(divide="ignore"):
            t = np.where(dz < 0.0, (plane_z - origin[2]) / dz, np.inf)
        x = origin[0] + dirs[:, 0] * t
        y = origin[1] + dirs[:, 1] * t
        keep = t < depth
        if top is None:
            rgb = np.broadcast_to(np.asarray(_FLOOR_RGB, dtype=np.float64), albedo.shape)
        else:
            keep &= (x >= top.x_range[0]) & (x <= top.x_range[1]) & (y >= top.y_range[0]) & (y <= top.y_range[1])
            line = ((np.abs(x / _TABLE_GRID_M - np.round(x / _TABLE_GRID_M)) < 0.015)
                    | (np.abs(y / _TABLE_GRID_M - np.round(y / _TABLE_GRID_M)) < 0.015))
            base = np.asarray(top.rgb, dtype=np.float64)
            rgb = np.where(line[:, None], base * 0.82, base)
        np.copyto(depth, t, where=keep)
        np.copyto(normal, (0.0, 0.0, 1.0), where=keep[:, None])
        np.copyto(albedo, rgb, where=keep[:, None])


_NEAR_M = 1e-3
#: Edges of a box whose corners are ordered as ``[(x, y, z) for x in .. for y in .. for z in ..]``.
_BOX_EDGES = tuple((i, i | bit) for i in range(8) for bit in (1, 2, 4) if not i & bit)


def _window(corners_world: np.ndarray, k: np.ndarray, t_cam_world: np.ndarray, width: int, height: int
            ) -> np.ndarray | None:
    """Flat indices of the pixels covered by a box (8 corners, see :data:`_BOX_EDGES`); None when off screen.

    The box is clipped against the near plane first, so boxes that reach behind the camera still get a
    tight window.
    """
    cam = corners_world @ t_cam_world[:3, :3].T + t_cam_world[:3, 3]
    front = cam[cam[:, 2] >= _NEAR_M]
    cuts = [cam[i] + (cam[j] - cam[i]) * (_NEAR_M - cam[i, 2]) / (cam[j, 2] - cam[i, 2])
            for i, j in _BOX_EDGES if (cam[i, 2] - _NEAR_M) * (cam[j, 2] - _NEAR_M) < 0.0]
    points = np.vstack([front, *cuts]) if cuts else front
    if len(points) == 0:
        return None
    u = k[0, 0] * points[:, 0] / points[:, 2] + k[0, 2]
    v = k[1, 1] * points[:, 1] / points[:, 2] + k[1, 2]
    u0, u1 = max(int(np.floor(u.min())), 0), min(int(np.ceil(u.max())), width - 1)
    v0, v1 = max(int(np.floor(v.min())), 0), min(int(np.ceil(v.max())), height - 1)
    if u0 > u1 or v0 > v1:
        return None
    cols, rows = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
    return (rows * width + cols).ravel()


def _object_window(obj: SceneObject, k: np.ndarray, t_cam_world: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Flat pixel indices covered by the projection of the object's bounding box."""
    lo, hi = obj.local_bounds()
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return _window(corners @ obj.t_world_obj[:3, :3].T + obj.t_world_obj[:3, 3], k, t_cam_world, width, height)


def render_rgbd(k: np.ndarray, t_world_cam: np.ndarray, width: int, height: int, table: Table,
                objects: Sequence[SceneObject], capsules: Sequence[Capsule] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast an RGB image (uint8) and a depth image (metres along the optical axis, NaN = no return)."""
    rot, origin = t_world_cam[:3, :3], t_world_cam[:3, 3]
    dirs = _pixel_rays(k, width, height) @ rot.T
    n_pix = len(dirs)
    depth = np.full(n_pix, np.inf)
    normal = np.zeros((n_pix, 3))
    albedo = np.zeros((n_pix, 3))
    _shade_planes(origin, dirs, table, depth, normal, albedo)
    t_cam_world = np.linalg.inv(t_world_cam)
    for obj in objects:
        sel = _object_window(obj, k, t_cam_world, width, height)
        if sel is None:
            continue
        t_obj_world = np.linalg.inv(obj.t_world_obj)
        o_local = np.broadcast_to(t_obj_world[:3, :3] @ origin + t_obj_world[:3, 3], (len(sel), 3))
        d_local = dirs[sel] @ t_obj_world[:3, :3].T
        for part in obj.parts:
            t, n, hit = _part_hit(o_local, d_local, part)
            closer = hit & (t < depth[sel])
            idx = sel[closer]
            depth[idx] = t[closer]
            normal[idx] = n[closer] @ obj.t_world_obj[:3, :3].T
            albedo[idx] = part.rgb
    for cap in capsules:
        sel = _capsule_window(cap, k, t_cam_world, width, height)
        if sel is None:
            continue
        t, n, hit = _capsule_hit(np.broadcast_to(origin, (len(sel), 3)), dirs[sel], cap)
        closer = hit & (t < depth[sel])
        idx = sel[closer]
        depth[idx] = t[closer]
        normal[idx] = n[closer]
        albedo[idx] = cap.rgb
    found = np.isfinite(depth)
    flip = np.einsum("ij,ij->i", normal, dirs) > 0.0
    light = _AMBIENT + (1.0 - _AMBIENT) * np.clip(np.where(flip, -1.0, 1.0) * (normal @ _LIGHT), 0.0, 1.0)
    rgb = albedo * np.where(found, light, 0.0)[:, None]
    out_depth = np.where(found, depth, np.nan).astype(np.float32)
    return np.clip(np.round(rgb), 0, 255).astype(np.uint8).reshape(height, width, 3), out_depth.reshape(height, width)


# ---------------------------------------------------------------------------- scene state


class Scene:
    """Table plus movable objects; objects held by a simulated gripper follow that gripper.

    Args:
        table: The table top.
        objects: Initial objects (names must be unique).
    """

    def __init__(self, table: Table, objects: Sequence[SceneObject]) -> None:
        names = [o.name for o in objects]
        if len(set(names)) != len(names):
            raise ValueError(f"object names must be unique, got {names}")
        self.table = table
        self._objects = {o.name: dataclasses.replace(o, t_world_obj=np.array(o.t_world_obj, dtype=np.float64))
                         for o in objects}
        self._held: dict[str, tuple[Callable[[], np.ndarray], np.ndarray]] = {}
        self._lock = threading.RLock()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._objects)

    def pose(self, name: str) -> np.ndarray:
        """Current world pose of object ``name``."""
        with self._lock:
            held = self._held.get(name)
            if held is not None:
                holder, t_holder_obj = held
                return holder() @ t_holder_obj
            return self._objects[name].t_world_obj.copy()

    def objects(self) -> list[SceneObject]:
        """Snapshot of every object at its current pose."""
        with self._lock:
            return [dataclasses.replace(o, t_world_obj=self.pose(o.name)) for o in self._objects.values()]

    def render(self, k: np.ndarray, t_world_cam: np.ndarray, width: int, height: int,
               capsules: Sequence[Capsule] = ()) -> tuple[np.ndarray, np.ndarray]:
        """RGB-D picture of the scene (and of ``capsules``) from a pinhole camera, see :func:`render_rgbd`."""
        return render_rgbd(k, t_world_cam, width, height, self.table, self.objects(), capsules)

    def width_between_fingers(self, t_world_hand: np.ndarray, opening_m: float, depths_m: Sequence[float],
                              offsets_m: Sequence[float] = (0.0,)) -> tuple[str, float] | None:
        """The free object between two fingers closing along the hand's x axis, and its width there.

        The finger pads are sampled at ``depths_m`` along the hand's z axis and ``offsets_m`` along its y
        axis. An object counts as between the fingers when, on some sample line, the extent of its convex
        hull lies within the current opening; its width is the largest such extent.
        """
        reach = 1.0
        samples = np.array([[-reach, y, z, 1.0] for z in depths_m for y in offsets_m])
        origins = (t_world_hand @ samples.T).T[:, :3]
        direction = np.broadcast_to(t_world_hand[:3, 0], origins.shape)
        half = 0.5 * float(opening_m) + 1e-3
        best: tuple[str, float] | None = None
        with self._lock:
            for obj in self.objects():
                if obj.name in self._held:
                    continue
                inv = np.linalg.inv(obj.t_world_obj)
                o_local = origins @ inv[:3, :3].T + inv[:3, 3]
                d_local = direction @ inv[:3, :3].T
                lo = np.full(len(origins), np.inf)
                hi = np.full(len(origins), -np.inf)
                for part in obj.parts:
                    t0, t1, hit = _part_hull_interval(o_local, d_local, part)
                    lo = np.where(hit, np.minimum(lo, t0), lo)
                    hi = np.where(hit, np.maximum(hi, t1), hi)
                inside = (lo - reach >= -half) & (hi - reach <= half) & (hi > lo)
                if inside.any():
                    width = float((hi - lo)[inside].max())
                    if best is None or width > best[1]:
                        best = (obj.name, width)
        return best

    def attach(self, name: str, holder: Callable[[], np.ndarray]) -> None:
        """Make object ``name`` follow ``holder()`` (a world pose) with its current relative pose."""
        with self._lock:
            pose = self.pose(name)
            self._held[name] = (holder, np.linalg.inv(holder()) @ pose)

    def release(self, name: str) -> np.ndarray:
        """Detach object ``name`` and let it settle upright on the surface below its centre."""
        with self._lock:
            pose = self.pose(name)
            del self._held[name]
            obj = self._objects[name]
            heading = pose[:3, 0] if np.linalg.norm(pose[:2, 0]) > 0.3 else pose[:3, 1]
            yaw = float(np.arctan2(heading[1], heading[0]))
            settled = np.eye(4)
            settled[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            lo, _ = obj.local_bounds()
            settled[:3, 3] = [pose[0, 3], pose[1, 3], self.support_height(pose[:2, 3], exclude=name) - lo[2]]
            obj.t_world_obj = settled
            return settled.copy()

    def support_height(self, xy: Sequence[float], exclude: str | None = None) -> float:
        """Height of the first surface straight below ``xy`` (objects, table, else the floor)."""
        top = self.table.z + 5.0
        origin = np.array([[xy[0], xy[1], top]])
        down = np.array([[0.0, 0.0, -1.0]])
        inside_table = (self.table.x_range[0] <= xy[0] <= self.table.x_range[1]
                        and self.table.y_range[0] <= xy[1] <= self.table.y_range[1])
        best = self.table.z if inside_table else self.table.z - _FLOOR_DROP_M
        with self._lock:
            for obj in self.objects():
                if obj.name == exclude or obj.name in self._held:
                    continue
                inv = np.linalg.inv(obj.t_world_obj)
                o_local = origin @ inv[:3, :3].T + inv[:3, 3]
                d_local = down @ inv[:3, :3].T
                for part in obj.parts:
                    t, _, hit = _part_hit(o_local, d_local, part)
                    if hit[0]:
                        best = max(best, top - float(t[0]))
        return best
