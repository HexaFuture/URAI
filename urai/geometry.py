from __future__ import annotations

import base64
import io
import time
import uuid
from dataclasses import dataclass, field
import numpy as np
from PIL import Image


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    k: np.ndarray
    t: np.ndarray
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def world_points(self):
        v, u = np.indices(self.depth.shape)
        z = np.where(np.isfinite(self.depth) & (self.depth > 0), self.depth, np.nan)
        p = np.stack([(u-self.k[0, 2])*z/self.k[0, 0], (v-self.k[1, 2])*z/self.k[1, 1], z], -1)
        return p @ self.t[:3, :3].T + self.t[:3, 3]

    def project(self, xyz):
        p = np.asarray(xyz).reshape(-1, 3)
        c = (p-self.t[:3, 3]) @ self.t[:3, :3]
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = np.stack([self.k[0, 0]*c[:, 0]/c[:, 2]+self.k[0, 2],
                           self.k[1, 1]*c[:, 1]/c[:, 2]+self.k[1, 2]], -1)
        uv[c[:, 2] <= 0] = np.nan
        return uv

    def pick(self, u, v, mode='plane', z=.15):
        u, v = float(u), float(v)
        h, w = self.depth.shape
        if not (np.isfinite([u, v, z]).all() and 0 <= u < w and 0 <= v < h):
            raise ValueError('Pixel outside the observation')
        ray = self.t[:3, :3] @ np.linalg.solve(self.k, [u, v, 1])
        origin = self.t[:3, 3]
        if mode == 'plane':
            if abs(ray[2]) < 1e-8:
                raise ValueError('Viewing ray is parallel to the work plane')
            d = (z-origin[2])/ray[2]
        elif mode == 'surface':
            ui, vi = int(u), int(v)
            patch = self.depth[max(0, vi-2):vi+3, max(0, ui-2):ui+3]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            if valid.size < max(1, patch.size//3):
                raise ValueError('No reliable depth at this pixel')
            d = float(np.median(valid))
        else:
            raise ValueError('Pick mode must be plane or surface')
        if d <= 0:
            raise ValueError('Selected point lies behind the camera')
        return origin+ray*d

    def public(self):
        image = io.BytesIO()
        Image.fromarray(self.rgb).save(image, format='JPEG', quality=85)
        points = self.world_points()[::6, ::6].reshape(-1, 3)
        colors = self.rgb[::6, ::6].reshape(-1, 3)
        valid = np.isfinite(points).all(axis=1) & (np.abs(points) < 2).all(axis=1)
        return {'id': self.id, 'timestamp': self.timestamp, 'width': self.rgb.shape[1], 'height': self.rgb.shape[0],
                'image': 'data:image/jpeg;base64,'+base64.b64encode(image.getvalue()).decode(),
                'k': self.k.tolist(), 't_world_camera': self.t.tolist(),
                'points': np.round(points[valid], 4).tolist(), 'colors': colors[valid].tolist(),
                'valid_depth_fraction': float(np.mean(np.isfinite(self.depth) & (self.depth > 0))),
                'unknown_geometry': 'Unobserved and occluded surfaces are not reconstructed'}
