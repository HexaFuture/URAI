"""State, settings and everything the cameras produce: observations, wrist frames and the live view."""
from __future__ import annotations

import base64
from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse
from ..backend import ARMS
from ..live import BOUNDARY
from .common import require_current_observation, require_fields


def jpeg_response(data_url):
    """JPEG bytes of a ``data:image/jpeg;base64,...`` URL."""
    return Response(base64.b64decode(data_url.split(',', 1)[1]), media_type='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


def build_router(service, live):
    router = APIRouter()

    @router.get('/api/state')
    def state():
        result = service.status()
        result['live'] = live.status()
        return result

    @router.get('/api/settings')
    def settings():
        return service.settings()

    @router.put('/api/settings')
    def set_settings(data: dict):
        return service.set_settings(data)

    @router.post('/api/observe')
    def observe(data: dict | None = None):
        data = data or {}
        if set(data)-{'preserve_draft'} or not isinstance(data.get('preserve_draft', False), bool):
            raise ValueError('preserve_draft must be a boolean')
        return service.observe(preserve_draft=data.get('preserve_draft', False))

    @router.get('/api/observation')
    def observation():
        if service.frame is None:
            raise ValueError('Capture an observation first')
        return service.frame.public()

    @router.get('/api/observation.jpg')
    def observation_snapshot():
        """The frame the draft is built against, as JPEG bytes; the pixels of a stroke refer to this frame."""
        with service.lock:
            if not service.frame:
                raise ValueError('还没有观测，请先 POST /api/observe')
            data = service.frame.public()['image']
        return jpeg_response(data)

    @router.post('/api/pick')
    def pick(data: dict):
        require_fields(data, 'u', 'v')
        with service.lock:
            require_current_observation(service, data)
            p = service.frame.pick(data['u'], data['v'], data.get('mode', 'plane'), data.get('z', .15))
            return {'xyz': p.tolist()}

    def wrist_frame(arm):
        if arm not in ARMS:
            raise ValueError('arm must be left or right')
        if service.backend.mode != 'real':
            raise ValueError('Wrist camera requires the real backend')
        return service.backend.capture('hand_'+arm).public()

    @router.get('/api/wrist/{arm}/observation')
    def wrist_observation(arm: str):
        # Read-only: the head observation and the draft built on it stay untouched.
        return {**wrist_frame(arm), 'camera': 'hand_'+arm}

    @router.get('/api/wrist/{arm}/image.jpg')
    def wrist_image(arm: str):
        return jpeg_response(wrist_frame(arm)['image'])

    @router.get('/api/live.jpg')
    def live_snapshot():
        return Response(live.snapshot(), media_type='image/jpeg')

    @router.get('/api/live.mjpg')
    def live_stream():
        # One shared camera worker; the subscription ends when this client disconnects.
        return StreamingResponse(live.multipart(), media_type=f'multipart/x-mixed-replace; boundary={BOUNDARY}')

    return router
