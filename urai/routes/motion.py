"""Endpoints that plan against the live robot and move it: preview, execute, home, the task queue, cancel."""
from __future__ import annotations

from fastapi import APIRouter, Request
from ..backend import ARMS
from .common import require_fields


def build_router(service):
    router = APIRouter()

    @router.post('/api/preview')
    def preview(data: dict | None = None):
        """Plan and check the draft. May send CAN commands that restore control and hold the current pose."""
        data = data or {}
        return service.preview(data.get('expected_revision'), data.get('expected_cancel_epoch'))

    @router.post('/api/execute')
    def execute(data: dict, request: Request):
        """Start the motion of a preview, recording who asked for it.

        The console, a person's own script and an agent all reach this route; the run record keeps the caller's
        user agent, origin and address so runs can be attributed afterwards. The console sends an Origin
        (it is same-origin); a command line does not.
        """
        caller = {'agent': (request.headers.get('user-agent') or '')[:120],
                  'origin': request.headers.get('origin') or request.headers.get('referer') or '',
                  'address': request.client.host if request.client else ''}
        caller['kind'] = 'console' if caller['origin'] else 'script'
        require_fields(data, 'preview_id')
        return service.execute(data['preview_id'], data.get('cancel_epoch'), caller=caller)

    @router.post('/api/home')
    def home(data: dict | None = None):
        data = data or {}
        if set(data) - {'arms', 'cancel_epoch', 'open_grippers'}:
            raise ValueError('home 只接受 arms、cancel_epoch 和 open_grippers')
        if not isinstance(data.get('open_grippers', True), bool):
            raise ValueError('open_grippers 必须为布尔值')
        return service.home(data.get('arms', list(ARMS)), data.get('cancel_epoch'), data.get('open_grippers', True))

    @router.post('/api/tasks')
    def submit_tasks(data: dict):
        return service.tasks.submit(data)

    @router.get('/api/tasks')
    def tasks():
        return service.tasks.status()

    @router.delete('/api/tasks')
    def cancel_tasks():
        return service.tasks.cancel()

    @router.post('/api/recover')
    def recover():
        return service.recover()

    @router.post('/api/cancel')
    def cancel():
        return service.cancel()

    return router
