"""HTTP application: the console page and the JSON API over one :class:`urai.service.Service`.

Security model (see docs/safety.md): the server binds to 127.0.0.1 by default. Binding to any other address
requires a token, sent as ``Authorization: Bearer <token>`` or ``X-URAI-Token``; a browser opens
``/?token=<token>`` once and receives an HttpOnly cookie for the rest of the session. Without a token only
loopback Host headers are accepted, which also blocks DNS-rebinding pages. State-changing requests from another
origin are refused either way.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from .live import LiveStream, live_settings_from_env
from .routes import drafting, meta, motion, observation

STATIC = Path(__file__).parent/'static'
TOKEN_COOKIE = 'urai_token'
TOKEN_ENV = 'URAI_TOKEN'
LOOPBACK_NAMES = ('localhost',)
log = logging.getLogger('urai')


def is_loopback(host):
    """True for 127.0.0.0/8, ::1 and ``localhost``."""
    if host in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host.strip('[]')).is_loopback
    except ValueError:
        return False


def _request_token(request):
    supplied = request.headers.get('authorization', '')
    if supplied.lower().startswith('bearer '):
        return supplied[7:].strip()
    return request.headers.get('x-urai-token') or request.cookies.get(TOKEN_COOKIE) or ''


def _host_name(request):
    return urlparse('//'+(request.headers.get('host') or '')).hostname or ''


def install_security(app, token):
    """Token or loopback-Host check on every request, same-origin check on every state-changing one."""

    @app.middleware('http')
    async def guard(request: Request, call_next):
        if token is None:
            if not is_loopback(_host_name(request)):
                return JSONResponse({'detail': 'Without a token the server only answers loopback Host names'},
                                    status_code=403)
        elif request.url.path == '/' and request.query_params.get('token') is not None:
            if not hmac.compare_digest(request.query_params['token'].encode(), token.encode()):
                return JSONResponse({'detail': 'Invalid token'}, status_code=401)
            response = RedirectResponse('/', status_code=303)
            response.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite='strict')
            return response
        elif not hmac.compare_digest(_request_token(request).encode(), token.encode()):
            return JSONResponse({'detail': 'Missing or invalid token'}, status_code=401)
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
            origin = request.headers.get('origin')
            if origin and urlparse(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail': 'Cross-origin control requests are not allowed'}, status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        return response


def create_app(service, token=None):
    """The FastAPI application for ``service``; ``token`` None means loopback-only access without a token."""
    live = LiveStream(service.backend, **live_settings_from_env())

    @asynccontextmanager
    async def lifespan(app):
        yield
        live.close()

    app = FastAPI(title='URAI', version='0.1.0', lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.live = live
    install_security(app, token)

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        # Refusals of the planner and the service carry a reason meant for the caller.
        return JSONResponse({'detail': str(exc)}, status_code=409)

    @app.exception_handler(Exception)
    async def error(request, exc):
        log.exception('Unhandled error in %s %s', request.method, request.url.path)
        return JSONResponse({'detail': 'Internal server error; see the server log'}, status_code=500,
                            headers={'Cache-Control': 'no-store'})

    app.include_router(meta.build_router())
    app.include_router(observation.build_router(service, live))
    app.include_router(drafting.build_router(service))
    app.include_router(motion.build_router(service))
    app.mount('/static', StaticFiles(directory=STATIC), name='static')
    return app


def apply_settings_file(service, path):
    """Apply a JSON body of ``PUT /api/settings`` (for example configs/paper-settings.json) at start-up."""
    settings = json.loads(Path(path).read_text())
    return service.set_settings({k: v for k, v in settings.items() if not k.startswith('_')})


def serve(service, host='127.0.0.1', port=7860, token=None, log_level='warning'):
    """Run the HTTP server in the calling thread until it is stopped."""
    import uvicorn
    if token is not None and not token:
        raise ValueError('An empty token is not a token')
    if not is_loopback(host) and token is None:
        raise ValueError(f'Refusing to bind {host} without a token: set --token or {TOKEN_ENV}')
    uvicorn.run(create_app(service, token), host=host, port=port, log_level=log_level)


def add_server_arguments(parser, default_port):
    parser.add_argument('--host', default='127.0.0.1', help='bind address (default 127.0.0.1; others need a token)')
    parser.add_argument('--port', type=int, default=default_port)
    parser.add_argument('--token', default=os.environ.get(TOKEN_ENV),
                        help=f'access token (default: ${TOKEN_ENV}); required for non-loopback hosts')
    parser.add_argument('--settings', help='JSON settings applied at start-up, e.g. configs/paper-settings.json')
    parser.add_argument('--log-dir', help='write one JSON record per executed motion into this directory')


def main(argv=None):
    """``urai``: the console and API on the synthetic simulation backend (no hardware)."""
    from .backend import SimulationBackend
    from .service import Service
    parser = argparse.ArgumentParser(description='URAI on the synthetic simulation backend. '
                                                 'The real robot (or its kinematic simulator) is started with urai-robot.')
    add_server_arguments(parser, default_port=7861)
    args = parser.parse_args(argv)
    service = Service(SimulationBackend(), args.log_dir)
    if args.settings:
        apply_settings_file(service, args.settings)
    serve(service, args.host, args.port, args.token)


if __name__ == '__main__':
    main()
