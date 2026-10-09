"""Access control and error hygiene of the real HTTP app over a real service (docs/safety.md, section 2)."""
from __future__ import annotations

import argparse
import logging

import pytest
from conftest import make_client
from fastapi.testclient import TestClient

from urai.app import TOKEN_COOKIE, add_server_arguments, create_app, is_loopback, serve

TOKEN = 'correct-horse-battery-staple'
#: A host the server is reachable under from the network; only allowed with a token.
LAN_HOST = '203.0.113.20'  # documentation range (RFC 5737), not loopback
REMOVED_ENDPOINTS = ('/api/rekep', '/api/rekep/replan', '/api/rekep/follow', '/api/rekep/track',
                     '/api/current-feedback/left/sample', '/api/transfer')
GENERIC_500 = {'detail': 'Internal server error; see the server log'}


def straight_spec():
    return {'path': {'mode': 'waypoints', 'points': [[.25, .12, .24], [.27, .12, .24]]}, 'speed': .02}


# ---------------------------------------------------------------------------- without a token

@pytest.mark.parametrize('host', ['127.0.0.1', '127.0.0.2', 'localhost', '[::1]'])
def test_without_a_token_loopback_hosts_are_answered(sim_service, host):
    response = make_client(sim_service, host=host).get('/api/state')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'


@pytest.mark.parametrize('host', [LAN_HOST, 'robot.example', '127.0.0.1.nip.io', 'localhost.example'])
def test_without_a_token_other_hosts_are_refused(sim_service, host):
    """A non-loopback Host is a network client or a DNS-rebinding page; neither reaches any endpoint."""
    client = make_client(sim_service, host=host)
    response = client.get('/api/state')
    assert response.status_code == 403 and 'loopback' in response.json()['detail']
    assert client.get('/').status_code == 403
    assert client.post('/api/observe', json={}).status_code == 403
    assert sim_service.frame is None and sim_service.revision == 0


@pytest.mark.parametrize('host, loopback', [
    ('127.0.0.1', True), ('127.8.9.1', True), ('::1', True), ('[::1]', True), ('localhost', True),
    ('0.0.0.0', False), ('::', False), (LAN_HOST, False), ('localhost.example', False), ('', False)])
def test_loopback_detection(host, loopback):
    assert is_loopback(host) is loopback


# ---------------------------------------------------------------------------- with a token

def test_with_a_token_requests_without_it_or_with_a_wrong_one_are_refused(sim_service):
    client = make_client(sim_service, token=TOKEN, host=LAN_HOST)
    assert client.get('/api/state').status_code == 401
    assert client.get('/').status_code == 401
    for headers in ({'Authorization': 'Bearer wrong'}, {'X-URAI-Token': 'wrong'}, {'Authorization': TOKEN},
                    {'Authorization': f'Basic {TOKEN}'}):
        response = client.get('/api/state', headers=headers)
        assert response.status_code == 401, headers
        assert response.json() == {'detail': 'Missing or invalid token'}
    client.cookies.set(TOKEN_COOKIE, 'wrong')
    assert client.get('/api/state').status_code == 401
    # The query string authenticates the console link only, not API calls.
    client.cookies.clear()
    assert client.get(f'/api/state?token={TOKEN}').status_code == 401
    assert client.post('/api/observe', json={}).status_code == 401
    assert sim_service.frame is None


def test_the_token_is_accepted_as_bearer_header_custom_header_or_cookie(sim_service):
    client = make_client(sim_service, token=TOKEN, host=LAN_HOST)
    assert client.get('/api/state', headers={'Authorization': f'Bearer {TOKEN}'}).status_code == 200
    assert client.get('/api/state', headers={'authorization': f'bearer {TOKEN}'}).status_code == 200
    assert client.get('/api/state', headers={'X-URAI-Token': TOKEN}).status_code == 200
    client.cookies.set(TOKEN_COOKIE, TOKEN)
    assert client.get('/api/state').status_code == 200


def test_with_a_token_loopback_clients_need_it_too(sim_service):
    client = make_client(sim_service, token=TOKEN)
    assert client.get('/api/state').status_code == 401
    assert client.get('/api/state', headers={'X-URAI-Token': TOKEN}).status_code == 200


def test_opening_the_console_with_the_token_sets_an_http_only_cookie(sim_service):
    client = make_client(sim_service, token=TOKEN, host=LAN_HOST)
    response = client.get(f'/?token={TOKEN}', follow_redirects=False)
    assert response.status_code == 303 and response.headers['location'] == '/'
    cookie = response.headers['set-cookie']
    assert cookie.startswith(f'{TOKEN_COOKIE}={TOKEN};')
    assert 'httponly' in cookie.lower() and 'samesite=strict' in cookie.lower()
    page = client.get('/')          # the browser follows the redirect and sends the cookie from now on
    assert page.status_code == 200 and page.headers['content-type'].startswith('text/html')
    assert client.get('/api/state').status_code == 200


def test_a_wrong_token_in_the_console_link_sets_no_cookie(sim_service):
    client = make_client(sim_service, token=TOKEN, host=LAN_HOST)
    response = client.get('/?token=wrong', follow_redirects=False)
    assert response.status_code == 401 and 'set-cookie' not in response.headers
    assert response.json() == {'detail': 'Invalid token'}
    assert client.get('/api/state').status_code == 401


# ---------------------------------------------------------------------------- browser origins

@pytest.mark.parametrize('token', [None, TOKEN])
def test_state_changing_requests_from_another_origin_are_refused(sim_service, token):
    client = make_client(sim_service, token=token)
    auth = {} if token is None else {'X-URAI-Token': token}
    for origin in ('http://evil.example', 'http://127.0.0.1:9999', 'http://localhost', 'null'):
        for method, path, body in (('POST', '/api/observe', {}), ('PUT', '/api/settings', {'motion_profile': 'fast'}),
                                   ('DELETE', '/api/draft', None), ('POST', '/api/cancel', None)):
            response = client.request(method, path, json=body, headers={**auth, 'Origin': origin})
            assert response.status_code == 403, (origin, method, path)
            assert response.json() == {'detail': 'Cross-origin control requests are not allowed'}
    assert sim_service.frame is None and sim_service.motion_profile == 'normal'
    assert sim_service.revision == 0 and sim_service.cancel_epoch == 0
    # The console itself is same-origin; a script sends no Origin at all.
    assert client.post('/api/observe', json={}, headers={**auth, 'Origin': 'http://127.0.0.1'}).status_code == 200
    assert client.post('/api/observe', json={}, headers=auth).status_code == 200
    # Reads change nothing, and without CORS headers a foreign page cannot read the reply.
    read = client.get('/api/state', headers={**auth, 'Origin': 'http://evil.example'})
    assert read.status_code == 200 and 'access-control-allow-origin' not in read.headers


# ---------------------------------------------------------------------------- surface

def test_interactive_docs_and_the_schema_are_not_served(sim_service):
    client = make_client(sim_service)
    for path in ('/docs', '/redoc', '/openapi.json', '/docs/oauth2-redirect'):
        assert client.get(path).status_code == 404, path


@pytest.mark.parametrize('path', REMOVED_ENDPOINTS)
def test_removed_endpoints_do_not_exist(sim_service, path):
    client = make_client(sim_service)
    assert client.post(path, json={}).status_code == 404
    assert client.get(path).status_code == 404


def test_the_grasp_line_endpoint_is_kept(sim_service):
    """``/api/grasp`` stays: it answers (here with a refusal for the missing observation) instead of 404."""
    response = make_client(sim_service).post('/api/grasp', json={})
    assert response.status_code == 409 and response.json() == {'detail': 'Stale observation'}


def test_an_unexpected_error_returns_a_generic_500_and_is_logged_on_the_server(sim_service, caplog):
    """The exception text stays in the server log.

    Every request the API defines validates its input, so no endpoint fails unexpectedly on purpose. The real
    application gets one extra route that raises, which exercises the real middleware and exception handler.
    """
    app = create_app(sim_service)

    @app.get('/api/raise-for-test')
    def raise_for_test():
        raise RuntimeError('secret detail 1234')

    client = TestClient(app, base_url='http://127.0.0.1', raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR, logger='urai'):
        response = client.get('/api/raise-for-test')
    assert response.status_code == 500
    assert response.json() == GENERIC_500
    assert response.headers['cache-control'] == 'no-store'
    assert 'secret detail' not in response.text and 'RuntimeError' not in response.text
    assert 'secret detail 1234' in caplog.text and 'GET /api/raise-for-test' in caplog.text
    assert client.get('/api/state').status_code == 200


def test_missing_fields_are_refusals_not_server_errors(sim_service):
    client = make_client(sim_service)
    response = client.post('/api/execute', json={})
    assert response.status_code == 409 and response.json() == {'detail': 'Missing field(s): preview_id'}
    observation = client.post('/api/observe', json={}).json()
    response = client.post('/api/pick', json={'observation_id': observation['id']})
    assert response.status_code == 409 and response.json() == {'detail': 'Missing field(s): u, v'}
    response = client.put('/api/draft', json={'observation_id': observation['id'], 'arms': {'left': 'x'}})
    assert response.status_code == 409 and response.json() == {'detail': 'Each arm draft must be a JSON object'}
    assert sim_service.execution['state'] == 'idle'


def test_planner_and_service_refusals_keep_their_reason(sim_service):
    """Refusals (ValueError) are meant for the caller and come back as 409 with their text."""
    response = make_client(sim_service).post('/api/compile')
    assert response.status_code == 409 and response.json() == {'detail': 'Create a trajectory draft first'}


def test_a_draft_with_path_constraints_is_refused(sim_service):
    client = make_client(sim_service)
    observation = client.post('/api/observe', json={}).json()
    saved = client.put('/api/draft', json={'observation_id': observation['id'], 'arms': {'left': straight_spec()}})
    assert saved.status_code == 200
    revision = sim_service.revision
    for merge in (False, True):
        refused = client.put('/api/draft', json={'observation_id': observation['id'], 'merge': merge, 'arms': {
            'right': {**straight_spec(), 'constraints': [{'type': 'keep_above', 'z': .1}]}}})
        assert refused.status_code == 409
        assert refused.json() == {'detail': 'Path constraints are not supported by this release'}
    assert sim_service.revision == revision
    assert client.get('/api/draft').json()['arms'] == {'left': straight_spec()}


# ---------------------------------------------------------------------------- serving

@pytest.mark.parametrize('host, token, reason', [
    ('0.0.0.0', None, 'without a token'), ('::', None, 'without a token'), (LAN_HOST, None, 'without a token'),
    ('127.0.0.1', '', 'empty token'), ('0.0.0.0', '', 'empty token')])
def test_serve_refuses_to_start_without_a_usable_token(sim_service, host, token, reason):
    """Checked before the server is created: nothing is bound when these raise."""
    with pytest.raises(ValueError, match=reason):
        serve(sim_service, host=host, port=7860, token=token)


def test_the_command_line_binds_to_loopback_by_default():
    parser = argparse.ArgumentParser()
    add_server_arguments(parser, default_port=7861)
    args = parser.parse_args([])
    assert (args.host, args.port) == ('127.0.0.1', 7861)
    args = parser.parse_args(['--host', '0.0.0.0', '--token', TOKEN, '--settings', 'configs/paper-settings.json'])
    assert (args.host, args.token, args.settings) == ('0.0.0.0', TOKEN, 'configs/paper-settings.json')
