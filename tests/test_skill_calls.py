"""No hardware: one-call HTTP semantics, interruption and response-loss recovery."""
import json
from pathlib import Path
import tempfile
import threading
import types
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

try:
    from urai.skill_calls import SkillCalls, install_skill_calls
except ModuleNotFoundError:
    from skill_calls import SkillCalls, install_skill_calls


class FakeService:
    def __init__(self, root):
        self.lock = threading.RLock()
        self.log_dir = Path(root)
        self.frame = types.SimpleNamespace(id='obs1')
        self.revision = 1
        self.cancel_epoch = 0
        self.worker = None
        self.execution = {'state': 'idle'}
        self.calls = []
        self.planning = threading.Event()
        self.plan_gate = threading.Event(); self.plan_gate.set()
        self.execute_gate = threading.Event(); self.execute_gate.set()
        self.fail_preview = False
        self.refresh_preview = False
        self.lose_dispatch = False
        self.empty_grasp = False

    def _editable(self):
        if self.execution['state'] in ('running', 'planning'):
            raise ValueError('busy')

    def plan(self, name, data):
        self._editable()
        self.calls.append('plan')
        self.planning.set()
        self.plan_gate.wait(2)
        self.revision += 1
        return {'revision': self.revision, 'summary': name}

    def preview(self, revision, epoch):
        self._editable()
        self.calls.append('preview')
        if epoch != self.cancel_epoch:
            raise ValueError('cancelled')
        if self.fail_preview:
            raise ValueError('IK or collision failed')
        return {'id': 'p1', 'revision': revision, 'observation_id': 'changed' if self.refresh_preview else 'obs1'}

    def execute(self, preview_id, epoch, caller=None):
        self._editable()
        self.calls.append('execute')
        if self.lose_dispatch:
            raise TimeoutError('dispatch response unknown')
        self.execution = {'state': 'running', 'id': preview_id}
        def worker():
            self.execute_gate.wait(2)
            with self.lock:
                if self.cancel_epoch != epoch:
                    self.execution.update(state='cancelled')
                elif self.empty_grasp:
                    self.execution.update(state='idle', last_error='empty', result={'completed': False, 'outcome': 'empty_grasp', 'ready_for_next': True})
                else:
                    self.execution.update(state='completed', result={'completed': True})
        self.worker = threading.Thread(target=worker)
        self.worker.start()
        return {'state': 'running', 'id': preview_id}


class SkillCallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.s = FakeService(self.temp.name)
        self.app = FastAPI()
        from fastapi.responses import JSONResponse
        @self.app.exception_handler(ValueError)
        async def error(request, exc):
            return JSONResponse({'detail': str(exc)}, status_code=409)
        @self.app.post('/api/skills/{name}')
        def legacy(name: str, data: dict):
            return self.s.plan(name, data)
        @self.app.post('/api/cancel')
        def cancel():
            with self.s.lock:
                self.s.cancel_epoch += 1
            return {'cancelled': True}
        self.runner = install_skill_calls(self.app, self.s, self.s.plan)
        self.client = TestClient(self.app)
        self.addCleanup(self.client.close)

    def post(self, **changes):
        body = {'request_id': 'r1', 'observation_id': 'obs1', 'arm': 'right', 'pixels': [[1, 2], [3, 4]], 'wait_s': 2}
        body.update(changes)
        return self.client.post('/api/skills/grasp_line/run', json=body)

    def test_one_post_finishes_all_three_steps(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['state'], 'completed')
        self.assertEqual(self.s.calls, ['plan', 'preview', 'execute'])
        self.assertEqual(set(response.json()['timings_s']), {'plan', 'preview', 'dispatch', 'execution_wait', 'total'})
        record = json.loads((self.runner.root/'r1.json').read_text())
        self.assertEqual(record['state'], 'completed')

    def test_duplicate_request_returns_original_and_new_wait_is_allowed(self):
        self.post()
        self.assertEqual(self.post(wait_s=0).json()['state'], 'completed')
        self.assertEqual(self.s.calls.count('execute'), 1)
        self.assertEqual(self.post(arm='left').status_code, 409)

    def test_preview_failure_never_executes(self):
        self.s.fail_preview = True
        result = self.post().json()
        self.assertEqual(result['state'], 'error')
        self.assertIn('IK', result['error'])
        self.assertNotIn('execute', self.s.calls)

    def test_stale_input_rejected_before_planning(self):
        self.assertEqual(self.post(observation_id='old').status_code, 409)
        self.assertEqual(self.s.calls, [])

    def test_preview_refresh_cannot_silently_change_target(self):
        self.s.refresh_preview = True
        self.assertEqual(self.post().json()['state'], 'error')
        self.assertNotIn('execute', self.s.calls)

    def test_async_completion_does_not_need_an_execute_request(self):
        self.s.execute_gate.clear()
        try:
            self.assertEqual(self.post(wait_s=0).status_code, 202)
        finally:
            self.s.execute_gate.set()
        result = self.client.get('/api/skill-runs/r1?wait_s=2').json()
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(self.s.calls.count('execute'), 1)

    def test_other_writers_rejected_while_cancel_is_responsive(self):
        self.s.plan_gate.clear()
        try:
            self.post(wait_s=0)
            self.assertTrue(self.s.planning.wait(1))
            self.assertEqual(self.client.post('/api/skills/grasp_line', json={}).status_code, 409)
            self.assertEqual(self.post(request_id='other', wait_s=0).status_code, 409)
            self.assertEqual(self.client.post('/api/cancel').status_code, 200)
        finally:
            self.s.plan_gate.set()
        result = self.runner.status('r1', 2)
        self.assertEqual(result['state'], 'cancelled')
        self.assertNotIn('execute', self.s.calls)
        self.s._editable()

    def test_dispatch_ambiguity_never_replays(self):
        self.s.lose_dispatch = True
        self.assertEqual(self.post().json()['state'], 'unknown')
        self.assertEqual(self.post().json()['state'], 'unknown')
        self.assertEqual(self.post(request_id='new').status_code, 409)
        self.assertEqual(self.s.calls.count('execute'), 1)

    def test_empty_grasp_recovery_is_failure_but_allows_next_attempt(self):
        self.s.empty_grasp = True
        result = self.post().json()
        self.assertEqual(result['state'], 'error')
        self.assertEqual(result['result']['outcome'], 'empty_grasp')
        self.assertTrue(result['result']['ready_for_next'])
        self.s.empty_grasp = False
        self.assertEqual(self.post(request_id='retry-new-target').json()['state'], 'completed')

    def test_cancelled_execution_is_terminal(self):
        self.s.execute_gate.clear()
        try:
            self.post(wait_s=0)
            self.client.post('/api/cancel')
        finally:
            self.s.execute_gate.set()
        self.assertEqual(self.runner.status('r1', 2)['state'], 'cancelled')

    def test_legacy_skill_remains_plan_only(self):
        self.client.post('/api/skills/grasp_line', json={})
        self.assertEqual(self.s.calls, ['plan'])

    def test_restart_marks_nonterminal_as_unknown(self):
        root = Path(self.temp.name)/'restart'
        root.mkdir()
        (root/'r2.json').write_text(json.dumps({'request_id': 'r2', 'state': 'dispatching', 'fingerprint': 'x'}))
        restored = SkillCalls(self.s, self.s.plan, root)
        self.assertEqual(restored.status('r2')['state'], 'unknown')

    def test_malformed_control_fields_do_not_start(self):
        for body in ({'wait_s': 121}, {'wait_s': True}, {'request_id': '../bad'}, {'commit': False}, {'execute': False}, {'merge': True}):
            self.assertEqual(self.post(**body).status_code, 409)
        self.assertEqual(self.s.calls, [])

    def test_unknown_id_is_not_success(self):
        self.assertEqual(self.client.get('/api/skill-runs/missing').json()['state'], 'not_found')


if __name__ == '__main__':
    unittest.main()
