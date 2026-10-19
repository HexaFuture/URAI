"""One HTTP skill call: plan, preview, execute, and wait without resubmission.

Uses the existing Service pipeline, including its skill-specific return-home policy.
No hardware commands are sent by installation or result lookup.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse


TERMINAL = {'completed', 'error', 'cancelled', 'unknown'}


class SkillCalls:
    def __init__(self, service, plan_skill, root=None):
        self.service, self.plan_skill = service, plan_skill
        self.cv = threading.Condition(service.lock)
        self.root = Path(root or (service.log_dir / 'skill_calls' if service.log_dir else tempfile.mkdtemp(prefix='urai-skill-calls-')))
        self.root.mkdir(parents=True, exist_ok=True)
        self.records = {}
        self.active = None
        self.thread = None
        for path in self.root.glob('*.json'):
            record = json.loads(path.read_text())
            if record['state'] not in TERMINAL:
                record.update(state='unknown', error='Service restarted: reconcile the previous action; do not resubmit')
                self._save(record)
            self.records[record['request_id']] = record

    def blocks_current_thread(self):
        return self.active is not None and threading.current_thread() is not self.thread

    def _save(self, record):
        target = self.root / (record['request_id'] + '.json')
        tmp = target.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump(record, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(target)

    @staticmethod
    def _wait_value(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 120:
            raise ValueError('wait_s must be a number between 0 and 120')
        return value

    def start(self, name, data, caller=None):
        data = copy.deepcopy(data)
        request_id = data.pop('request_id', None)
        wait_s = self._wait_value(data.pop('wait_s', 20))
        if not isinstance(request_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', request_id):
            raise ValueError('Provide a unique request_id (1..80 letters, digits, _ or -)')
        if data.get('commit', True) is not True or 'execute' in data or 'merge' in data:
            raise ValueError('/run commits, previews and executes one skill; omit commit, execute and merge')
        data['commit'] = True
        fingerprint = hashlib.sha256(json.dumps([name, data], sort_keys=True, allow_nan=False).encode()).hexdigest()
        s = self.service
        with self.cv:
            old = self.records.get(request_id)
            if old:
                if old['fingerprint'] != fingerprint:
                    raise ValueError('request_id already belongs to different skill arguments')
            else:
                if self.active or any(r['state'] == 'unknown' for r in self.records.values()):
                    raise ValueError('Another skill call is active or unresolved; query its request_id')
                s._editable()
                if s.worker is not None and s.worker.is_alive():
                    raise ValueError('Previous robot worker is still finishing')
                if s.frame is None or data.get('observation_id') != s.frame.id:
                    raise ValueError('Stale observation: use the frame on which the pixels were chosen')
                epoch = s.cancel_epoch
                record = {'request_id': request_id, 'skill': name, 'fingerprint': fingerprint,
                          'state': 'accepted', 'phase': 'plan', 'timings_s': {}, 'submitted_at': time.time()}
                self._save(record)
                self.records[request_id] = record
                self.active = request_id
                self.thread = threading.Thread(target=self._run, args=(record, name, data, epoch, caller),
                                               name='urai-skill-call', daemon=True)
                try:
                    self.thread.start()
                except Exception as exc:
                    record.update(state='error', phase='done', error=str(exc))
                    self.active = None
                    self._save(record)
                    self.cv.notify_all()
        return self.status(request_id, wait_s)

    def status(self, request_id, wait_s=0):
        wait_s = self._wait_value(wait_s)
        with self.cv:
            if request_id not in self.records:
                return {'request_id': request_id, 'state': 'not_found',
                        'error': 'Acceptance is unknown; do not blindly resubmit'}
            record = self.records[request_id]
            self.cv.wait_for(lambda: record['state'] in TERMINAL, timeout=wait_s)
            result = copy.deepcopy(record)
            result.pop('fingerprint', None)
            result['status_url'] = '/api/skill-runs/' + request_id
            if self.active == request_id:
                result['progress'] = self.service.execution.get('progress', 0.)
            return result

    def _phase(self, record, phase, fn):
        with self.cv:
            record['phase'] = phase
        began = time.monotonic()
        try:
            return fn()
        finally:
            with self.cv:
                record['timings_s'][phase] = time.monotonic() - began

    def _run(self, record, name, data, epoch, caller):
        s, began = self.service, time.monotonic()
        dispatched, terminal = False, 'error'
        try:
            with s.lock:
                if s.cancel_epoch != epoch:
                    raise ValueError('Skill call cancelled before planning')
            draft = self._phase(record, 'plan', lambda: self.plan_skill(name, data))
            preview = self._phase(record, 'preview', lambda: s.preview(draft['revision'], epoch))
            with s.lock:
                if s.cancel_epoch != epoch:
                    raise ValueError('Skill call cancelled before execution')
                if preview['observation_id'] != data['observation_id'] or preview['revision'] != draft['revision']:
                    raise ValueError('Preview rebound the visual target; observe and choose pixels again')
                record.update(preview_id=preview['id'], revision=preview['revision'], state='dispatching')
                self._save(record)  # Persist intent before executing; uncertain actions are never replayed.
                dispatched = True
                execution = self._phase(record, 'dispatch', lambda: s.execute(preview['id'], epoch, caller=caller))
                worker = s.worker
                record.update(state='running', execution_id=execution['id'])
                self._save(record)
            self._phase(record, 'execution_wait', worker.join)
            with s.lock:
                execution = copy.deepcopy(s.execution)
                if execution.get('id') != record['execution_id']:
                    raise RuntimeError('Execution result changed before collection')
                result = execution.get('result') or {}
                terminal = execution['state'] if execution['state'] in {'completed', 'error', 'cancelled'} else 'unknown'
                if execution['state'] == 'idle' and result.get('outcome') == 'empty_grasp':
                    terminal = 'error'
                    record['error'] = execution.get('last_error', 'Empty grasp; returned home and ready for a new observation')
                record['execution'] = {k: execution[k] for k in ('id', 'state', 'stage', 'progress', 'elapsed_s', 'error', 'outcome') if k in execution}
                record['result'] = {k: result[k] for k in ('completed', 'elapsed_s', 'outcome', 'error', 'ready_for_next', 'holding_position') if k in result}
        except Exception as exc:
            terminal = 'unknown' if dispatched else ('cancelled' if s.cancel_epoch != epoch else 'error')
            record['error'] = str(exc)
        finally:
            with self.cv:
                record.update(state=terminal, phase='done')
                record['timings_s']['total'] = time.monotonic() - began
                try:
                    self._save(record)
                except Exception as exc:
                    record.update(state='unknown', error='Could not persist final result: ' + str(exc))
                self.active = None
                self.cv.notify_all()


def install_skill_calls(app, service, plan_skill, root=None):
    """Install on the existing app and service; keep all original HTTP routes."""
    if getattr(service, 'skill_calls', None) is not None:
        raise ValueError('Skill calls are already installed')
    calls = SkillCalls(service, plan_skill, root)
    original_editable = service._editable

    def editable():
        if calls.blocks_current_thread():
            raise ValueError('A skill call is active; wait or cancel before editing')
        original_editable()

    # Reserve edits across plan/preview/dispatch without holding the service lock
    # throughout planning or waiting. The existing cancel route remains responsive.
    service._editable = editable
    service.skill_calls = calls

    @app.post('/api/skills/{name}/run')
    def run_skill_once(name: str, data: dict, request: Request):
        caller = {'agent': (request.headers.get('user-agent') or '')[:120],
                  'origin': request.headers.get('origin') or request.headers.get('referer') or '',
                  'address': request.client.host if request.client else ''}
        caller['kind'] = 'console' if caller['origin'] else 'script'
        result = calls.start(name, data, caller)
        return JSONResponse(result, status_code=200 if result['state'] in TERMINAL else 202,
                            headers={'Cache-Control': 'no-store'})

    @app.get('/api/skill-runs/{request_id}')
    def skill_run_status(request_id: str, wait_s: float = 0):
        return calls.status(request_id, wait_s)

    app.openapi_schema = None
    return calls
