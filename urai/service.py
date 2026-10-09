from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from .backend import ARMS, Cancelled, RobotStateError
from .trajectory import compile_arms
from .settings import DEFAULT_MOTION_PROFILE, motion_limits, motion_profiles, normalize_pose_tolerance
from .tasks import TaskQueue

# How long a plan is allowed to stand before it is remade. It guards against executing a route planned for a
# scene that has moved on; execute() regenerates an expired preview from the same draft (see _renew_if_expired).
PREVIEW_TTL_S = 120.


class DispatchCancellation:
    """Cancellation acceptance and short hardware writes share one mutex."""
    def __init__(self):
        self.event = threading.Event()
        self.lock = threading.RLock()

    def clear(self):
        with self.lock:
            self.event.clear()

    def set(self):
        with self.lock:
            self.event.set()

    def is_set(self):
        return self.event.is_set()

    def wait(self, timeout):
        return self.event.wait(timeout)

    @contextmanager
    def dispatch(self):
        with self.lock:
            if self.event.is_set():
                raise Cancelled()
            yield


def public(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k != 'curve'}
    if isinstance(value, (list, tuple)):
        return [public(v) for v in value]
    return value


class Service:
    def __init__(self, backend, log_dir=None):
        self.backend = backend
        self.pose_tolerance = normalize_pose_tolerance()
        self.motion_profile = DEFAULT_MOTION_PROFILE
        self.home_speed_m_s = .8
        self.auto_prepare = {'refresh_observation': True, 'restore_can': True}
        self.lock = threading.RLock()
        self.frame = None; self.observation_start = None; self.draft = {}; self.preview_data = None; self.revision = 0
        self.execution = {'state': 'idle', 'progress': 0., 'elapsed_s': 0.}
        self.cancel_event = DispatchCancellation(); self.worker = None
        self.cancel_epoch = 0
        self.drop_point = None
        self.bottle_cap = None
        self.tasks = TaskQueue(self)
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _editable(self):
        if self.execution['state'] in ('running', 'cancelling', 'planning'):
            raise ValueError('An operation is running; wait or cancel before editing')
        if self.tasks.blocks_current_thread():
            raise ValueError('任务队列正在执行；请等待完成或先停止队列')

    def arm_calibration(self, arm):
        """Calibrated table height (m) and fingertip bias (mm) of one arm; zero in simulation."""
        model = self.backend.models.get(arm)
        if model is None:
            return 0., 0.
        return float(model.table_z_mm)/1000, float(model.fingertip_bias_mm)

    def table_z(self, arm=None):
        """Table height for one arm, or the mean of both when the arm is not chosen yet."""
        arms = (arm,) if arm else ('left', 'right')
        return float(np.mean([self.arm_calibration(a)[0] for a in arms]))

    def set_drop_point(self, xyz):
        with self.lock:
            point = np.asarray(xyz, dtype=float) if isinstance(xyz, (list, tuple)) and len(xyz) == 3 \
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xyz) else None
            if point is None or not np.isfinite(point).all() or np.any(np.abs(point) > 2):
                raise ValueError('投放点需要三个有限的世界坐标（米），绝对值不超过 2 m')
            self.drop_point = point.tolist()
            return {'xyz': self.drop_point}

    def status(self):
        with self.lock:
            result = {'backend': self.backend.info(), 'execution': copy.deepcopy(self.execution),
                      'observation_id': self.frame.id if self.frame else None, 'revision': self.revision,
                      'preview_id': self.preview_data['id'] if self.preview_data else None,
                      'pose_tolerance': copy.deepcopy(self.pose_tolerance), 'motion_profile':self.motion_profile, 'home_speed_m_s':self.home_speed_m_s}
            result['auto_prepare'] = copy.deepcopy(self.auto_prepare)
            result['table_checks'] = self.backend.table_checks
            result['tracking_limit_deg'] = self.backend.tracking_limit_deg
            result['cancel_epoch'] = self.cancel_epoch
            result['drop_point'] = copy.deepcopy(self.drop_point)
            result['bottle_cap'] = copy.deepcopy(self.bottle_cap)
            result['task_queue'] = self.tasks.summary()
            if self.execution['state'] in ('error', 'cancelled'):
                result['recovery_actions'] = [
                    {'name': 'clear_error', 'label': '清除错误并重新规划', 'method': 'POST', 'path': '/api/recover', 'body': {}},
                    {'name': 'home', 'label': '保持夹爪并回原位', 'method': 'POST', 'path': '/api/home',
                     'body': {'arms': list(ARMS), 'open_grippers': False}}]
            else:
                result['recovery_actions'] = []
            limits=motion_limits(self.motion_profile)
            result['backend'].update(motion_limits=limits,max_joint_speed_deg_s=limits['joint_speed_deg_s'],
                                     max_joint_acceleration_deg_s2=limits['joint_acceleration_deg_s2'])
        result['arms'] = self.backend.state()
        result['observation_stale'] = self._observation_stale(result['arms'])
        return result

    def settings(self):
        with self.lock:
            return {'pose_tolerance': copy.deepcopy(self.pose_tolerance), 'motion_profile':self.motion_profile, 'home_speed_m_s':self.home_speed_m_s,
                    'auto_prepare':copy.deepcopy(self.auto_prepare),'table_checks':self.backend.table_checks,
                    'tracking_limit_deg':self.backend.tracking_limit_deg,
                    'motion_profiles': motion_profiles(),
                    'revision': self.revision}

    def set_settings(self, data):
        with self.lock:
            self._editable()
            if not isinstance(data,dict) or not data or set(data)-{'pose_tolerance','motion_profile','auto_prepare','table_checks','tracking_limit_deg','home_speed_m_s'}:
                raise ValueError('请提供 pose_tolerance、motion_profile、auto_prepare、table_checks、tracking_limit_deg 或 home_speed_m_s 设置')
            home_speed = data.get('home_speed_m_s', self.home_speed_m_s)
            if type(home_speed) not in (int,float) or not np.isfinite(home_speed) or not .05 <= home_speed <= 1.:
                raise ValueError('回位速度需为 0.05–1.0 m/s')
            tracking = data.get('tracking_limit_deg', self.backend.tracking_limit_deg)
            if type(tracking) not in (int, float) or not np.isfinite(tracking) or not 0 <= tracking <= 90:
                raise ValueError('tracking_limit_deg 需为 0–90 度（0 表示关闭跟踪检查）')
            table_checks = data.get('table_checks', self.backend.table_checks)
            if not isinstance(table_checks, bool):
                raise ValueError('table_checks 只接受布尔值')
            value = normalize_pose_tolerance(data.get('pose_tolerance',self.pose_tolerance))
            profile = motion_limits(data.get('motion_profile',self.motion_profile))['profile']
            changes = data.get('auto_prepare', {})
            if not isinstance(changes,dict) or set(changes)-set(self.auto_prepare) or any(not isinstance(v,bool) for v in changes.values()):
                raise ValueError('auto_prepare 只接受 refresh_observation 和 restore_can 布尔值')
            self.auto_prepare.update(changes)
            self.backend.table_checks = table_checks
            self.backend.tracking_limit_deg = float(tracking)
            self.pose_tolerance = value
            self.motion_profile = profile
            self.home_speed_m_s = float(home_speed)
            self.preview_data = None
            self.revision += 1
            if self.execution['state'] == 'ready':
                self.execution = {'state': 'idle', 'progress': 0., 'elapsed_s': 0.}
            return self.settings()

    def _observation_stale(self, current):
        if not self.frame or not self.observation_start:
            return True
        return any(np.max(np.abs(np.asarray(current[a]['joints_deg'])-self.observation_start[a]['joints_deg'])) > .5
                   for a in ('left', 'right'))

    def observe(self, preserve_draft=False):
        with self.lock:
            self._editable()
            # The head camera is fixed in world coordinates. Arm motion does not invalidate RGB-D.
            # Planning/dispatch still validate the measured robot state independently.
            frame = self.backend.capture()
            after = self.backend.state()
            self.frame = frame; self.observation_start = after
            if preserve_draft and self.draft:
                self.draft = copy.deepcopy(self.draft)
                self.draft['observation_id'] = frame.id
            else:
                self.draft = {}
            self.preview_data = None; self.revision += 1
            self.execution = {'state':'idle','progress':0.,'elapsed_s':0.}
            return self.frame.public()

    def recover(self):
        """Acknowledge a stopped operation; invalidate its plan without commanding motion."""
        with self.lock:
            self._editable()
            if self.worker is not None and self.worker.is_alive():
                raise ValueError('停止收尾尚未完成，请稍后重试')
            previous = copy.deepcopy(self.execution)
            self.cancel_epoch += 1
            self.cancel_event.clear()
            self.preview_data = None
            self.execution = {'state': 'idle', 'progress': 0., 'elapsed_s': 0.}
            self.revision += 1
            return {'execution': copy.deepcopy(self.execution), 'previous_execution': previous,
                    'revision': self.revision, 'cancel_epoch': self.cancel_epoch}

    def clear_draft(self):
        with self.lock:
            self._editable()
            if self.execution['state'] in ('error', 'cancelled'):
                self.recover()
            self.draft = {}
            self.preview_data = None
            if self.execution['state'] == 'ready':
                self.execution = {'state': 'idle', 'progress': 0., 'elapsed_s': 0.}
            self.revision += 1
            return {'revision': self.revision, 'draft': {}}

    def set_draft(self, draft, merge=False):
        """Replace the draft, or with merge keep the arms this one does not name.

        Merging is how a two-armed move gets drawn one arm at a time: the first stroke drafts one arm, the
        second names the other, and both run together. The kept arm is only valid while the picture it was
        planned against is still the current one - a stroke planned on a stale frame grasps where the cloth
        no longer is - so the observation has to match before anything is carried over.
        """
        with self.lock:
            self._editable()
            if draft.get('expected_revision', self.revision) != self.revision:
                raise ValueError('草稿版本已变化，请重新画线')
            if not self.frame or draft.get('observation_id') != self.frame.id:
                raise ValueError('Observation is stale; refresh and rebuild the draft')
            arms = draft.get('arms')
            if not isinstance(arms, dict) or not arms or set(arms)-{'left', 'right'}:
                raise ValueError('Choose left, right, or both arms')
            if any(not isinstance(spec, dict) for spec in arms.values()):
                raise ValueError('Each arm draft must be a JSON object')
            if any('constraints' in spec for spec in arms.values()):
                raise ValueError('Path constraints are not supported by this release')
            if merge:
                kept = (self.draft or {}).get('arms') or {}
                if kept and (self.draft or {}).get('observation_id') == draft.get('observation_id'):
                    draft = {**draft, 'arms': {**kept, **arms}}
            if len(json.dumps(draft, allow_nan=False)) > 64000:
                raise ValueError('Draft is too large')
            from .skills.bottle_cap import stage_metadata
            stage_metadata({'input_draft':draft})
            self.draft = copy.deepcopy(draft); self.draft.pop('expected_revision',None)
            self.preview_data = None; self.revision += 1
            return {'revision': self.revision, 'draft': copy.deepcopy(self.draft)}

    def preview(self, expected_revision=None, expected_cancel_epoch=None):
        begun = time.monotonic()
        with self.lock:
            self._editable()
            if expected_revision is not None and expected_revision != self.revision:
                raise ValueError('草稿版本已变化，自动执行已终止')
            if expected_cancel_epoch is not None and expected_cancel_epoch != self.cancel_epoch:
                raise ValueError('自动搬运已取消')
            if not self.draft:
                raise ValueError('Create a trajectory draft first')
            draft = copy.deepcopy(self.draft); rev = self.revision
            tolerance = copy.deepcopy(self.pose_tolerance)
            profile = self.motion_profile
            options = copy.deepcopy(self.auto_prepare)
            self.preview_data = None
            self.cancel_event.clear()
            self.execution = {'state': 'planning', 'kind': 'preview', 'stage': 'compile',
                              'progress': 0., 'elapsed_s': 0., 'started_at': time.time()}
        progress = self._progress_reporter(begun)
        try:
            preparation = {'restored_arms': [], 'observation_refreshed': False}
            for attempt in range(2):
                try:
                    progress(stage='prepare',arm=None,done=0,total=1)
                    recovered = self.backend.prepare(self.cancel_event,progress,restore_can=options['restore_can'])
                    preparation['restored_arms'] = sorted(set(preparation['restored_arms']+recovered['restored_arms']))
                    start = self.backend.state()
                    stale = self._observation_stale(start)
                    if options['refresh_observation'] and (stale or recovered['restored_arms']):
                        progress(stage='refresh',arm=None,done=0,total=1)
                        frame = self.backend.capture()
                        after = self.backend.state()
                        if any(np.max(np.abs(np.asarray(after[a]['joints_deg'])-start[a]['joints_deg'])) > .5 for a in after):
                            raise RobotStateError('Robot moved during automatic observation', reason='moved')
                        with self.lock:
                            if self.cancel_event.is_set():
                                raise Cancelled()
                            if self.revision != rev:
                                raise ValueError('Draft changed during automatic observation')
                            self.frame = frame; self.observation_start = after
                            draft['observation_id'] = frame.id
                            self.draft = copy.deepcopy(draft)
                            self.revision += 1; rev = self.revision
                        start = after; preparation['observation_refreshed'] = True
                    elif stale:
                        raise ValueError('Robot moved since observation; refresh the depth observation and rebuild the draft')
                    progress(stage='compile',arm=None,done=0,total=1)
                    from .skills.bottle_cap import validate_stage, PRECISION
                    cap_metadata=validate_stage(self, {'input_draft':draft}, start)
                    if cap_metadata:
                        tolerance = {k:min(tolerance[k],v) for k,v in PRECISION.items()}
                    if cap_metadata and cap_metadata['stage']=='place' and cap_metadata.get('worker'):
                        from .bottle_place import plan as plan_bottle_place
                        plan=plan_bottle_place(self.backend,draft,start,cap_metadata,tolerance,profile,progress)
                    else:
                        compiled = compile_arms(draft['arms'], start)
                        plan = self.backend.plan(compiled,start,pose_tolerance=tolerance,progress=progress,motion_profile=profile)
                    break
                except RobotStateError as exc:
                    retry = (exc.reason == 'teaching' and options['restore_can']) or (exc.reason == 'moved' and options['refresh_observation'])
                    if attempt or not retry or self.cancel_event.is_set():
                        raise
            plan.update(id=uuid.uuid4().hex, revision=rev, observation_id=draft['observation_id'],
                        created_at=time.time(), start=start, input_draft=draft,preparation=preparation,
                        auto_prepare=options, table_checks=self.backend.table_checks)
            if preparation['observation_refreshed']:
                plan['observation'] = self.frame.public()
                plan['notes'].insert(0,'已自动刷新深度观测，保留世界坐标路径并从当前位置规划接近段。')
            if preparation['restored_arms']:
                plan['notes'].insert(0,'已在稳定后自动恢复 CAN 控制：'+', '.join(preparation['restored_arms']))
            with self.lock:
                if self.cancel_event.is_set():
                    raise Cancelled()
                if rev != self.revision:
                    raise ValueError('Draft changed during preview')
                self.preview_data = plan
                self.execution = {'state': 'ready', 'progress': 0., 'elapsed_s': 0.,
                                  'preview_elapsed_s': time.monotonic()-begun}
            return public(plan)
        except Cancelled:
            with self.lock:
                self.execution = {'state': 'cancelled', 'kind': 'preview', 'progress': 0.,
                                  'elapsed_s': time.monotonic()-begun}
            raise ValueError('轨迹检查已取消；未执行任务轨迹') from None
        except Exception as exc:
            with self.lock:
                self.execution = {'state': 'error', 'kind': 'preview', 'progress': 0.,
                                  'elapsed_s': time.monotonic()-begun, 'error': str(exc)}
            raise

    def _progress_reporter(self, begun):
        def progress(**detail):
            with self.lock:
                if self.cancel_event.is_set():
                    raise Cancelled()
                self.execution.update(detail, elapsed_s=time.monotonic()-begun)
                self.execution['progress'] = detail.get('done', 0)/max(1, detail.get('total', 1))
        return progress

    @staticmethod
    def _home_arms(arms):
        if not isinstance(arms, (list, tuple)) or not arms or any(arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
            raise ValueError('arms 只接受 left、right 或两者')
        return [arm for arm in ARMS if arm in arms]

    def home(self, arms=ARMS, cancel_epoch=None, open_grippers=True):
        """Open the selected grippers and return those arms home, executing right after the plan passes its checks."""
        arms = self._home_arms(arms)
        begun = time.monotonic()
        with self.lock:
            self._editable()
            if cancel_epoch is not None and cancel_epoch != self.cancel_epoch:
                raise ValueError('回原位已取消')
            tolerance = copy.deepcopy(self.pose_tolerance)
            profile = self.motion_profile
            options = copy.deepcopy(self.auto_prepare)
            self.preview_data = None
            self.cancel_event.clear()
            self.execution = {'state': 'planning', 'kind': 'home', 'arms': arms, 'stage': 'prepare',
                              'progress': 0., 'elapsed_s': 0., 'started_at': time.time()}
        progress = self._progress_reporter(begun)
        try:
            preparation = {'restored_arms': []}
            for attempt in range(2):
                try:
                    progress(stage='prepare', arm=None, done=0, total=1)
                    recovered = self.backend.prepare(self.cancel_event, progress,
                                                     restore_can=options['restore_can'], restore_disabled=True)
                    preparation['restored_arms'] = sorted(set(preparation['restored_arms'] + recovered['restored_arms']))
                    start = self.backend.state()
                    plan = self.backend.plan_home(start, arms, pose_tolerance=tolerance, progress=progress,
                                                  motion_profile=profile, speed_m_s=self.home_speed_m_s,
                                                  open_grippers=open_grippers)
                    break
                except RobotStateError as exc:
                    retry = exc.reason == 'moved' or (exc.reason == 'teaching' and options['restore_can'])
                    if attempt or not retry or self.cancel_event.is_set():
                        raise
            plan.update(id=uuid.uuid4().hex, kind='home', revision=self.revision,
                        observation_id=self.frame.id if self.frame else None, created_at=time.time(), start=start,
                        input_draft={'home': {'arms': arms}}, preparation=preparation, auto_prepare=options)
            if preparation['restored_arms']:
                plan['notes'].insert(0, '已在稳定后自动恢复 CAN 控制：' + ', '.join(preparation['restored_arms']))
            with self.lock:
                if self.cancel_event.is_set():
                    raise Cancelled()
                self.execution = {'state': 'running', 'kind': 'home', 'arms': arms, 'progress': 0., 'elapsed_s': 0.,
                                  'id': plan['id']}
                self.worker = threading.Thread(target=self._run, args=(plan,), daemon=True, name='urai-execution')
                self.worker.start()
                return copy.deepcopy(self.execution)
        except Cancelled:
            with self.lock:
                self.execution = {'state': 'cancelled', 'kind': 'home', 'progress': 0.,
                                  'elapsed_s': time.monotonic() - begun}
            raise ValueError('回原位已取消；未发送运动') from None
        except Exception as exc:
            with self.lock:
                self.execution = {'state': 'error', 'kind': 'home', 'progress': 0.,
                                  'elapsed_s': time.monotonic() - begun, 'error': str(exc)}
            raise

    def execute(self, preview_id, cancel_epoch=None, task=None, caller=None):
        preview_id = self._renew_if_expired(preview_id, cancel_epoch)
        with self.lock:
            self._editable()
            if cancel_epoch is not None and cancel_epoch != self.cancel_epoch:
                raise ValueError('自动搬运已取消')
            p = self.preview_data
            if not p or preview_id != p['id'] or p['revision'] != self.revision or p['observation_id'] != self.frame.id:
                raise ValueError('Missing or stale preview; generate a new preview')
            if time.time()-p['created_at'] > PREVIEW_TTL_S:
                # _renew_if_expired should have replaced it already; this is the last line of defence.
                self.preview_data = None
                raise ValueError('Preview expired; generate a new preview')
            from .skills.bottle_cap import validate_stage
            validate_stage(self, p, self.backend.state())
            self.backend.validate_start(p['start'])
            self.preview_data = None
            p['task'] = copy.deepcopy(task)
            p['caller'] = copy.deepcopy(caller)
            self.cancel_event.clear()
            self.execution = {'state': 'running', 'progress': 0., 'elapsed_s': 0., 'id': p['id']}
            self.worker = threading.Thread(target=self._run, args=(p,), daemon=True, name='urai-execution')
            self.worker.start()
            return copy.deepcopy(self.execution)

    def _renew_if_expired(self, preview_id, cancel_epoch):
        """Regenerate a preview that timed out, instead of handing the operator an error to act on.

        A preview's life is a guard against executing a plan made for a scene that has moved on. The only
        answer to an expired one is "make another one", from the same draft and through the same checks (and,
        by default, a fresh observation), so execute() does exactly that before it moves. The regenerated plan
        passes every check a manual preview would; note that this means a stale execute request plans and then
        moves the robot in one call (documented in docs/api.md).

        Regenerating is done outside the service lock: planning takes seconds, and preview() takes the lock
        for itself where it needs it. A preview that is missing or superseded rather than expired is left
        alone - that one means the draft itself changed, which is the caller's to see.
        """
        with self.lock:
            current = self.preview_data
            if not current or current['id'] != preview_id:
                return preview_id
            if time.time()-current['created_at'] <= PREVIEW_TTL_S:
                return preview_id
            revision, epoch = self.revision, self.cancel_epoch
        if cancel_epoch is not None and cancel_epoch != epoch:
            return preview_id
        return self.preview(revision, epoch)['id']

    def _clear_executed_draft(self, plan):
        """Drop the trajectory that just ran, under the caller's lock.

        A path that has been flown is done: leaving it in the draft keeps the console drawing it over the next
        stroke and lets a second preview run it again by accident. Homing is not
        a user trajectory - it is usually the step before executing a prepared draft - and a queue item's draft
        belongs to the queue, which writes the next one itself and clears once the whole queue is done.
        """
        if plan.get('kind') == 'home' or plan.get('task') is not None or not self.draft:
            return
        self.draft = {}
        self.preview_data = None
        self.revision += 1
        self.execution['draft_cleared'] = True

    def _run(self, plan):
        cap_stage = None
        last_holder_check = [0.]
        def progress(fraction, elapsed):
            if cap_stage and cap_stage["stage"] == "twist" and time.monotonic()-last_holder_check[0] >= .1:
                from .skills.bottle_cap import held_session
                held_session(self, self.backend.state(), active_hold=bool(cap_stage.get('active_holder')))
                last_holder_check[0] = time.monotonic()
            with self.lock:
                self.execution.update(progress=float(fraction), elapsed_s=float(elapsed))
        result = None; final_execution = None
        try:
            from .skills.bottle_cap import validate_stage, complete_stage
            cap_stage = validate_stage(self, plan, self.backend.state())
            if plan.get('sequence'):
                from .bottle_place import execute as execute_bottle_place
                def phase_changed(label):
                    with self.lock:self.execution.update(stage=label)
                result=execute_bottle_place(self.backend,plan,self.cancel_event,progress,phase_changed)
            else:
                result = self.backend.execute(plan, plan['start'], self.cancel_event, progress)
            if cap_stage and not result.get('completed'):
                raise ValueError('拧瓶盖阶段未完成，保持当前位置')
            if self.cancel_event.is_set():
                raise Cancelled()
            if plan.get('kind') != 'home' and result.get('completed') and (not cap_stage or cap_stage['stage']=='place'):
                with self.lock:
                    self.execution.update(stage='return_home', progress=0.)
                    tolerance = copy.deepcopy(self.pose_tolerance)
                    profile = self.motion_profile
                def home_progress(**detail):
                    if self.cancel_event.is_set():
                        raise Cancelled()
                for attempt in range(2):
                    try:
                        self.backend.prepare(self.cancel_event, home_progress,
                                             restore_can=False, restore_disabled=False)
                        if self.cancel_event.is_set():
                            raise Cancelled()
                        start = self.backend.state()
                        home_plan = self.backend.plan_home(start, list(plan['arms']), pose_tolerance=tolerance,
                                                           motion_profile=profile, progress=home_progress,
                                                           open_grippers=False,
                                                           speed_m_s=float((plan.get('task') or {}).get('home_speed_m_s', self.home_speed_m_s)))
                        if self.cancel_event.is_set():
                            raise Cancelled()
                        home_plan.update(kind='home', start=start, id=uuid.uuid4().hex)
                        result['return_home_plan'] = public(home_plan)
                        returned = self.backend.execute(home_plan, start, self.cancel_event, progress)
                        break
                    except RobotStateError as exc:
                        if attempt or exc.reason != 'moved' or self.cancel_event.is_set():
                            raise
                result['return_home'] = {k: v for k, v in returned.items() if k != 'trace'}
                if self.cancel_event.is_set():
                    raise Cancelled()
                if not returned.get('completed'):
                    raise ValueError('技能已完成，但自动回位未完成')
            with self.lock:
                complete_stage(self,plan)
                if cap_stage:
                    result['bottle_cap'] = copy.deepcopy(self.bottle_cap)
                    result['holding_position'] = cap_stage['stage']!='place'
                self.execution.update(state='completed', progress=1., result={k: v for k, v in result.items() if k != 'trace'})
                self._clear_executed_draft(plan)
                final_execution = copy.deepcopy(self.execution)
        except Cancelled:
            with self.lock:
                self.bottle_cap = None
                self.execution.update(state='cancelled')
                final_execution = copy.deepcopy(self.execution)
        except Exception as exc:
            with self.lock:
                from .skills.bottle_cap import failed_stage
                failed_stage(self,plan,exc)
                self.execution.update(state='error', error=str(exc))
                final_execution = copy.deepcopy(self.execution)
        finally:
            if self.log_dir:
                try:
                    record = {'preview': public(plan), 'execution': final_execution, 'result': public(result),
                              'task': plan.get('task'), 'caller': plan.get('caller')}
                    (self.log_dir/(plan['id']+'.json')).write_text(json.dumps(record))
                except Exception as exc:
                    with self.lock:
                        if self.execution.get('id') == plan['id']:
                            self.execution['log_error'] = str(exc)

    def cancel(self):
        with self.lock:
            self.cancel_epoch += 1
            self.bottle_cap = None
            self.preview_data = None
            if self.execution['state'] in ('running', 'cancelling', 'planning'):
                self.execution['state'] = 'cancelling'; self.cancel_event.set()
            return copy.deepcopy(self.execution)
