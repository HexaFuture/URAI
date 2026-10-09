"""Execution half of the PiPER-X backend: robot readiness, the execution lock and the local 50 Hz loop.

Hardware is reached only through :class:`urai.robot.runtime.RobotRuntime` (drivers, cameras, motion gate),
so the same class runs against the real arms and against the kinematic simulator of :mod:`urai.robot.simulated`.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
import numpy as np
from scipy.spatial.transform import Rotation
from ..geometry import Frame
from ..settings import DEFAULT_TRACKING_LIMIT_DEG
from .common import ARMS, Cancelled, RobotStateError
from .planner import PiperPlanner
from .retiming import measured_progress, tracking_gap


def scheduler_stall(gap_s, t, limit_s):
    """Error text when two control iterations are ``gap_s`` apart, more than ``limit_s``; None otherwise.

    ``t`` is the reference clock of the last iteration (0 before the first sample).
    """
    if gap_s > limit_s:
        return f'Trajectory scheduler stalled for {gap_s:.3f}s at t={t:.2f}s (limit {limit_s:g}s)'
    return None


class PiperBackend(PiperPlanner):
    """Plans against the calibrated arm models and executes on the robot runtime's two PiPER-X arms."""
    mode = 'real'
    #: Largest allowed gap (degrees) between the streamed joint reference and the measured joints during
    #: execution; the arm is held when it is exceeded. 0 turns the guard off, and then nothing stops a blocked
    #: arm: position mode keeps pulling until the motion is cancelled or the emergency stop is pressed.
    tracking_limit_deg = DEFAULT_TRACKING_LIMIT_DEG
    #: The PiPER firmware trails a fast joint reference instead of failing it (tens of degrees behind at
    #: 150 deg/s, the lag growing with rate). A joint that is late but on its path is following; one that has
    #: stopped is not. The guard therefore compares against the nearest reference point up to this long behind
    #: the clock: at rest that is the plain reference.
    tracking_lag_s = .5
    #: Longest gap tolerated between two 20 ms control iterations before the arm is held.
    scheduler_gap_s = .25

    def __init__(self, runtime, workers=0):
        super().__init__(runtime.models, workers)
        self.runtime = runtime
        self.execution_lock = runtime.execution_lock
        # Observation captures and the live preview share this lock; nothing else touches the cameras.
        self.capture_lock = threading.Lock()

    def info(self):
        return {'mode': self.mode, 'name': self.runtime.name,
                'motion_allowed': bool(self.runtime.motion_allowed), 'force_control': False,
                'checks': ['IK + orientation residual', 'joint limits', *(['table clearance'] if self.table_checks else []),
                           'self collision', 'inter-arm collision'], 'table_checks': self.table_checks,
                'control': 'ordinary joint position mode, 50 Hz reference',
                'scene_collision': 'Point cloud is visual context; arbitrary object collisions are not validated'}

    def state(self):
        out = {}
        for a in ARMS:
            d = self.runtime.arms[a]; st = d.state(); q = np.asarray(st.joints_deg, dtype=float)
            pose = self.models[a].fk_tcp_world(q)
            out[a] = {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
                      'joints_deg': q.tolist(), 'gripper_mm': float(st.gripper_mm), 'enabled': bool(st.enabled),
                      'feedback_age_s': float(d.feedback_age_s()), 'ctrl_mode': str(st.ctrl_mode)}
        return out

    def capture(self, camera='head'):
        """One RGB-D frame of ``camera`` ('head', 'hand_left' or 'hand_right') in world coordinates."""
        with self.capture_lock:
            f = self.runtime.capture(camera)
        return Frame(f.rgb, f.depth_m, f.k, f.t_world_cam)

    @contextmanager
    def _exclusive(self):
        if not self.execution_lock.acquire(blocking=False):
            raise ValueError('The robot is busy with another operation; retry with a fresh observation')
        try:
            yield
        finally:
            self.execution_lock.release()

    def _fresh(self, allow_teaching=False, allow_disabled=False):
        for a in ARMS:
            d = self.runtime.arms[a]; s = d.state()
            name = '左臂' if a == 'left' else '右臂'
            if not s.enabled and not allow_disabled:
                raise RobotStateError(f'{name}未使能，请恢复机械臂使能后重新规划。')
            # The low-speed frames tell a disabled arm from a silent bus; the joint frames are what the
            # planner and the control loop read, so both have to be fresh.
            age = max(float(d.feedback_age_s()), float(d.joint_feedback_age_s()))
            if not np.isfinite(age) or age < 0:
                raise RobotStateError(f'{name}反馈时间无效，请检查 CAN 通信后重新规划。')
            if age > .5:
                raise RobotStateError(f'{name} CAN 反馈已过期（{age*1000:.0f} ms，上限 500 ms），请检查通信后重新规划。')
            mode = str(s.ctrl_mode)
            if 'CAN' not in mode and not (allow_teaching and 'TEACHING' in mode):
                if 'TEACHING' in mode:
                    raise RobotStateError(f'{name}处于拖动示教模式（{mode}），请先松开机械臂、退出示教并恢复 CAN 控制后重新规划。', reason='teaching')
                raise RobotStateError(f'{name}当前控制模式为 {mode}，请恢复 CAN 控制后重新规划。')
            if int(s.arm_status) != 0 or int(s.err_code) != 0:
                raise RobotStateError(f'{name}控制器报错（arm_status={int(s.arm_status)}, err_code={int(s.err_code)}），请检查控制器状态。')

    def prepare(self, cancel, progress, restore_can=True, restore_disabled=False):
        def check(allow_disabled=False):
            if cancel.is_set():
                raise Cancelled()
            self._fresh(allow_teaching=restore_can, allow_disabled=allow_disabled)
        def wait(dt, allow_disabled=False):
            if cancel.wait(dt):
                raise Cancelled()
            check(allow_disabled=allow_disabled)
        restored = []; touched = []
        with self._exclusive():
            # Ordinary preview/execute keeps the old strict behaviour. Only an
            # explicitly requested home may opt into re-enabling an arm that a
            # previous protection stop disabled.
            check(allow_disabled=restore_disabled)
            states = self.state()
            disabled = [a for a in ARMS if not states[a]['enabled']]
            teaching = [a for a in ARMS if 'TEACHING' in states[a]['ctrl_mode']]
            if not teaching and not disabled:
                return {'restored_arms': []}
            begun = stable_since = time.monotonic()
            anchor = {a: np.asarray(states[a]['joints_deg']).copy() for a in ARMS}
            while time.monotonic()-stable_since < 1.:
                check(allow_disabled=restore_disabled)
                now = time.monotonic()
                current = {a: np.asarray(self.runtime.joints_deg(a)) for a in ARMS}
                if any(not np.isfinite(q).all() for q in current.values()):
                    raise RobotStateError('关节反馈无效，不能自动恢复控制。')
                if any(np.max(np.abs(current[a]-anchor[a])) > .25 for a in ARMS):
                    stable_since = now; anchor = {a:q.copy() for a,q in current.items()}
                progress(stage='settle',arm=None,done=0,total=1)
                if now-begun > 12:
                    raise RobotStateError('机械臂仍在移动，请松开后再次发起操作。')
                wait(.05, allow_disabled=restore_disabled)
            self.runtime.require_motion('automatic CAN restoration before planning')
            try:
                for a in disabled:
                    check(allow_disabled=True)
                    driver = self.runtime.arms[a]
                    if driver.state().enabled:
                        continue
                    current = {k:np.asarray(self.runtime.joints_deg(k)).copy() for k in ARMS}
                    if any(np.max(np.abs(current[k]-anchor[k])) > .25 for k in ARMS):
                        raise RobotStateError('机械臂重新开始移动，自动使能已停止。')
                    q = current[a]
                    self._geometry_check(current['left'], current['right'], feedback=True)
                    progress(stage='enable',arm=a,done=0,total=1)
                    touched.append(a)
                    # The runtime clears only disabled motor channels and brackets the
                    # enable with the measured joint target, so an old controller target
                    # cannot cause a jump.
                    with cancel.dispatch():
                        self.runtime.clear_motor_faults(a)
                    if cancel.is_set():
                        raise Cancelled()
                    if not driver.state().enabled:
                        raise RobotStateError(f'{a} 自动使能失败，请检查驱动器状态。')
                    if np.max(np.abs(np.asarray(self.runtime.joints_deg(a))-q)) > .5:
                        raise RobotStateError(f'{a} 自动使能后位置变化过大，已停止复位。')
                    restored.append(a)
                check()
                for a in teaching:
                    check()
                    driver = self.runtime.arms[a]
                    if 'CAN' in str(driver.state().ctrl_mode):
                        continue
                    current = {k:np.asarray(self.runtime.joints_deg(k)).copy() for k in ARMS}
                    if any(np.max(np.abs(current[k]-anchor[k])) > .25 for k in ARMS):
                        raise RobotStateError('机械臂重新开始移动，自动恢复已停止。')
                    q = current[a]
                    self._geometry_check(current['left'], current['right'], feedback=True)
                    progress(stage='restore',arm=a,done=0,total=1)
                    touched.append(a)
                    with cancel.dispatch():
                        driver.exit_teaching()
                    wait(.3)
                    for _ in range(8):
                        check()
                        if np.max(np.abs(np.asarray(self.runtime.joints_deg(a))-q)) > .5:
                            raise RobotStateError(f'{a} 接管期间位置变化，已停止自动恢复。')
                        # Ordinary position protocol, immediately replacing any old
                        # controller target with the measured hold.
                        with cancel.dispatch():
                            driver.set_joint_mode(5)
                            driver.command_joints(q)
                        wait(.2)
                        if 'CAN' in str(driver.state().ctrl_mode):
                            break
                    else:
                        raise RobotStateError(f'{a} 未能退出示教模式，请检查示教开关。')
                    if np.max(np.abs(np.asarray(self.runtime.joints_deg(a))-q)) > .5:
                        raise RobotStateError(f'{a} 恢复控制后位置变化过大，已保持。')
                    restored.append(a)
                check(); self._fresh()
                return {'restored_arms': restored}
            except BaseException as exc:
                # Never force an arm still in teaching mode into CAN on cancellation.
                controlled = [a for a in touched if 'CAN' in str(self.runtime.arms[a].state().ctrl_mode)]
                if controlled:
                    hold_errors = self._hold(controlled)
                    if hold_errors:
                        raise RuntimeError(f'{exc}; position hold failed: {hold_errors}') from exc
                raise

    def validate_start(self, start):
        self._fresh()
        for a in ARMS:
            if np.max(np.abs(self.runtime.joints_deg(a)-start[a]['joints_deg'])) > .5:
                raise RobotStateError(f'{a} moved since preview; generate a new preview', reason='moved')
            if abs(self.runtime.gripper_mm(a)-start[a]['gripper_mm']) > 2:
                raise RobotStateError(f'{a} gripper moved since preview; generate a new preview', reason='moved')

    def _hold(self, arms):
        errors = []
        for a in arms:
            d = self.runtime.arms[a]
            try:
                if max(d.feedback_age_s(), d.joint_feedback_age_s()) > .5:
                    raise RuntimeError('Cannot hold using stale feedback')
                q = d.joints()
                d.set_joint_mode(5)
                d.command_joints(q)
            except Exception as exc:
                errors.append(f'{a}: {exc}')
        return errors

    def execute(self, plan, start, cancel, progress):
        with self._exclusive():
            return self._execute(plan, start, cancel, progress)

    def _read_control_feedback(self, active, cancel, retry_budget):
        """Discard a late sample, hold, and retry once; never dispatch from stale geometry."""
        began = time.monotonic()
        for attempt in range(2):
            if cancel.is_set():
                raise Cancelled()
            now = time.monotonic()
            cpu = time.thread_time()
            self._fresh()
            actual = {a: self.runtime.joints_deg(a) for a in ARMS}
            read_at = time.monotonic()
            self._geometry_check(actual['left'], actual['right'], feedback=True)
            checked_at = time.monotonic()
            if checked_at-now <= .08:
                return actual, now, read_at, checked_at, cpu, now-began, attempt
            detail = ('Feedback validation exceeded the control deadline: '
                      f'total={(checked_at-now)*1000:.1f} ms, read={(read_at-now)*1000:.1f} ms, '
                      f'geometry={(checked_at-read_at)*1000:.1f} ms, '
                      f'CPU={(time.thread_time()-cpu)*1000:.1f} ms (limit 80 ms)')
            if attempt or retry_budget <= 0 or checked_at-now > .25:
                raise RuntimeError(detail)
            with cancel.dispatch():
                errors = self._hold(active)
            if errors:
                raise RuntimeError(f'{detail}; position hold failed: {errors}')
        raise AssertionError('unreachable')

    def _execute(self, plan, start, cancel, progress):
        if cancel.is_set():
            raise Cancelled()
        self.runtime.require_motion('previewed trajectory')
        self.validate_start(start)
        active = list(plan['arms']); fired = set(); trace = []
        approach_end = plan.get('approach_duration_s', 0.)
        entered_task = approach_end <= 0
        approach_wait = 0.; barrier_started = None; settled_since = None
        gates=[];allowed_events=set();gate_index=0;gate_started=None;gate_settled=None
        for a,p in plan['arms'].items():
            for n,e in enumerate(p['gripper_events']):
                if e.get('wait_for_arrival'):
                    gates.append((e['time_s'],a,n,'arrival'))
                if e.get('verify'):
                    gates.append((e['hold_end_time_s'],a,n,e['verify']))
        gates.sort()
        # Events flagged on_measured fire when the measured joints have progressed past the event
        # time along the reference (minus lead_s), so a lagging swing still releases at its release pose.
        tracked = {(a, n) for a, p in plan['arms'].items() for n, e in enumerate(p['gripper_events']) if e.get('on_measured')}
        measured_t = {a: 0. for a in active}; end_started = None
        ramps = {}   # arm -> (started, from_mm, to_mm, ramp_s, effort): jaws commanded gradually over ramp_s
        started = None
        t = 0.
        feedback_retries = 0
        feedback_pause_s = 0.
        try:
            for a in active:
                with cancel.dispatch():
                    self.runtime.arms[a].set_joint_mode(plan.get('motion_limits',{}).get('controller_speed_percent',10))
            started = time.monotonic(); previous = started; deadline = started
            while True:
                if cancel.is_set():
                    raise Cancelled()
                delay = deadline-time.monotonic()
                if delay > 0 and cancel.wait(delay):
                    raise Cancelled()
                now = time.monotonic()
                validation_cpu_start = time.thread_time()
                stall = scheduler_stall(now-previous, t, self.scheduler_gap_s)
                if stall:
                    raise RuntimeError(stall)
                actual, now, feedback_read_at, geometry_checked_at, validation_cpu_start, paused, retried = self._read_control_feedback(
                    active, cancel, 0 if tracked and entered_task else 3-feedback_retries)
                feedback_retries += retried
                feedback_pause_s += paused
                approach_wait += paused
                if paused:
                    if barrier_started is not None: barrier_started += paused
                    if settled_since is not None: settled_since += paused
                    if gate_started is not None: gate_started += paused
                    if gate_settled is not None: gate_settled += paused
                    if end_started is not None: end_started += paused
                    ramps = {a:(v[0]+paused, *v[1:]) for a,v in ramps.items()}
                t = min(now-started-approach_wait, plan['duration_s'])
                if not entered_task and t >= approach_end:
                    # Freeze the common task clock while both arms hold their start poses.
                    # This also gates s=0 gripper events on measured arrival, not schedule time.
                    t = approach_end
                    if barrier_started is None:
                        barrier_started = now
                    residual = max(np.max(np.abs(actual[a]-plan['arms'][a]['curve'](approach_end))) for a in active)
                    if residual < .7:
                        if settled_since is None:
                            settled_since = now
                        if now-settled_since >= .06:
                            entered_task = True
                            approach_wait = now-started-approach_end
                    else:
                        settled_since = None
                    if not entered_task and now-barrier_started > 3:
                        raise RuntimeError(f'Approach start pose not reached: {residual:.2f} degrees')
                if entered_task and gate_index<len(gates) and t>=gates[gate_index][0]:
                    moment,gate_arm,number,kind=gates[gate_index]
                    t=moment
                    if gate_started is None:gate_started=now
                    residual=max(np.max(np.abs(actual[a]-plan['arms'][a]['curve'](min(t,plan['arms'][a]['time_s'][-1])))) for a in active)
                    opening=float(self.runtime.gripper_mm(gate_arm))
                    valid=residual<.7 and (kind=='arrival' or
                        (kind=='holding' and np.isfinite(opening) and 1.5<opening<67))
                    if valid:
                        if gate_settled is None:gate_settled=now
                        if now-gate_settled>=.06:
                            if kind=='arrival':allowed_events.add((gate_arm,number))
                            gate_index+=1;gate_started=None;gate_settled=None
                            approach_wait=now-started-t
                    else:gate_settled=None
                    if gate_started is not None and now-gate_started>3:
                        if kind=='arrival':raise RuntimeError(f'{gate_arm}：抓取／放置位置未到达（关节误差 {residual:.2f}°）')
                        raise RuntimeError(f'{gate_arm}：夹爪疑似空抓或未闭合（反馈 {opening:.1f} mm），已停止后续搬运')
                target = {}
                for a in ARMS:
                    if a not in active:
                        if np.max(np.abs(actual[a]-start[a]['joints_deg'])) > .5:
                            raise RuntimeError(f'Inactive {a} arm moved')
                        continue
                    p = plan['arms'][a]
                    clock_t = min(t, p['time_s'][-1])
                    target[a] = p['curve'](clock_t)
                    if self.tracking_limit_deg and np.abs(actual[a]-target[a]).max() > self.tracking_limit_deg:
                        # Late on the path is the firmware's lag; off the path or stopped is a fault.
                        gap, worst = tracking_gap(p['curve'], actual[a], clock_t, self.tracking_lag_s)
                        if gap > self.tracking_limit_deg:
                            raise RuntimeError(f'{a}: joint tracking error {gap:.1f} deg on j{worst+1} exceeded '
                                               f'{self.tracking_limit_deg:g} deg at t={t:.2f}s (against the reference up '
                                               f'to {self.tracking_lag_s:g} s behind)')
                for a in active:
                    if any(key in tracked and key not in fired for key in ((a, n) for n in range(len(plan['arms'][a]['gripper_events'])))):
                        measured_t[a] = measured_progress(plan['arms'][a]['curve'], actual[a], measured_t[a], t)
                validation_end = time.monotonic()
                if validation_end-now > .08:
                    raise RuntimeError('Feedback validation exceeded the control deadline: '
                                       f'total={(validation_end-now)*1000:.1f} ms, '
                                       f'read={(feedback_read_at-now)*1000:.1f} ms, '
                                       f'geometry={(geometry_checked_at-feedback_read_at)*1000:.1f} ms, '
                                       f'reference={(validation_end-geometry_checked_at)*1000:.1f} ms, '
                                       f'CPU={(time.thread_time()-validation_cpu_start)*1000:.1f} ms (limit 80 ms)')
                if retried:
                    for a in active:
                        with cancel.dispatch():
                            self.runtime.arms[a].set_joint_mode(plan.get('motion_limits', {}).get('controller_speed_percent', 10))
                for a, q in target.items():
                    with cancel.dispatch():
                        self.runtime.arms[a].command_joints(q)
                    if a in ramps:
                        began, from_mm, to_mm, ramp_s, effort = ramps[a]
                        fraction = min(1., (now-began)/ramp_s)
                        with cancel.dispatch():
                            self.runtime.arms[a].set_gripper(from_mm+(to_mm-from_mm)*fraction, effort=effort)
                            self.runtime.last_gripper_command[a] = from_mm+(to_mm-from_mm)*fraction
                        if fraction >= 1.:
                            del ramps[a]
                    for n, e in enumerate(plan['arms'][a]['gripper_events']):
                        clock = measured_t[a] if e.get('on_measured') else t
                        due = entered_task and clock >= e['time_s']-e.get('lead_s', 0.) and (a, n) not in fired
                        if not due or (e.get('wait_for_arrival') and (a, n) not in allowed_events):
                            continue
                        if 'opening_mm' not in e:
                            # A pure pause only waits at its pose; the gripper keeps its last command.
                            fired.add((a, n))
                            continue
                        measured_mm = float(self.runtime.gripper_mm(a))
                        if e.get('effort') is not None:
                            effort = int(e['effort'])          # the plan's own torque limit, e.g. a gentle cup grasp
                        else:
                            effort = self.runtime.open_effort if e['opening_mm'] > measured_mm else self.runtime.grip_effort
                        if e.get('ramp_s', 0.) > 0:
                            # Start from the measured opening and let the cycles above walk the jaws to the target.
                            start_mm = measured_mm if np.isfinite(measured_mm) else float(self.runtime.last_gripper_command.get(a, e['opening_mm']))
                            ramps[a] = (now, start_mm, float(e['opening_mm']), float(e['ramp_s']), effort)
                            fired.add((a, n))
                            continue
                        with cancel.dispatch():
                            self.runtime.arms[a].set_gripper(e['opening_mm'], effort=effort)
                            self.runtime.last_gripper_command[a] = e['opening_mm']
                            fired.add((a, n))
                trace.append({'t': round(t, 4), 'wall_t': round(now-started, 4),
                              'phase': 'user' if entered_task else 'approach',
                              'joints_deg': {a: q.tolist() for a, q in actual.items()},
                              **({'progress_s': {a: round(v, 4) for a, v in measured_t.items()}} if tracked else {})})
                progress(t/plan['duration_s'], t)
                if t >= plan['duration_s'] and gate_index >= len(gates):
                    if not tracked-fired:
                        break
                    # The reference is complete but a measured-progress release is still pending:
                    # keep streaming the end pose until the arm gets there, then let it fire.
                    if end_started is None:
                        end_started = now
                    residual = max(np.max(np.abs(actual[a]-plan['arms'][a]['joints_deg'][-1])) for a in active)
                    if residual < .7:
                        measured_t = {a: float(plan['arms'][a]['time_s'][-1]) for a in active}
                    elif now-end_started > 3:
                        raise RuntimeError(f'Endpoint not reached: {residual:.2f} degrees')
                previous = now; deadline = max(deadline+.02, time.monotonic())
            # Distinguish command completion from measured arrival.
            end = time.monotonic()+3
            while True:
                if cancel.is_set():
                    raise Cancelled()
                self._fresh()
                residual = max(np.max(np.abs(self.runtime.joints_deg(a)-plan['arms'][a]['joints_deg'][-1])) for a in active)
                if residual < .7:
                    break
                if cancel.wait(.02):
                    raise Cancelled()
                if time.monotonic() > end:
                    raise RuntimeError(f'Endpoint not reached: {residual:.2f} degrees')
            return {'completed': True, 'endpoint_error_deg': float(residual), 'samples': len(trace),
                    'elapsed_s': time.monotonic()-started, 'approach_wait_s': approach_wait, 'trace': trace,
                    'feedback_retries': feedback_retries, 'feedback_pause_s': feedback_pause_s}
        except BaseException as exc:
            hold_errors = self._hold(active)
            if hold_errors:
                raise RuntimeError(f'{exc}; position hold failed: {hold_errors}') from exc
            raise
