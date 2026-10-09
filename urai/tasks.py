"""Sequential server-side queue of top-down transfer tasks.

Each item observes, relocates its object on the fresh frame, generates a direct
transfer draft, plans it through ``Service.preview`` and runs it through
``Service.execute``; the next item starts only after the previous one reached a
terminal state. Any failure stops the queue and marks the remaining items skipped.
"""
from __future__ import annotations

import copy
import threading
import time
import numpy as np
from .backend import ARMS
from .skills.registry import choice, number
from .transfer import leveled_cloud, locate_object, transfer_between
from .toss import (DEFAULT_LEAD_S, DEFAULT_TOSS_SPEED_M_S, MAX_LEAD_S, MAX_TOSS_SPEED_M_S, MIN_TOSS_SPEED_M_S,
                   STYLE_LABELS, SWING_RADII_M, TOSS_STYLES, toss_between)


MAX_ITEMS = 32
MAX_LABEL_CHARS = 64
ITEM_FIELDS = {'object_pixel', 'object_xy', 'landing_xyz', 'arm', 'label', 'placement', 'grasp_pixels'}
PLACEMENTS = ('place', 'drop', 'toss')
TERMINAL_STATES = ('completed', 'failed', 'skipped', 'cancelled')
#: Queue-wide options of ``POST /api/tasks`` as (default, minimum, maximum). ``TaskQueue.submit`` validates
#: against these and :func:`describe_toss` publishes them, so the catalogue cannot drift from the validation.
OPTION_LIMITS = {'approach_speed_m_s': (.12, .001, .20), 'speed_m_s': (.10, .001, .15), 'clearance_m': (.12, .02, .30),
                 'release_tilt_deg': (60., 0., 80.),
                 'toss_speed_m_s': (DEFAULT_TOSS_SPEED_M_S, MIN_TOSS_SPEED_M_S, MAX_TOSS_SPEED_M_S),
                 'toss_lead_s': (DEFAULT_LEAD_S, 0., MAX_LEAD_S),
                 'carry_speed_m_s': (.6, .01, 1.), 'windup_speed_m_s': (.15, .01, .3),
                 'home_speed_m_s': (.6, .05, 1.), 'grasp_hold_s': (.5, .5, 2.)}


def arm_bases(backend):
    """Horizontal base positions used to hand each object to the nearer arm."""
    if backend.mode == 'real':
        return {arm: np.asarray(backend.models[arm].base_xy, dtype=float)[:2] for arm in ARMS}
    state = backend.state()
    return {arm: np.asarray(state[arm]['xyz'][:2], dtype=float) for arm in ARMS}


ARM_BODY_MARGIN_M = .03   # a detected "object" this close to an arm's own link capsules is the arm itself


def arm_body_reason(models, current, xyz):
    """Why a detected object is part of a robot arm, or None.

    Whole-frame detection sees the arms' own links, fingers and base plates as raised regions. Each real
    arm model carries capsule geometry for the current joints (world frame, millimetres); a point within
    ARM_BODY_MARGIN_M of one of those capsules is the arm, whatever the colour or height rules made of it.
    """
    point = np.asarray(xyz, dtype=float)
    for arm, model in models.items():
        capsules = getattr(model, 'body_capsules', None)
        if capsules is None:
            continue
        for name, start, end, radius_mm in capsules(np.asarray(current[arm]['joints_deg'], dtype=float)):
            start, end = np.asarray(start, dtype=float)/1000., np.asarray(end, dtype=float)/1000.
            axis = end-start
            along = float(np.clip(((point-start) @ axis)/max(float(axis @ axis), 1e-12), 0., 1.))
            if np.linalg.norm(point-(start+along*axis))-radius_mm/1000. < ARM_BODY_MARGIN_M:
                return f"是{'左臂' if arm == 'left' else '右臂'}本体（{name}），不是桌面物体"
    return None


def nearest_arm(bases, xy):
    xy = np.asarray(xy, dtype=float)[:2]
    return min(ARMS, key=lambda arm: float(np.linalg.norm(bases[arm]-xy)))


def finite_vector(value, size, message):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(message)
    vector = np.asarray(value, dtype=float) if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value) else None
    if vector is None or not np.isfinite(vector).all():
        raise ValueError(message)
    return vector


def release_speed(arm_plan):
    """Planned fingertip speed (m/s) at the last opening command, from the public plan of one arm."""
    events = [e for e in arm_plan.get('gripper_events', []) if e.get('opening_mm') == 70]
    if not events:
        return None
    times = np.asarray(arm_plan['time_s'], dtype=float)
    xyz = np.asarray(arm_plan['xyz'], dtype=float)
    i = int(np.clip(np.searchsorted(times, events[-1]['time_s']), 1, len(times)-1))
    step = times[i]-times[i-1]
    return float(np.linalg.norm(xyz[i]-xyz[i-1])/step) if step > 0 else 0.


def describe_toss():
    """Catalogue entry for tossing, shaped like ``Skill.describe()`` so ``GET /api/skills`` can list it.

    Tossing is not planned through ``POST /api/skills/<name>``: the task queue observes, relocates, plans and
    executes each item on the server, so it is submitted with ``POST /api/tasks`` (``call``). ``inputs`` are the
    queue-wide options with the ranges and defaults :meth:`TaskQueue.submit` enforces (``OPTION_LIMITS``);
    ``item_fields`` are the fields of one entry of ``items`` as :meth:`TaskQueue._build_item` validates them.
    """
    def option(key, label, unit, step, hint=''):
        default, minimum, maximum = OPTION_LIMITS[key]
        return number(label, minimum, maximum, default, step=step, unit=unit, hint=hint)

    radii = '/'.join(f'{radius:.2f}' for radius in SWING_RADII_M)
    return {
        'name': 'toss', 'label': '抛掷', 'group': '抓取搬运',
        'summary': '任务队列逐项执行：重新观测 → 重新定位物体 → 抓取 → 抛向落点；同一队列也可放到落点（place）或在落点上方释放（drop）',
        'stroke': 'point',
        'stroke_hint': '不经画笔调用：每项用 object_pixel 或 object_xy 指定物体（可另给 grasp_pixels 画线抓取），'
                       '用 landing_xyz 或已设置的投放点指定落点',
        'arms': 'single', 'call': 'POST /api/tasks',
        'inputs': {
            'speed_m_s': option('speed_m_s', '移动速度', ' m/s', .005, '抓取、搬运与放置段的目标速度。'),
            'approach_speed_m_s': option('approach_speed_m_s', '接近速度', ' m/s', .005),
            'clearance_m': option('clearance_m', '抬起余量', ' m', .01),
            'toss_style': choice('抛法', [(style, STYLE_LABELS[style]) for style in TOSS_STYLES], TOSS_STYLES[0],
                                 hint='对整个队列生效；overhand 需要机械臂运动学模型，仿真后端只能 sidearm。'),
            'toss_speed_m_s': option('toss_speed_m_s', '抛掷速度', ' m/s', .1,
                                     'sidearm 为放手时的指尖速度；overhand 只能在设计关节速率内放慢，不能加快。'),
            'toss_lead_s': option('toss_lead_s', '放手提前量', ' s', .01, '张开夹爪的指令比实测到达放手位姿提前的时间。'),
            'carry_speed_m_s': option('carry_speed_m_s', '抓起后搬到蓄力位的速度', ' m/s', .05),
            'windup_speed_m_s': option('windup_speed_m_s', '进入蓄力位的速度', ' m/s', .01),
            'grasp_hold_s': option('grasp_hold_s', '闭爪停留', ' s', .1),
            'home_speed_m_s': option('home_speed_m_s', '每项结束后回收纳位的速度', ' m/s', .05),
            'release_tilt_deg': option('release_tilt_deg', '释放后仰角', '°', 5.,
                                       'place / drop 项释放时手腕向基座后仰的角度；0 表示不额外后仰。'),
        },
        'item_fields': {
            'object_pixel': {'type': 'pixel', 'label': '物体像素 [u, v]', 'required': False,
                             'hint': '与 object_xy 二选一（必给其一），须落在当前观测画面内。'},
            'object_xy': {'type': 'world_xy', 'label': '物体世界坐标 [x, y]', 'unit': ' m', 'required': False,
                          'hint': '与 object_pixel 二选一（必给其一），按该臂的桌面高度投影到画面。'},
            'grasp_pixels': {'type': 'pixel_pair', 'label': '抓取线 [[u, v], [u, v]]', 'required': False,
                             'hint': '可选：两指位置，按画线抓取、不做物体分割；只能用于 placement 为 toss 的项。'},
            'landing_xyz': {'type': 'world_xyz', 'label': '落点 [x, y, z]', 'unit': ' m', 'required': False,
                            'hint': '省略时使用 PUT /api/drop-point 设置的投放点。'},
            'placement': {**choice('投放方式', [('place', '放到落点'), ('drop', '落点上方释放'), ('toss', '抛向落点')], None,
                                   hint='省略时：落点来自投放点则为 drop，否则为 place；抛掷必须写 toss。'),
                          'required': False},
            'arm': {**choice('机械臂', [('left', '左臂'), ('right', '右臂')], None,
                             hint='省略时用基座离物体最近的一只臂。'), 'required': False},
            'label': {'type': 'string', 'label': '名称', 'min_length': 1, 'max_length': MAX_LABEL_CHARS,
                      'required': False, 'hint': '省略时为“物体 <序号>”。'},
        },
        'max_items': MAX_ITEMS,
        'hint': f'请求体：{{"observation_id": 当前观测 id, "items": [1–{MAX_ITEMS} 项], 以及 inputs 里的队列参数}}；'
                '队列执行中不能再提交，DELETE /api/tasks 停止队列。',
        'limits': '每项执行前都会重新观测并在新画面上重新定位物体；任一项失败或被取消，队列即停止，其余项标为 skipped。'
                  f'sidearm 依次尝试摆动半径 {radii} m，直到预览通过；overhand 够不到落点时照常抛出并报告落差。',
        'references': [],
    }


class TaskQueue:
    def __init__(self, service):
        self.service = service
        self.items = []
        self.thread = None
        self.active = False
        self.stop_requested = False
        self.epoch = None
        self.options = {}
        self.finished_at = None

    @property
    def lock(self):
        return self.service.lock

    def blocks_current_thread(self):
        """Editing is reserved for the runner thread while the queue is active."""
        return self.active and threading.current_thread() is not self.thread

    def status(self):
        with self.lock:
            return {'active': self.active, 'items': copy.deepcopy(self.items)}

    def summary(self):
        with self.lock:
            current = next((item for item in self.items if item['status'] not in TERMINAL_STATES), None)
            return {'active': self.active, 'total': len(self.items),
                    'completed': sum(item['status'] == 'completed' for item in self.items),
                    'current_index': current['index'] if current else None,
                    'current_status': current['status'] if current else None,
                    'finished_at': self.finished_at}

    def submit(self, data):
        service = self.service
        with self.lock:
            service._editable()
            if self.active:
                raise ValueError('任务队列正在执行，请等待完成或先停止队列')
            if not isinstance(data, dict):
                raise ValueError('请提供 observation_id 和 items')
            if not service.frame or data.get('observation_id') != service.frame.id:
                raise ValueError('Observation is stale; refresh and rebuild the draft')
            raw_items = data.get('items')
            if not isinstance(raw_items, list) or not 1 <= len(raw_items) <= MAX_ITEMS:
                raise ValueError(f'请提交 1–{MAX_ITEMS} 个任务项')
            options = {key: float(data.get(key, OPTION_LIMITS[key][0]))
                       for key in ('approach_speed_m_s', 'speed_m_s', 'clearance_m', 'release_tilt_deg',
                                   'toss_speed_m_s', 'toss_lead_s')}

            def within(key):
                return OPTION_LIMITS[key][1] <= options[key] <= OPTION_LIMITS[key][2]

            if not np.isfinite([options['toss_speed_m_s'], options['toss_lead_s']]).all() \
                    or not within('toss_speed_m_s') or not within('toss_lead_s'):
                raise ValueError('toss_speed_m_s 需为 0.3–5.0 m/s，toss_lead_s 需为 0–0.5 s')
            if not np.isfinite(options['approach_speed_m_s']) or not within('approach_speed_m_s'):
                raise ValueError('接近速度需为 0.001–0.20 m/s')
            for key in ('carry_speed_m_s', 'windup_speed_m_s', 'home_speed_m_s', 'grasp_hold_s'):
                default, minimum, maximum = OPTION_LIMITS[key]
                value = float(data.get(key, default))
                if not np.isfinite(value) or not minimum <= value <= maximum:
                    raise ValueError(f'{key} 需为 {minimum}–{maximum}')
                options[key] = value
            toss_style = data.get('toss_style', TOSS_STYLES[0])
            if toss_style not in TOSS_STYLES:
                raise ValueError('toss_style 只接受 overhand（伸臂投掷）或 sidearm（侧摆）')
            if not np.isfinite(options['release_tilt_deg']) or not within('release_tilt_deg'):
                raise ValueError('release_tilt_deg 需在 0–80 度之间')
            if not np.isfinite(list(options.values())).all() or not within('speed_m_s') or not within('clearance_m'):
                raise ValueError('搬运高度需为 0.02–0.30 m，速度需为 0.001–0.15 m/s')
            options['toss_style'] = toss_style
            current = service.backend.state()
            if service._observation_stale(current) and not service.auto_prepare['refresh_observation']:
                raise ValueError('机器人已移动，请更新图像后再提交任务')
            frame = service.frame
            # One leveled cloud for every item; the calibrated table anchors all heights.
            cloud = None
            if any('grasp_pixels' not in item for item in raw_items if isinstance(item, dict)):
                cloud, _ = leveled_cloud(frame, service.table_z())
            bases = arm_bases(service.backend)
            items = [self._build_item(index, raw, frame, cloud, current, bases) for index, raw in enumerate(raw_items)]
            if any(item['placement'] == 'toss' for item in items) and service.motion_profile != 'throw':
                # Every other profile retimes the swing far below release speed: the object would be let go
                # at the right pose but barely move.
                raise ValueError('抛掷需要 throw 运动档：其它档位会把甩臂放慢，物体抛不出去。'
                                 '请先 PUT /api/settings {"motion_profile": "throw"}，并确认抛掷方向无人')
            self.items = items
            self.options = options
            self.stop_requested = False
            self.finished_at = None
            self.epoch = service.cancel_epoch
            self.active = True
            self.thread = threading.Thread(target=self._run, daemon=True, name='urai-task-queue')
            self.thread.start()
            return self.status()

    def _build_item(self, index, raw, frame, cloud, current, bases):
        service = self.service
        if not isinstance(raw, dict) or set(raw)-ITEM_FIELDS:
            raise ValueError(f'第 {index+1} 项只接受 {", ".join(sorted(ITEM_FIELDS))}')
        arm = raw.get('arm')
        if arm is not None and arm not in ARMS:
            raise ValueError('Choose left or right arm')
        if ('object_pixel' in raw) == ('object_xy' in raw):
            raise ValueError(f'第 {index+1} 项需要 object_pixel 或 object_xy 之一')
        h, w = frame.depth.shape
        table_z = service.table_z(arm)
        if 'object_pixel' in raw:
            pixel = finite_vector(raw['object_pixel'], 2, f'第 {index+1} 项的 object_pixel 需要两个有限像素坐标')
        else:
            xy = finite_vector(raw['object_xy'], 2, f'第 {index+1} 项的 object_xy 需要两个有限世界坐标')
            pixel = frame.project([xy[0], xy[1], table_z])[0]
        if not np.isfinite(pixel).all() or not (0 <= pixel[0] < w and 0 <= pixel[1] < h):
            raise ValueError(f'第 {index+1} 项的物体不在相机画面内')
        yaw = current[arm]['rpy_deg'][2] if arm else 0.
        if 'grasp_pixels' in raw:
            from .grasp import line_object
            if raw.get('placement') != 'toss':
                raise ValueError('画线队列仅支持 toss 投放')
            found = line_object(frame, raw['grasp_pixels'])
            pixel = np.asarray(raw['grasp_pixels'], dtype=float).mean(axis=0)
        else:
            found = locate_object(frame, pixel, table_z, yaw, cloud)
        width = found['width']      # reported, not a gate: see transfer.place_transfer
        if arm is None:
            arm = nearest_arm(bases, found['center'])
        landing = raw.get('landing_xyz')
        if landing is None:
            if service.drop_point is None:
                raise ValueError('尚未设置投放点：请先设置投放点，或为任务项提供 landing_xyz')
            landing = list(service.drop_point)
            source = 'drop_point'
        else:
            landing = finite_vector(landing, 3, f'第 {index+1} 项的 landing_xyz 需要三个有限世界坐标').tolist()
            source = 'item'
        placement = raw.get('placement', 'drop' if source == 'drop_point' else 'place')
        if placement not in PLACEMENTS:
            raise ValueError('placement 只接受 place（放到落点）、drop（落点上方释放）或 toss（抛向落点）')
        label = raw.get('label', f'物体 {index+1}')
        if not isinstance(label, str) or not 1 <= len(label) <= MAX_LABEL_CHARS:
            raise ValueError(f'label 需要 1–{MAX_LABEL_CHARS} 个字符')
        return {'grasp_line_world': found.get('line_world'),
                'index': index, 'kind': 'transfer', 'arm': arm, 'label': label,
                'object_xyz': [*found['center'].tolist(), found['top']],
                'object_pixel': [float(pixel[0]), float(pixel[1])], 'width_mm': width*1000,
                'landing_xyz': landing, 'landing_source': source, 'placement': placement,
                'status': 'queued', 'error': None, 'preview_id': None, 'observation_id': None}

    def _update(self, item, **fields):
        with self.lock:
            item.update(fields)

    def _interrupted(self):
        with self.lock:
            return self.stop_requested or self.service.cancel_epoch != self.epoch

    def _run(self):
        try:
            for item in self.items:
                if self._interrupted():
                    self._update(item, status='skipped')
                    continue
                try:
                    self._run_item(item)
                except Exception as exc:
                    self._update(item, status='cancelled' if self._interrupted() else 'failed', error=str(exc))
                if item['status'] != 'completed':
                    with self.lock:
                        self.stop_requested = True
        finally:
            with self.lock:   # TaskQueue.lock is the service lock
                # Each item wrote its own draft and the queue carried the revision chain between them, so the
                # draft is dropped once here rather than after every item: nothing stays drawn once the queue
                # stops, and no leftover draft can be previewed and run again.
                if self.service.draft:
                    self.service.draft = {}
                    self.service.preview_data = None
                    self.service.revision += 1
                self.active = False
                self.finished_at = time.time()

    def _run_item(self, item):
        service = self.service
        arm = item['arm']
        self._update(item, status='observing')
        service.observe(preserve_draft=True)
        with self.lock:
            frame = service.frame
            revision = service.revision
        self._update(item, observation_id=frame.id)
        pixel = frame.project(item['object_xyz'])[0]
        h, w = frame.depth.shape
        if not np.isfinite(pixel).all() or not (0 <= pixel[0] < w and 0 <= pixel[1] < h):
            raise ValueError('物体不在当前画面内，请重新观测后再提交')
        current = service.backend.state()
        table_z, fingertip_bias_mm = service.arm_calibration(arm)
        if item['placement'] == 'toss':
            style = self.options['toss_style']
            model = service.backend.models.get(arm)
            common = dict(table_z=table_z, clearance_m=self.options['clearance_m'], speed_m_s=self.options['speed_m_s'],
                          preferred_yaw_deg=current[arm]['rpy_deg'][2], fingertip_bias_mm=fingertip_bias_mm,
                          toss_speed_m_s=self.options['toss_speed_m_s'], lead_s=self.options['toss_lead_s'],
                          base_xy=arm_bases(self.service.backend)[arm], style=style,
                          fk=model.fk_tcp_world if model is not None else None,
                          reach_probe=(lambda xyz, rotation, seed: service.backend.pose_error(arm, xyz, rotation, seed, service.pose_tolerance))
                          if model is not None else None,
                          seed_joints=current[arm]['joints_deg'],
                          pose_tolerance=(service.pose_tolerance['position_mm'], service.pose_tolerance['orientation_deg']),
                          carry_speed_m_s=self.options.get('carry_speed_m_s',.6),
                          windup_speed_m_s=self.options.get('windup_speed_m_s',.15),
                          grasp_hold_s=self.options.get('grasp_hold_s',.5))
            # The sidearm's widest swing circle gives the fastest fingertips but may leave the arm's reach;
            # the queue falls back to the next radius. The overhand throw picks its release on fixed lines.
            attempts = [(radius,) for radius in SWING_RADII_M] if style == 'sidearm' else [SWING_RADII_M]
            plan = None
            for radii in attempts:
                manual = {}
                if item.get('grasp_line_world') is not None:
                    from .grasp import grasp_terms
                    pixels = [frame.project(point)[0].tolist() for point in item['grasp_line_world']]
                    manual = {'grasp_pixels': pixels,
                              'grasp_options': {**grasp_terms(arm, model, service.backend.table_checks),
                                                'surface_xyz': item['object_xyz']}}
                draft = toss_between(frame, pixel, item['landing_xyz'], radii=radii, **manual, **common)
                draft['arm']['approach']['speed_m_s'] = self.options.get('approach_speed_m_s', .12)
                saved = service.set_draft({'observation_id': frame.id, 'arms': {arm: draft['arm']},
                                           'expected_revision': revision})
                revision = saved['revision']
                detail = ({'swing_radius_m': draft['swing_radius_m']} if style == 'sidearm' else
                          {'toss_speed_m_s': draft['toss_speed_m_s'], 'throw_distance_m': draft['throw_distance_m'],
                           'landing_shortfall_m': draft['landing_shortfall_m'], 'release_time_s': draft['release_time_s'],
                           'tool_off_deg': draft['tool_off_deg'], 'throw_line': draft['throw_line']})
                self._update(item, status='planning', grasp_xyz=draft['source_xyz'], release_xyz=draft['target_xyz'],
                             toss_style=style, **detail)
                try:
                    plan = service.preview(expected_revision=saved['revision'], expected_cancel_epoch=self.epoch)
                    break
                except ValueError as exc:
                    if radii is attempts[-1] or self._interrupted():
                        raise
                    self._update(item, radius_error=f'半径 {radii[0]:.2f} m：{exc}')
                    with self.lock:
                        revision = service.revision
        else:
            draft = transfer_between(frame, pixel, item['landing_xyz'], table_z=table_z,
                                     clearance_m=self.options['clearance_m'], speed_m_s=self.options['speed_m_s'],
                                     preferred_yaw_deg=current[arm]['rpy_deg'][2],
                                     fingertip_bias_mm=fingertip_bias_mm, drop=item['placement'] == 'drop',
                                     release_tilt_deg=self.options['release_tilt_deg'],
                                     base_xy=arm_bases(self.service.backend)[arm])
            draft['arm']['approach']['speed_m_s'] = self.options.get('approach_speed_m_s', .12)
            saved = service.set_draft({'observation_id': frame.id, 'arms': {arm: draft['arm']},
                                       'expected_revision': revision})
            self._update(item, status='planning', grasp_xyz=draft['source_xyz'], release_xyz=draft['target_xyz'])
            plan = service.preview(expected_revision=saved['revision'], expected_cancel_epoch=self.epoch)
        self._update(item, status='running', preview_id=plan['id'], release_speed_m_s=release_speed(plan['arms'][arm]))
        service.execute(plan['id'], cancel_epoch=self.epoch,
                        task={'index': item['index'], 'label': item['label'], 'arm': arm, 'home_speed_m_s':self.options.get('home_speed_m_s',.6)})
        service.worker.join()
        with self.lock:
            final = copy.deepcopy(service.execution)
        if final['state'] == 'completed':
            self._update(item, status='completed')
        elif final['state'] == 'cancelled':
            self._update(item, status='cancelled')
        else:
            self._update(item, status='failed', error=final.get('error', final['state']))

    def cancel(self):
        with self.lock:
            self.stop_requested = True
            if self.active:
                self.service.cancel()
            return self.status()
