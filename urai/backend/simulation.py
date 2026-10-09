"""Synthetic backend for development without hardware: no IK, no collision model, ideal tracking."""
from __future__ import annotations

import threading
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ..approach import needs_approach, with_approaches
from ..geometry import Frame
from ..settings import DEFAULT_TABLE_CHECKS, DEFAULT_TRACKING_LIMIT_DEG, motion_limits, normalize_pose_tolerance
from ..trajectory import compile_arm
from .common import ARMS, HOME_SPEED_M_S, Cancelled, home_gripper_event, home_note


class SimulationBackend:
    mode = 'simulation'
    #: Same setting names as the hardware backend so the service and the console treat both alike.
    tracking_limit_deg = DEFAULT_TRACKING_LIMIT_DEG
    table_checks = DEFAULT_TABLE_CHECKS
    #: No calibrated arm models: skills fall back to geometry-only candidates.
    models = {}

    def __init__(self):
        self.positions = {'left': np.array([.25, .12, .24]), 'right': np.array([.25, -.45, .24])}
        self.rotations = {a: [180., 0., 0.] for a in ARMS}
        self.openings = dict.fromkeys(ARMS, 0.)
        self.home_positions = {arm: position.copy() for arm, position in self.positions.items()}
        self.home_rotations = {arm: list(rotation) for arm, rotation in self.rotations.items()}
        self.hold_count = 0
        self.capture_lock = threading.Lock()

    def info(self):
        return {'mode': self.mode, 'name': 'Simulation', 'motion_allowed': True,
                'checks': ['synthetic workspace'], 'force_control': False, 'table_checks': self.table_checks}

    def state(self):
        return {a: {'xyz': self.positions[a].tolist(), 'rpy_deg': list(self.rotations[a]),
                    'joints_deg': (self.positions[a]*100).tolist(), 'gripper_mm': self.openings[a],
                    'enabled': True, 'feedback_age_s': 0., 'ctrl_mode': 'simulation'} for a in ARMS}

    def capture(self):
        with self.capture_lock:
            return self._synthetic_frame()

    def _synthetic_frame(self):
        # Explicit synthetic scene for development. Never substituted for real observations.
        h, w = 480, 800
        rgb = np.full((h, w, 3), [65, 73, 83], dtype=np.uint8)
        rgb[::40, :] = [91, 100, 110]; rgb[:, ::40] = [91, 100, 110]
        depth = np.full((h, w), 1.0)
        depth[180:260, 350:440] = .94
        rgb[180:260, 350:440] = [190, 130, 70]
        k = np.array([[600., 0, 400], [0, 600, 240], [0, 0, 1]])
        t = np.eye(4); t[:3, :3] = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        t[:3, 3] = [.4, -.2, 1.]
        return Frame(rgb, depth, k, t)

    def plan(self, compiled, start, pose_tolerance=None, progress=None, motion_profile='normal'):
        if progress:
            progress(stage='approach',done=0,total=1)
        compiled = with_approaches(compiled, start, {a:'direct' for a,p in compiled.items() if needs_approach(p,start[a])})
        return {'duration_s': max(a['time_s'][-1] for a in compiled.values()), 'arms': compiled,
                'approach_duration_s': max(p.get('task_offset_s',0.) for p in compiled.values()),
                'retiming_factor': 1., 'pose_tolerance': normalize_pose_tolerance(pose_tolerance),
                'motion_limits': motion_limits(motion_profile),
                'notes': ['Simulation: no hardware IK or collision validation']}

    def plan_home(self, start, arms, pose_tolerance=None, progress=None, motion_profile='normal', open_grippers=True, speed_m_s=HOME_SPEED_M_S):
        """Open the gripper, then move the selected arms back to the simulation's initial pose."""
        compiled = {}
        for arm in arms:
            state = start[arm]
            compiled[arm] = compile_arm({
                'path': {'mode': 'waypoints', 'points': [list(state['xyz']), self.home_positions[arm].tolist()]},
                'speed': speed_m_s,
                'orientation': {'mode': 'keyframes', 'points': [[0, *state['rpy_deg']], [1, *self.home_rotations[arm]]]},
                'gripper_events': [home_gripper_event()] if open_grippers else []}, state)
        plan = self.plan(compiled, start, pose_tolerance, progress, motion_profile)
        plan['notes'].insert(0, home_note(arms, '回到模拟初始位置'))
        return plan

    def validate_start(self, start):
        now = self.state()
        for a in ARMS:
            if np.max(np.abs(np.array(now[a]['joints_deg'])-start[a]['joints_deg'])) > .5:
                raise ValueError(f'{a} moved since preview; generate a new preview')

    def prepare(self, cancel, progress, restore_can=True, restore_disabled=False):
        if cancel.is_set():
            raise Cancelled()
        return {'restored_arms': []}

    def execute(self, plan, start, cancel, progress):
        begun = time.monotonic(); fired = set()
        while True:
            if cancel.is_set():
                self.hold_count += 1
                raise Cancelled()
            t = min(time.monotonic()-begun, plan['duration_s'])
            for arm, p in plan['arms'].items():
                self.positions[arm] = np.array([np.interp(t, p['time_s'], p['xyz'][:, j]) for j in range(3)])
                i = min(np.searchsorted(p['time_s'], t), len(p['time_s'])-1)
                self.rotations[arm] = Rotation.from_matrix(p['rotation_matrices'][i]).as_euler('xyz', degrees=True).tolist()
                for n, e in enumerate(p['gripper_events']):
                    if t >= e['time_s']-e.get('lead_s', 0.) and (arm, n) not in fired:
                        if 'opening_mm' in e:
                            self.openings[arm] = e['opening_mm']
                        fired.add((arm, n))
            progress(t/plan['duration_s'], t)
            if t >= plan['duration_s']:
                return {'completed': True, 'mode': self.mode}
            cancel.wait(.02)
