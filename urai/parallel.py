"""Persistent worker processes for the planner's IK chains and per-sample geometry checks.

The workers are spawned once per backend and rebuild both arm models from their calibration
parameters. They run the very same :class:`urai.backend.planner.PiperPlanner` methods as the
serial path: each arm's IK chain (``_solve_arm_joints``, arms in parallel, with the per-plan IK
memo shipped along) and the collision sweep (``_geometry_check`` on contiguous chunks of the
40 ms samples, the earliest failing sample and its message returned unchanged). Running them in
separate processes also keeps them clear of the service's camera and control threads.
"""
from __future__ import annotations

import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from concurrent.futures.process import BrokenProcessPool
import numpy as np

#: The planner built once per worker process.
_WORKER = None


def model_spec(model):
    """The parameters that rebuild an ``ArmModel`` for geometry checks in another process."""
    return {'name': model.name, 'urdf_path': str(model.kin.urdf_path),
            't_world_base': np.asarray(model.t_world_base, dtype=float), 'tcp_offset_mm': float(model.tcp_offset_mm),
            'table_z_mm': float(model.table_z_mm), 'link_clearance_mm': float(model.link_clearance_mm),
            'fingertip_bias_mm': float(model.fingertip_bias_mm),
            'fingertip_below_table_mm': float(model.fingertip_below_table_mm),
            'max_reach_m': float(model.max_reach_m), 'ik_position_tol_mm': float(model.ik_position_tol_mm)}


def build_model(spec):
    from .robot.arm_model import ArmModel
    from .robot.kinematics import PiperXKinematics
    spec = dict(spec)
    return ArmModel(spec.pop('name'), PiperXKinematics(spec.pop('urdf_path')), spec.pop('t_world_base'), **spec)


def _initialize(specs):
    global _WORKER
    from .backend.planner import PiperPlanner
    _WORKER = PiperPlanner({arm: build_model(spec) for arm, spec in specs.items()})


def _solve_arm(arm, p, state, tolerance, table_checks, cache):
    """One arm's IK chain; returns the solved trajectory dict, its table lifts and the IK memo."""
    _WORKER.table_checks = table_checks
    _WORKER._ik_cache = cache
    lifts = _WORKER._solve_arm_joints(arm, p, state, tolerance, lambda **detail: None)
    return p, lifts, _WORKER._ik_cache


def _check_chunk(offset, left, right, table_checks, joint_bounds):
    """The first failing sample of one chunk as (index, message), or None when every sample passes."""
    _WORKER.table_checks = table_checks
    kwargs = {'joint_bounds': joint_bounds} if joint_bounds else {}
    for i, (qa, qb) in enumerate(zip(left, right)):
        try:
            _WORKER._geometry_check(qa, qb, **kwargs)
        except ValueError as exc:
            return offset + i, str(exc)
    return None


class PlannerPool:
    """Spawned worker processes that solve IK chains and check joint samples with the real arm models."""

    def __init__(self, models, workers):
        self.specs = {arm: model_spec(model) for arm, model in models.items()}
        self.workers = int(workers)
        self._executor = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._executor is None:
                self._executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context('spawn'),
                                                     initializer=_initialize, initargs=(self.specs,))
            return self._executor

    def warm_up(self):
        """Spawn every worker now so the first preview does not wait for interpreter start-up."""
        executor = self.start()
        for future in [executor.submit(os.getpid) for _ in range(self.workers)]:
            future.result()

    def worker_pids(self):
        with self._lock:
            return [] if self._executor is None else list(self._executor._processes)

    def solve_arms(self, compiled, start, tolerance, table_checks, cache, tick=None):
        """Solve every arm's joint samples in parallel.

        Returns ``{arm: (trajectory, lifts, memo)}`` in the order of ``compiled``; a worker's
        typed planning error is raised unchanged. ``tick`` runs every 0.1 s while waiting so
        the caller can report progress or cancel.
        """
        executor = self.start()
        try:
            futures = {arm: executor.submit(_solve_arm, arm, p, start[arm], tolerance, table_checks,
                                            {key: value for key, value in cache.items() if key[0] == arm})
                       for arm, p in compiled.items()}
            pending = set(futures.values())
            while pending:
                pending = wait(pending, timeout=.1).not_done
                if tick is not None:
                    tick()
            return {arm: future.result() for arm, future in futures.items()}
        except BrokenProcessPool as exc:
            self.close()
            raise ValueError('规划工作进程已退出，工作池已重建；请重新预览') from exc

    def check(self, left, right, table_checks, joint_bounds, progress=None):
        """Earliest failing sample of the paired joint arrays as (index, message), or None."""
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        pieces = [piece for piece in np.array_split(np.arange(len(left)), max(1, min(len(left), 2 * self.workers))) if len(piece)]
        executor = self.start()
        try:
            futures = {executor.submit(_check_chunk, int(piece[0]), left[piece], right[piece], table_checks, joint_bounds): len(piece)
                       for piece in pieces}
            failures = []
            done = 0
            for future in as_completed(futures):
                result = future.result()
                done += futures[future]
                if progress is not None:
                    progress(done)
                if result is not None:
                    failures.append(result)
        except BrokenProcessPool as exc:
            self.close()
            raise ValueError('规划工作进程已退出，工作池已重建；请重新预览') from exc
        return min(failures) if failures else None

    def close(self):
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
