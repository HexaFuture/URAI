"""Planning half of the PiPER-X backend: IK chains, automatic approaches, time scaling and geometry checks.

Everything here is computed from the calibrated arm models; nothing talks to hardware. The execution half
(:mod:`urai.backend.piperx`) adds the robot readiness checks, the execution lock and the 50 Hz loop.
Planner worker processes (:mod:`urai.parallel`) instantiate this class directly.
"""
from __future__ import annotations

from contextlib import nullcontext
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from ..approach import needs_approach, with_approaches
from ..robot.arm_model import INTER_ARM_MARGIN_MM, capsules_clearance_mm, inter_arm_reason, joint_limit_reason
from ..settings import DEFAULT_TABLE_CHECKS, motion_limits, normalize_pose_tolerance
from ..trajectory import SAMPLES, time_samples
from .common import ARMS, HOME_HOLD_S, HOME_SPEED_M_S, ApproachError, FloorError, RobotStateError, TaskError, home_gripper_event, home_note
from .model_checks import (SELF_COLLISION_MIN_CLEARANCE_MM, canonical_planned_joints, floor_world_mm, measured_joint_bounds,
                           model_joint_limit_reason, recovery_joint_bounds, self_collision_reason, solve_ik)
from .retiming import angular_interval_scales, interval_retiming_scales, joint_curve


class PiperPlanner:
    """Plans dual-arm trajectories against the two calibrated PiPER-X arm models."""
    #: When False the planner and the 50 Hz feedback guard skip the table checks (link clearance above the
    #: table and the fingertip floor); joint limits, self-collision and inter-arm clearance stay active.
    table_checks = DEFAULT_TABLE_CHECKS
    #: Per-plan memo of IK solves keyed by (arm, target, seed); None outside ``plan``.
    _ik_cache = None
    #: Worker processes for IK chains and the collision sweep (:class:`urai.parallel.PlannerPool`); None runs in-process.
    planner_pool = None

    def __init__(self, models, workers=0):
        self.models = models
        if workers:
            from ..parallel import PlannerPool
            self.planner_pool = PlannerPool(models, workers)

    def warm_up(self):
        """Spawn the planner workers ahead of the first preview."""
        if self.planner_pool is not None:
            self.planner_pool.warm_up()

    def close(self):
        """Release the planner workers."""
        if self.planner_pool is not None:
            self.planner_pool.close()

    def validate_start(self, start):
        """Planning alone has no live robot to compare against; the hardware backend checks it."""

    def _exclusive(self):
        """Planning alone holds no hardware; the hardware backend takes its execution lock here."""
        return nullcontext()

    def _geometry_check(self, qa, qb, *, feedback=False, joint_bounds=None):
        """Raise ValueError when the joint pair violates a limit, the table, self-collision or inter-arm clearance.

        ``feedback`` marks measured joints: they may cross a nominal bound during hand guiding or servo
        settling, so only their geometry is validated (see :meth:`_feedback_geometry`).
        """
        if feedback:
            return self._feedback_geometry(qa, qb)
        qa, qb = [canonical_planned_joints(self.models[a], q) for a, q in zip(ARMS, (qa, qb))]
        for a, q in zip(ARMS, (qa, qb)):
            model = self.models[a]
            bounds = (joint_bounds or {}).get(a)
            why = model_joint_limit_reason(model, q, bounds=bounds) if bounds is not None else joint_limit_reason(q)
            why = why or (model.table_check(q) if self.table_checks else None) or self_collision_reason(model, q)
            if why:
                raise ValueError(f'{a}: {why}')
        why = inter_arm_reason(self.models['left'], qa, self.models['right'], qb)
        if why:
            raise ValueError(why)

    def _feedback_geometry(self, qa, qb):
        """One capsule construction per changed arm, shared by self/inter-arm checks.

        Uses the arm model's capsule radii, transforms, distance function and thresholds.
        Exact feedback equality can reuse geometry; changed joints always rebuild it.
        """
        cache = getattr(self, '_feedback_capsules', {})
        caps = {}
        for a, q in zip(ARMS, (qa, qb)):
            model = self.models[a]
            q = np.asarray(q, dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError(f'{a}: joint feedback must contain six finite angles')
            why = model.table_check(q) if self.table_checks else None
            if why:
                raise ValueError(f'{a}: {why}')
            key = (id(model), q.tobytes())
            cached = cache.get(a)
            if cached is None or cached[0] != key:
                cache[a] = (key, model.body_capsules(q))
            caps[a] = cache[a][1]
            # Names, rather than indices, keep this identical to the model's self-collision geometry.
            movers = [c for c in caps[a] if c[0] in ('tool body', 'left finger', 'right finger')]
            targets = [c for c in caps[a] if c[0] in ('base plate', 'base column', 'upper arm')]
            if len(movers) != 3 or len(targets) != 3:
                raise ValueError('Feedback capsule model is incomplete')
            clearance, mover, target = capsules_clearance_mm(movers, targets)
            if clearance < SELF_COLLISION_MIN_CLEARANCE_MM:
                raise ValueError(f'{a}: {model.name} self-collision: {mover} vs {target} (clearance {clearance:.0f} mm)')
        self._feedback_capsules = cache
        clearance, part, other_part = capsules_clearance_mm(caps['left'], caps['right'])
        if clearance < INTER_ARM_MARGIN_MM:
            left, right = self.models['left'].name, self.models['right'].name
            raise ValueError(f"{left} would hit the {right} arm: {left} {part} vs {right} {other_part} "
                             f"(clearance {clearance:.0f} mm < {INTER_ARM_MARGIN_MM:.0f}); the {right} arm is in the way — "
                             f"home it first (POST /api/home) or choose a target farther from it")

    def _local_retime(self,compiled,limits,progress=None):
        """Stretch local intervals on a shared dual-arm clock, then verify again."""
        offset=max(p.get('task_offset_s',0.) for p in compiled.values())
        raw=np.unique(np.concatenate([np.array([0.,offset])]+[p['time_s'] for p in compiled.values()]))
        timed=raw.copy();boundary=int(np.searchsorted(raw,offset));indices={};bounds={}
        for a,p in compiled.items():
            indices[a]=np.searchsorted(raw,p['time_s'])
            model=self.models[a]
            bounds[a]=np.asarray([model.kin.limits_deg[i+1] for i in range(6)])
            if p.get('recovery_joint_bounds') is not None:
                bounds[a]=p['recovery_joint_bounds']
        def requirements():
            required=np.ones(len(raw)-1)
            for a,p in compiled.items():
                index=indices[a]
                curve=joint_curve(timed[index],p['joints_deg'],p.get('stop_indices',()),bounds[a])
                local=interval_retiming_scales(curve,limits)
                if p.get('orientation_mode')=='free':
                    local=np.maximum(local,angular_interval_scales(self.models[a],curve))
                for i,value in enumerate(local):
                    required[index[i]:index[i+1]]=np.maximum(required[index[i]:index[i+1]],value)
            return required
        # Rebuilding a C2 curve changes derivatives at neighboring knots, so
        # recompute after each local adjustment instead of trusting one pass.
        for iteration in range(8):
            if progress: progress(stage='retiming',arm=None,done=iteration,total=8)
            need=requirements()
            if need.max()<=1.+1e-8:
                return raw,timed
            growth=np.where(need>1.,need*1.03,1.)
            spread=growth.copy()
            for distance in range(1,min(13,len(growth))):
                weight=np.exp(-distance/4.)
                spread[distance:]=np.maximum(spread[distance:],growth[:-distance]**weight)
                spread[:-distance]=np.maximum(spread[:-distance],growth[distance:]**weight)
            timed=np.r_[0.,np.cumsum(np.diff(timed)*spread)]
        # Bounded fallback: uniform scaling within each stopped phase preserves
        # the final interpolant's shape and guarantees joint derivative limits.
        need=requirements();growth=np.ones_like(need)
        for lo,hi in [(0,boundary),(boundary,len(need))]:
            if hi>lo: growth[lo:hi]=max(1.,float(need[lo:hi].max()))*1.02
        return raw,np.r_[0.,np.cumsum(np.diff(timed)*growth)]

    def pose_error(self, arm, xyz, rotation, seed_joints, pose_tolerance=None):
        """Best-effort IK errors of one TCP pose for ``arm``: ``(position_mm, rotation_deg, joints)``.

        Poses the IK rejects outright (for example a solution outside the joint limits) come back as
        ``(inf, inf, None)`` so draft generators can rank candidate poses without parsing the reason text.
        """
        target = np.eye(4); target[:3, :3] = np.asarray(rotation, dtype=float); target[:3, 3] = np.asarray(xyz, dtype=float)
        try:
            q, position_error, rotation_error = solve_ik(self.models[arm], target, np.asarray(seed_joints, dtype=float), pose_tolerance)
        except ValueError:
            return float('inf'), float('inf'), None
        return float(position_error), float(rotation_error), q

    def _cached_ik(self, arm, model, target, seed, tolerance, free):
        """``solve_ik`` with reuse inside one ``plan`` call.

        Approach candidates end at the same task start, so the task segment (and any shared
        approach prefix) repeats the exact (target, seed) chain; a candidate then costs no IK.
        """
        key = None
        if self._ik_cache is not None:
            key = (arm, free, np.asarray(target, dtype=float).tobytes(), np.asarray(seed, dtype=float).tobytes())
            hit = self._ik_cache.get(key)
            if hit is not None:
                return hit
        result = solve_ik(model, target, seed, tolerance, **({'orientation_free': True} if free else {}))
        if key is not None:
            self._ik_cache[key] = result
        return result

    FLOOR_MARGIN_MM = .5
    FLOOR_LIFT_ROUNDS = 3
    #: Roundoff allowance of the floor comparison: an exact IK solution lands within 1e-13 mm of the floor.
    FLOOR_ROUNDOFF_MM = 1e-6

    def _keep_above_floor(self, arm, model, target, q, pos_error, rot_error, tolerance, free):
        """Re-solve IK with a raised target until the real fingertip clears the table floor.

        Generators clamp their targets to the floor, but the accepted IK residual can leave the
        solved fingertip a millimetre lower, which the model's floor check then rejects. Lifting only
        the offending sample keeps the rest of the drawn path intact. Returns the total lift in mm.
        """
        floor = floor_world_mm(model)+self.FLOOR_MARGIN_MM
        bias = float(model.fingertip_bias_mm)
        lifted = 0.
        for _ in range(self.FLOOR_LIFT_ROUNDS):
            tip = float(model.fk_tcp_world(q)[2, 3])*1000.+bias
            if tip >= floor-self.FLOOR_ROUNDOFF_MM:
                return q, pos_error, rot_error, lifted
            deficit = floor-tip
            target[2, 3] += deficit/1000.
            lifted += deficit
            q, pos_error, rot_error = self._cached_ik(arm, model, target, q, tolerance, free)
        tip = float(model.fk_tcp_world(q)[2, 3])*1000.+bias
        if tip < floor-self.FLOOR_ROUNDOFF_MM:
            name = '左臂' if arm == 'left' else '右臂'
            raise FloorError(f'{name}：指尖高度 {tip:.1f} mm 低于桌面允许的 {floor-self.FLOOR_MARGIN_MM:.1f} mm，'
                             f'抬高 {lifted:.1f} mm 后仍无法满足；请把这段路径抬高后重试')
        return q, pos_error, rot_error, lifted

    def _joint_approach(self, arm, task, start, pose_tolerance=None):
        tolerance = normalize_pose_tolerance(pose_tolerance)
        position_limit = tolerance['position_mm']; rotation_limit = tolerance['orientation_deg']
        model = self.models[arm]
        free = task.get('orientation_mode')=='free'
        target = np.eye(4); target[:3,3] = task['xyz'][0]; target[:3,:3] = task['rotation_matrices'][0]
        q0 = np.asarray(start['joints_deg'], dtype=float)
        goal, pos_error, rot_error = self._cached_ik(arm, model, target, q0, tolerance, free)
        if goal is None or pos_error > position_limit or (not free and rot_error > rotation_limit):
            raise ValueError(f'{arm} approach: pose error {pos_error:.2f} mm / {rot_error:.2f} deg (allowed {position_limit:g} mm / {rotation_limit:g} deg)')
        achieved = model.fk_tcp_world(goal)
        if np.linalg.norm(achieved[:3,3]-target[:3,3])*1000 > position_limit or (not free and (Rotation.from_matrix(achieved[:3,:3]).inv()*Rotation.from_matrix(target[:3,:3])).magnitude() > np.deg2rad(rotation_limit)):
            raise ValueError(f'{arm}: joint approach start solution exceeds FK residual tolerance')
        s = np.linspace(0,1,SAMPLES)
        qs = q0[None,:] + s[:,None]*(goal-q0)[None,:]
        poses = np.asarray([model.fk_tcp_world(q) for q in qs])
        xyz = poses[:,:3,3]; rotations = Rotation.from_matrix(poses[:,:3,:3])
        requested = np.full(SAMPLES, task['approach_config']['speed_m_s'])
        times, length, requested_duration = time_samples(xyz, rotations, requested)
        return {'s':s, 'xyz':xyz, 'rotation_matrices':rotations.as_matrix(), 'time_s':times,
                'requested_speed_m_s':requested, 'requested_duration_s':requested_duration,
                'path_length_m':length, 'gripper_events':[], 'route':'joint', 'source_joints_deg':qs,
                'orientation_mode':task.get('orientation_mode','hold')}

    def _home_arm(self, arm, state, open_grippers=True, speed_m_s=HOME_SPEED_M_S):
        """Joint-space straight line from the measured joints to the all-zero pose, preceded by the opening dwell.

        The whole joint sequence is handed to ``_plan`` as ``approach_joints_deg`` so it skips IK but still fits
        the curve, retimes against the joint limits and runs the table, self-collision and inter-arm checks.
        """
        model = self.models[arm]
        q0 = np.asarray(state['joints_deg'], dtype=float)
        s = np.linspace(0, 1, SAMPLES)
        joints = q0[None, :] * (1 - s)[:, None]
        joints[-1] = 0.
        poses = np.asarray([model.fk_tcp_world(q) for q in joints])
        xyz = poses[:, :3, 3]
        rotations = Rotation.from_matrix(poses[:, :3, :3])
        requested = np.full(SAMPLES, speed_m_s)
        times, length, requested_duration = time_samples(xyz, rotations, requested)
        matrices = rotations.as_matrix()
        # Same dwell layout as compile_arm: the start pose is repeated so the curve stands still while the gripper opens.
        event = home_gripper_event()
        event['time_s'] = 0.
        event['hold_end_time_s'] = HOME_HOLD_S
        return {'s': np.r_[0., s], 'xyz': np.vstack([xyz[:1], xyz]), 'rotation_matrices': np.concatenate([matrices[:1], matrices]),
                'orientation_mode': 'joint', 'time_s': np.r_[0., times + HOME_HOLD_S],
                'requested_speed_m_s': np.r_[0., 0., requested[1:]], 'requested_duration_s': requested_duration + HOME_HOLD_S,
                'path_length_m': length, 'gripper_events': [event] if open_grippers else [], 'stop_indices': [0, 1], 'start_hold_s': 0.,
                'approach_config': {'speed_m_s': speed_m_s, 'clearance_m': .12},
                'approach_joints_deg': np.vstack([joints[:1], joints]), 'route': 'home'}

    def plan_home(self, start, arms, pose_tolerance=None, progress=None, motion_profile='normal', open_grippers=True, speed_m_s=HOME_SPEED_M_S):
        """Plan the selected arms back to the all-zero joint pose; unselected arms stay put and join every check."""
        arms = list(arms)
        report = progress or (lambda **detail: None)
        with self._exclusive():
            self.validate_start(start)
            report(stage='approach', candidate=1, candidates=1, route='关节空间回零', done=0, total=1, arm=None)
            compiled = {arm: self._home_arm(arm, start[arm], open_grippers=open_grippers, speed_m_s=speed_m_s) for arm in arms}
            result = self._plan(compiled, start, pose_tolerance, progress=report, motion_profile=motion_profile)
            result['approach_routes'] = {arm: 'home' for arm in arms}
            result['notes'].insert(0, home_note(arms, '关节空间线性回到全零位'))
            return result

    def plan(self, compiled, start, pose_tolerance=None, progress=None, motion_profile='normal'):
        tolerance = normalize_pose_tolerance(pose_tolerance)
        # A vertical garment pinch is intentionally preferred over an exact
        # wrist orientation.  The real robot can be off by a few degrees while
        # still placing both fingers correctly; the old 3° gate rejected valid
        # points (for example 3.20° residual with 5.92 mm position error).
        # Relax only grasp-line plans; ordinary trajectories retain the global
        # 3° tolerance.
        if any(p.get('grasp_line') for p in compiled.values()):
            tolerance = dict(tolerance)
            tolerance['orientation_deg'] = max(float(tolerance['orientation_deg']), 6.)
        report = progress or (lambda **detail: None)
        with self._exclusive():
            self.validate_start(start)
            recovering = [a for a in compiled if
                model_joint_limit_reason(self.models[a],start[a]['joints_deg'],tolerance_deg=1e-9)]
            changed = [a for a,p in compiled.items() if a in recovering or needs_approach(p,start[a])]
            if not changed:
                report(stage='approach',candidate=1,candidates=1,route='无需接近',done=0,total=1,arm=None)
                return self._plan(compiled, start, tolerance, progress=report, motion_profile=motion_profile)
            errors = []
            # An arm's approach segment only depends on its own start and route, so once it fails
            # by itself every combination that keeps that route is skipped. Task failures end the
            # search at once; other geometric failures (approach-phase collisions) try the next combination.
            infeasible = set()
            options=[('joint',) if a in recovering else
                     (('direct','joint') if compiled[a].get('air_track') else ('direct','lift','joint'))
                     for a in changed]
            candidates = int(np.prod([len(o) for o in options]))
            self._ik_cache = {}
            try:
                for number, choices in enumerate(product(*options), 1):
                    routes = dict(zip(changed, choices))
                    if any((a, r) in infeasible for a, r in routes.items()):
                        continue
                    report(stage='approach',candidate=number,candidates=candidates,routes=routes,done=0,total=1,arm=None)
                    self.validate_start(start)
                    try:
                        prepared = {a:self._joint_approach(a,compiled[a],start[a],tolerance) for a,r in routes.items() if r=='joint'}
                        candidate = with_approaches(compiled, start, routes, prepared)
                        result = self._plan(candidate, start, tolerance, progress=report, motion_profile=motion_profile)
                        result['approach_routes'] = routes
                        result['notes'].insert(0, 'Automatic low-speed approach: '+', '.join(f'{a}={r}' for a,r in routes.items())+
                                               '. Both user trajectories start after both arms reach their start poses.')
                        return result
                    except (RobotStateError, FloorError, TaskError):
                        raise
                    except ValueError as exc:
                        self.validate_start(start)
                        if isinstance(exc, ApproachError):
                            infeasible.add((exc.arm, routes[exc.arm]))
                        errors.append(f'{routes}: {exc}')
            finally:
                self._ik_cache = None
            raise ValueError('No feasible complete approach + task trajectory. '+'; '.join(errors))

    def _solve_arm_joints(self, a, p, state, tolerance, report):
        """Solve the joint samples of one arm's combined approach + task trajectory in place.

        Fills ``p['joints_deg']`` and, where applicable, the solved rotations (free orientation),
        lifted targets (table checks) and recovery bounds (taught start). Failures are typed by
        segment: ApproachError for approach samples, TaskError from the task start onwards.
        Returns the list of table lifts in millimetres.
        """
        position_limit = tolerance['position_mm']; rotation_limit = tolerance['orientation_deg']
        model = self.models[a]
        free = p.get('orientation_mode')=='free'
        q = np.asarray(state['joints_deg'], dtype=float)
        pose0 = model.fk_tcp_world(q)
        rotation_error = (Rotation.from_matrix(pose0[:3, :3]).inv()*Rotation.from_matrix(p['rotation_matrices'][0])).magnitude()
        if np.linalg.norm(p['xyz'][0]-pose0[:3, 3]) > .002 or (not free and rotation_error > np.deg2rad(1)):
            raise ValueError(f'{a}: path must start at the current TCP position and orientation; add an approach segment')
        task_start = len(p['s'])-len(p['user_trajectory']['s']) if p.get('user_trajectory') is not None else 0
        qs = [q.copy()]
        lifts = []
        actual_rotations = [pose0[:3,:3].copy()]
        approach_qs = p.get('approach_joints_deg', [])
        for i in range(1, len(p['s'])):
            report(stage='ik',arm=a,done=i-1,total=len(p['s'])-1)
            target = np.eye(4); target[:3, :3] = p['rotation_matrices'][i]; target[:3, 3] = p['xyz'][i]
            try:
                if i < len(approach_qs):
                    q = approach_qs[i].copy(); pos_error = 0.; rot_error = 0.
                elif np.allclose(p['xyz'][i],p['xyz'][i-1],atol=1e-10,rtol=0) and np.allclose(p['rotation_matrices'][i],p['rotation_matrices'][i-1],atol=1e-10,rtol=0):
                    q = q.copy(); pos_error = 0.; rot_error = 0.
                else:
                    q, pos_error, rot_error = self._cached_ik(a, model, target, q, tolerance, free)
                    if self.table_checks:
                        q, pos_error, rot_error, lifted_mm = self._keep_above_floor(a, model, target, q, pos_error, rot_error, tolerance, free)
                        if lifted_mm:
                            p['xyz'][i] = target[:3, 3]
                            lifts.append(lifted_mm)
                if q is None or pos_error > position_limit or (not free and rot_error > rotation_limit):
                    if free:
                        raise ValueError(f'{a} path at {p["s"][i]:.3f}: position error {pos_error:.2f} mm (allowed {position_limit:g} mm; orientation is free)')
                    raise ValueError(f'{a} path at {p["s"][i]:.3f}: pose error {pos_error:.2f} mm / {rot_error:.2f} deg (allowed {position_limit:g} mm / {rotation_limit:g} deg); adjust position or orientation')
                actual = model.fk_tcp_world(q)
                error_mm = np.linalg.norm(actual[:3, 3]-target[:3, 3])*1000
                error_deg = np.rad2deg((Rotation.from_matrix(actual[:3, :3]).inv()*Rotation.from_matrix(target[:3, :3])).magnitude())
                if error_mm > position_limit or (not free and error_deg > rotation_limit):
                    raise ValueError(f'{a}: final pose error {error_mm:.2f} mm / {error_deg:.2f} deg (allowed {position_limit:g} mm / {rotation_limit:g} deg)')
            except FloorError:
                raise
            except ValueError as exc:
                raise (TaskError(str(exc)) if i >= task_start else ApproachError(str(exc), a)) from exc
            qs.append(q.copy())
            actual_rotations.append(actual[:3,:3].copy())
        p['joints_deg'] = np.asarray(qs)
        if lifts:
            if p.get('user_trajectory') is not None:
                task = p['user_trajectory']
                task['xyz'] = p['xyz'][-len(task['s']):].copy()
            if p.get('approach') is not None:
                approach = p['approach']
                approach['xyz'] = p['xyz'][:len(approach['s'])].copy()
        # Homing is one joint-space recovery move: every knot may return from a taught
        # or limit-grazing pose as long as it keeps moving inward. Other routes only
        # recover during their approach; the task segment must stay inside the bounds.
        task_index=len(p['joints_deg']) if p.get('route')=='home' else int(np.searchsorted(p['time_s'],p.get('task_offset_s',0.)))
        recovery=recovery_joint_bounds(model,p['joints_deg'],task_index)
        if model_joint_limit_reason(model,qs[0],tolerance_deg=1e-9):
            p['recovery_joint_bounds']=recovery
        if free:
            p['rotation_matrices'] = np.asarray(actual_rotations)
            if p.get('user_trajectory') is not None:
                task=p['user_trajectory'];task['rotation_matrices']=p['rotation_matrices'][-len(task['s']):].copy()
            if p.get('approach') is not None:
                approach=p['approach'];approach['rotation_matrices']=p['rotation_matrices'][:len(approach['s'])].copy()
        report(stage='ik',arm=a,done=len(p['s'])-1,total=len(p['s'])-1)
        return lifts

    def _solve_arms_pooled(self, compiled, start, tolerance, report):
        """Solve every arm's joint samples in the worker pool, arms in parallel; returns table lifts by arm."""
        cache = self._ik_cache if self._ik_cache is not None else {}
        for a, p in compiled.items():
            report(stage='ik', arm=a, done=0, total=len(p['s'])-1)
        solved = self.planner_pool.solve_arms(compiled, start, tolerance, self.table_checks, cache,
                                              tick=lambda: report(stage='ik', arm=None, done=0, total=1))
        floor_lifts = {}
        for a, (p, lifts, memo) in solved.items():
            compiled[a] = p
            if lifts:
                floor_lifts[a] = lifts
            if self._ik_cache is not None:
                self._ik_cache.update(memo)
            report(stage='ik', arm=a, done=len(p['s'])-1, total=len(p['s'])-1)
        return floor_lifts

    def _plan(self, compiled, start, pose_tolerance=None, progress=None, motion_profile='normal'):
        tolerance = normalize_pose_tolerance(pose_tolerance)
        report = progress or (lambda **detail: None)
        position_limit = tolerance['position_mm']; rotation_limit = tolerance['orientation_deg']
        self.validate_start(start)
        floor_lifts = {}
        if self.planner_pool is not None:
            floor_lifts = self._solve_arms_pooled(compiled, start, tolerance, report)
        else:
            for a, p in compiled.items():
                report(stage='ik',arm=a,done=0,total=len(p['s'])-1)
                lifts = self._solve_arm_joints(a, p, start[a], tolerance, report)
                if lifts:
                    floor_lifts[a] = lifts
        # Both arms follow the same wall clock. A shorter arm finishes and holds.
        limits=motion_limits(motion_profile)
        offset=max(p.get('task_offset_s',0.) for p in compiled.values())
        raw_clock,timed_clock=self._local_retime(compiled,limits,report)
        def warp(t):
            return np.interp(t,raw_clock,timed_clock)
        scales={'approach':float(warp(offset)/offset) if offset>0 else 1.,
                'task':float((timed_clock[-1]-warp(offset))/(raw_clock[-1]-offset)) if raw_clock[-1]>offset else 1.}
        original_duration=float(raw_clock[-1])
        bounds={a:np.asarray([self.models[a].kin.limits_deg[i+1] for i in range(6)]) for a in compiled}
        initial_bounds={a:measured_joint_bounds(self.models[a],start[a]['joints_deg'])
                        for a in ARMS if
                        model_joint_limit_reason(self.models[a],start[a]['joints_deg'],tolerance_deg=1e-9)}
        bounds.update({a:p['recovery_joint_bounds'] for a,p in compiled.items() if 'recovery_joint_bounds' in p})
        for a,p in compiled.items():
            p['time_s'] = warp(p['time_s'])
            p['task_offset_s'] = float(warp(p.get('task_offset_s',0.)))
            for segment in (p.get('user_trajectory'),p.get('approach')):
                if segment is not None:
                    segment['time_s'] = warp(segment['time_s'])
                    for event in segment['gripper_events']:
                        event['time_s'] = float(warp(event['time_s']) if event.get('hold_s') else
                            (segment['time_s'][0] if event['s']==0 else np.interp(event['s'],segment['s'],segment['time_s'])))
                        if 'hold_end_time_s' in event:
                            event['hold_end_time_s'] = float(warp(event['hold_end_time_s']))
            user=p.get('user_trajectory',p)
            for e in p['gripper_events']:
                e['time_s'] = float(warp(e['time_s']) if e.get('hold_s') else
                    (user['time_s'][0] if e['s']==0 else np.interp(e['s'],user['s'],user['time_s'])))
                if 'hold_end_time_s' in e:
                    e['hold_end_time_s'] = float(warp(e['hold_end_time_s']))
            p['curve'] = joint_curve(p['time_s'],p['joints_deg'],p.get('stop_indices',()),bounds[a])
        duration = float(max(p['time_s'][-1] for p in compiled.values()))
        scale=duration/original_duration
        if duration > 180:
            raise ValueError(f'预计执行 {duration:.1f} 秒（原始 {duration/scale:.1f} 秒，关节速度/加速度/jerk 限制使时长放大 {scale:.1f} 倍），超过单段 180 秒；请缩短轨迹或调整姿态。这是运动时长，不是检查耗时。')
        # Check the actual joint interpolant, not only the Cartesian input knots.
        grid = np.unique(np.concatenate([np.linspace(0, duration, int(duration/.04)+2)] +
                                         [p['time_s'] for p in compiled.values()]))
        if len(grid) > 10000:
            raise ValueError('Preview exceeds sample budget')
        self._check_joint_samples(compiled, start, grid, initial_bounds, report)
        self.validate_start(start)
        notes = [f'位置容差 {position_limit:g} mm；固定/指定姿态容差 {rotation_limit:g}°。',
                 'Local 50 Hz execution; ordinary position mode.',
                 ('Collision checks cover the robot and calibrated table, not arbitrary scene objects.' if self.table_checks else
                  '桌面净空与桌面碰撞检测已关闭：只检查关节限位、自碰撞和双臂互碰，路径可以压到桌面。'),
                 'Gripper event times are command dispatch times; physical release latency is not calibrated.']
        for a,p in compiled.items():
            if 'recovery_joint_bounds' in p:
                notes.insert(0,f'{a}: 接近段从实测关节位置平滑返回模型限位内，任务轨迹仍使用原有限位。')
            if p.get('orientation_mode')=='free':
                notes.insert(0,f'{"左臂" if a=="left" else "右臂"}姿态自由：跟随 XYZ 路径，RPY 由 IK 连续求解；预览显示求解后的朝向。')
        for a, lifts in floor_lifts.items():
            notes.insert(0, f'{"左臂" if a == "left" else "右臂"}路径贴近桌面：已把 {len(lifts)} 个采样点最多抬高 {max(lifts):.1f} mm 以满足桌面净空。')
        if scale > 1.03:
            notes.append(f'{limits["label"]}档：局部调整速度；接近段总时长 ×{scales["approach"]:.2f}，绘制段总时长 ×{scales["task"]:.2f}。')
        return {'duration_s': duration, 'arms': compiled, 'retiming_factor': scale, 'notes': notes, 'pose_tolerance': tolerance,
                'initial_joint_bounds':initial_bounds,
                'motion_limits':limits,'phase_retiming':scales,
                'approach_duration_s': max(p.get('task_offset_s',0.) for p in compiled.values())}

    def _check_joint_samples(self, compiled, start, grid, initial_bounds, report):
        """Geometry-check the interpolated joints of both arms at every grid time.

        A failure at or after the common task start is a TaskError: both arms are then on the
        user's own trajectories, which no approach route changes.
        """
        offset = max(p.get('task_offset_s', 0.) for p in compiled.values())
        joints = {a: (compiled[a]['curve'](np.minimum(grid, compiled[a]['time_s'][-1])) if a in compiled
                      else np.tile(np.asarray(start[a]['joints_deg'], dtype=float), (len(grid), 1))) for a in ARMS}
        kwargs = {'joint_bounds': initial_bounds} if initial_bounds else {}
        report(stage='collision', arm=None, done=0, total=len(grid))
        if self.planner_pool is not None:
            failure = self.planner_pool.check(joints['left'], joints['right'], self.table_checks, initial_bounds,
                                              progress=lambda done: report(stage='collision', arm=None, done=done, total=len(grid)))
            if failure is not None:
                index, message = failure
                raise (TaskError if grid[index] >= offset else ValueError)(message)
        else:
            for i, t in enumerate(grid):
                if i % 20 == 0:
                    report(stage='collision', arm=None, done=i, total=len(grid))
                try:
                    self._geometry_check(joints['left'][i], joints['right'][i], **kwargs)
                except ValueError as exc:
                    if t >= offset:
                        raise TaskError(str(exc)) from exc
                    raise
        report(stage='collision', arm=None, done=len(grid), total=len(grid))
