"""One metric trajectory representation for the GUI and the function API."""
from __future__ import annotations

import ast
import operator
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

FUNCTIONS = {'sin': np.sin, 'cos': np.cos, 'sqrt': np.sqrt, 'abs': np.abs,
             'minimum': np.minimum, 'maximum': np.maximum}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.Div: operator.truediv, ast.Pow: operator.pow}
MAX_SPEED = 5.0
# Cartesian acceleration envelope that ramps a path from and to rest; tosses raise it per draft.
DEFAULT_ACCELERATION = .1
MAX_ACCELERATION = 50.
MAX_ANGULAR_SPEED = 30.0
MAX_ANGULAR_SPEED_LIMIT = 720.0   # a draft may raise its orientation rate up to this (sidearm tosses turn with the base joint)
SAMPLES = 201
CONTROL_PERIOD_S = .02     # the robot streams references at 50 Hz; denser samples only encode IK jitter
MAX_SAMPLES = 2001
POSITION_STEP_M = .008
ORIENTATION_STEP_DEG = 2.5


def expression(source, s, parameters=None):
    """Interpret bounded math AST, never execute arbitrary Python/JavaScript."""
    source = str(source)
    if len(source) > 512:
        raise ValueError('Expression is too long')
    names = {'s': np.asarray(s), 'pi': np.pi}
    for k, v in (parameters or {}).items():
        if not k.isidentifier() or k.startswith('_') or k in names or k in FUNCTIONS:
            raise ValueError('Invalid parameter name')
        if not isinstance(v, (int, float)) or not np.isfinite(v) or abs(v) > 1000:
            raise ValueError('Invalid parameter value')
        names[k] = float(v)
    try:
        root = ast.parse(source, mode='eval')
        if len(list(ast.walk(root))) > 96:
            raise ValueError('Expression is too complex')

        def visit(n):
            if isinstance(n, ast.Constant) and type(n.value) in (int, float):
                if not np.isfinite(n.value) or abs(n.value) > 1000:
                    raise ValueError('Constant outside bounds')
                return float(n.value)
            if isinstance(n, ast.Name) and n.id in names:
                return names[n.id]
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                return visit(n.operand) * (-1 if isinstance(n.op, ast.USub) else 1)
            if isinstance(n, ast.BinOp) and type(n.op) in OPS:
                if isinstance(n.op, ast.Pow) and (not isinstance(n.right, ast.Constant) or
                                                  type(n.right.value) not in (int, float) or abs(n.right.value) > 8):
                    raise ValueError('Exponent must be a constant between -8 and 8')
                return OPS[type(n.op)](visit(n.left), visit(n.right))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCTIONS and not n.keywords:
                arity = 2 if n.func.id in ('minimum', 'maximum') else 1
                if len(n.args) != arity:
                    raise ValueError(f'{n.func.id} requires exactly {arity} arguments')
                return FUNCTIONS[n.func.id](*[visit(a) for a in n.args])
            raise ValueError('Only bounded arithmetic, s, parameters and math functions are allowed')

        with np.errstate(all='raise'):
            result = np.broadcast_to(np.asarray(visit(root.body), dtype=float), np.shape(s)).copy()
        if not np.isfinite(result).all() or np.max(np.abs(result)) > 1e4:
            raise ValueError('Expression produced nonfinite or excessive values')
        return result
    except (SyntaxError, TypeError, ArithmeticError, OverflowError) as exc:
        raise ValueError(f'Invalid expression: {exc}') from exc


def keyframes(points, columns):
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != columns or not 2 <= len(a) <= 128 or not np.isfinite(a).all():
        raise ValueError('Expected 2 to 128 finite keyframes')
    if a[0, 0] != 0 or a[-1, 0] != 1 or np.any(np.diff(a[:, 0]) <= 0):
        raise ValueError('Keyframes must increase strictly from progress 0 to 1')
    return a


def time_samples(xyz, rotations, requested, accel_m_s2=DEFAULT_ACCELERATION, angular_speed_deg_s=MAX_ANGULAR_SPEED):
    ds = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    distance = np.r_[0, np.cumsum(ds)]
    angles = (rotations[:-1].inv() * rotations[1:]).magnitude()
    if np.max(angles) > np.deg2rad(5):
        raise ValueError('Orientation changes too quickly; simplify or split the segment')
    requested_duration = float(np.sum(2 * ds / (requested[:-1] + requested[1:])))
    # An acceleration envelope gives rest endpoints; angular motion is timed independently.
    speed_bound = np.minimum(requested, np.sqrt(2*accel_m_s2 * np.minimum(distance, distance[-1] - distance)))
    denominators = speed_bound[:-1] + speed_bound[1:]
    dt_position = np.divide(2 * ds, denominators, out=np.zeros_like(ds), where=denominators > 1e-12)
    dt_angle = angles / np.deg2rad(angular_speed_deg_s)
    dt = np.maximum(np.maximum(dt_position, dt_angle), .001)
    times = np.r_[0., np.cumsum(dt)]
    if times[-1] > 180:
        raise ValueError('Motion exceeds 180 seconds; split into shorter segments')
    return times, float(distance[-1]), requested_duration


def control_period_mask(s, times, pinned_s):
    """Keep the first and last samples, every pinned progress value, and otherwise only samples at least one
    control period after the previously kept one. A 1 m/s stroke refined to millimetre steps would put samples
    a millisecond apart; the joint spline through them turns IK jitter into enormous accelerations and jerk."""
    pinned = np.isin(s, pinned_s)
    keep = pinned.copy()
    keep[0] = keep[-1] = True
    last = 0
    for i in range(1, len(s)-1):
        if pinned[i]:
            # A pinned pose wins over an ordinary sample crowding it from before.
            if last > 0 and not pinned[last] and times[i]-times[last] < CONTROL_PERIOD_S-1e-12:
                keep[last] = False
            last = i
        elif times[i]-times[last] >= CONTROL_PERIOD_S-1e-12:
            keep[i] = True
            last = i
    return keep


def gripper_event(raw):
    """Validate one gripper event; without ``opening_mm`` it is a pure pause that only needs ``hold_s``."""
    event = {'s': float(raw['s'])}
    if not np.isfinite(event['s']) or not 0 <= event['s'] <= 1:
        raise ValueError('Invalid gripper event')
    if raw.get('opening_mm') is not None:
        event['opening_mm'] = float(raw['opening_mm'])
        if not np.isfinite(event['opening_mm']) or not 0 <= event['opening_mm'] <= 70:
            raise ValueError('Invalid gripper event')
    if raw.get('hold_s') is not None:
        event['hold_s'] = float(raw['hold_s'])
        if not np.isfinite(event['hold_s']) or not 0 <= event['hold_s'] <= 3:
            raise ValueError('Gripper hold_s must be between 0 and 3 seconds')
    if 'opening_mm' not in event and not event.get('hold_s'):
        raise ValueError('A pause event without opening_mm requires a positive hold_s')
    if raw.get('lead_s') is not None:
        event['lead_s'] = float(raw['lead_s'])
        if not np.isfinite(event['lead_s']) or not 0 <= event['lead_s'] <= .5:
            raise ValueError('Gripper lead_s must be between 0 and 0.5 seconds')
        if 'opening_mm' not in event or event.get('hold_s'):
            raise ValueError('lead_s applies to an opening command without a hold')
    if raw.get('ramp_s') is not None:
        # The jaws are commanded through intermediate openings over ``ramp_s`` instead of one jump.
        event['ramp_s'] = float(raw['ramp_s'])
        if not np.isfinite(event['ramp_s']) or not 0 <= event['ramp_s'] <= 2:
            raise ValueError('Gripper ramp_s must be between 0 and 2 seconds')
        if 'opening_mm' not in event:
            raise ValueError('ramp_s applies to an opening command')
    if raw.get('effort') is not None:
        # Torque limit for this command in 0.001 N*m (PiPER range 0-5000): a paper cup needs less than a block.
        event['effort'] = int(raw['effort'])
        if 'opening_mm' not in event or not 1 <= event['effort'] <= 5000:
            raise ValueError('Gripper effort applies to an opening command and must be 1-5000')
    if raw.get('on_measured'):
        # Dispatch against the measured progress along the reference instead of the schedule:
        # a fast swing lags the 50 Hz reference, and the release must follow the real arm.
        if 'opening_mm' not in event or event.get('hold_s'):
            raise ValueError('on_measured applies to an opening command without a hold')
        event['on_measured'] = True
    if raw.get('wait_for_arrival'):
        if not event.get('hold_s'):
            raise ValueError('Arrival-gated gripper events require a hold')
        event['wait_for_arrival'] = True
    if raw.get('verify'):
        if 'opening_mm' not in event:
            raise ValueError('Pause events cannot verify the gripper')
        # Only a grip is verified, never an *opening*: the jaws report 62-69 mm when fully open depending
        # on what is between the fingers, and an opening gate stopped a carry in the middle because a
        # release read 62.8 mm instead of the commanded 70.
        if raw['verify'] != 'holding' or not event.get('hold_s') or not event.get('wait_for_arrival'):
            raise ValueError('Gripper verification requires an arrival-gated hold on a closing event')
        event['verify'] = raw['verify']
    return event


def refine_samples(s, position_at, rotation_at, position_step=None, orientation_step_deg=None):
    """Bisect sample intervals until consecutive positions and rotations stay within the given steps."""
    for _ in range(8):
        split = np.zeros(len(s) - 1, dtype=bool)
        if position_step is not None:
            xyz = position_at(s)
            split |= np.linalg.norm(np.diff(xyz, axis=0), axis=1) > position_step
        if orientation_step_deg is not None:
            rotations = rotation_at(s)
            split |= (rotations[:-1].inv() * rotations[1:]).magnitude() > np.deg2rad(orientation_step_deg)
        if not split.any():
            break
        s = np.unique(np.r_[s, (s[:-1][split] + s[1:][split]) / 2])
        if len(s) > MAX_SAMPLES:
            raise ValueError('轨迹采样点过多，请缩短这一笔或减少姿态变化')
    return s


def compile_arm(spec, state):
    approach = spec.get('approach', {})
    approach_speed = float(approach.get('speed_m_s', .8))
    clearance = float(approach.get('clearance_m', .12))
    if not .001 <= approach_speed <= 1. or not .02 <= clearance <= .30:
        raise ValueError('Approach speed must be 0.001–1.0 m/s and clearance 0.02–0.30 m')
    raw_events = spec.get('gripper_events', [])
    if not isinstance(raw_events, list) or len(raw_events) > 16:
        raise ValueError('At most 16 gripper events are supported')
    events = [gripper_event(raw) for raw in raw_events]
    events.sort(key=lambda e:e['s'])
    if len({e['s'] for e in events}) != len(events):
        raise ValueError('Only one gripper event is allowed at each progress value')
    s = np.unique(np.r_[np.linspace(0, 1, SAMPLES), [e['s'] for e in events if e.get('hold_s')]])
    path = spec.get('path', {})
    parameters = dict(zip(('x0', 'y0', 'z0'), state['xyz']))
    parameters.update(spec.get('parameters', {}))
    position_step = None
    if path.get('mode') == 'waypoints':
        p = np.asarray(path.get('points'), dtype=float)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError('每个路径点都需要完整、有效的 X、Y、Z 数值')
        if len(p) < 2:
            raise ValueError(f'当前只有 {len(p)} 个路径点；请添加终点，或取消这只机械臂的执行勾选')
        if len(p) > 128:
            raise ValueError(f'当前有 {len(p)} 个路径点；每段最多支持 128 个，请分段执行')
        knots = np.linspace(0, 1, len(p))
        position_at = PchipInterpolator(knots, p)
        # Sample every waypoint path finely enough for the preview. The fixed SAMPLES points are only dense
        # enough for a short stroke: a 1.7 m one through six far-apart control points leaves 22 mm between
        # neighbours and would be refused outright, even though bisecting it costs ~215 points.
        position_step = POSITION_STEP_M
        if any(e.get('hold_s') for e in events):
            # Dwell trajectories hold exact poses, so the samples must follow every corner closely.
            s = np.unique(np.r_[s, knots])
    elif path.get('mode') == 'function':
        def position_at(progress):
            return np.stack([expression(path[k], progress, parameters) for k in ('x', 'y', 'z')], axis=1)
    else:
        raise ValueError('Path mode must be waypoints or function')
    orient = spec.get('orientation', {'mode': 'hold'})
    mode = orient.get('mode', 'hold')
    orientation_step = None
    if mode in ('hold', 'free'):
        def rotation_at(progress):
            return Rotation.from_euler('xyz', np.tile(state['rpy_deg'], (len(progress), 1)), degrees=True)
    elif mode == 'keyframes':
        k = keyframes(orient['points'], 4)
        rotation_at = Slerp(k[:, 0], Rotation.from_euler('xyz', k[:, 1:], degrees=True))
        # Keyframes may turn in place; sample finely enough for the 5 degree timing check.
        orientation_step = ORIENTATION_STEP_DEG
    elif mode == 'function':
        def rotation_at(progress):
            return Rotation.from_euler('xyz', np.stack([expression(orient[k], progress, parameters)
                                                       for k in ('roll', 'pitch', 'yaw')], axis=1), degrees=True)
    else:
        raise ValueError('Unknown orientation mode')
    s = refine_samples(s, position_at, rotation_at, position_step, orientation_step)
    xyz = position_at(s)
    if np.any(np.abs(xyz) > 2):
        raise ValueError('Path exceeds the prototype coordinate bounds (2 metres)')
    gap = float(np.max(np.linalg.norm(np.diff(xyz, axis=0), axis=1)))
    if gap > .01:
        # Only a function path can still get here: waypoint paths are bisected until they fit.
        raise ValueError(f'路径在预览分辨率下变化太快：相邻采样点最大 {gap*1000:.0f} mm，上限 10 mm；'
                         '请缩短这一段或分段执行')
    rotations = rotation_at(s)
    speed = spec.get('speed', '.03')
    if isinstance(speed, list):
        k = keyframes(speed, 2)
        requested = np.interp(s, k[:, 0], k[:, 1])
    else:
        requested = expression(speed, s, parameters)
    if np.any(requested < .001) or np.any(requested > MAX_SPEED):
        raise ValueError(f'Speed must be between 0.001 and {MAX_SPEED} m/s; endpoints are ramped to rest')
    accel = float(spec.get('accel_m_s2', DEFAULT_ACCELERATION))
    if not np.isfinite(accel) or not .01 <= accel <= MAX_ACCELERATION:
        raise ValueError(f'accel_m_s2 must be between 0.01 and {MAX_ACCELERATION} m/s²')
    angular = float(spec.get('angular_speed_deg_s', MAX_ANGULAR_SPEED))
    if not np.isfinite(angular) or not 1. <= angular <= MAX_ANGULAR_SPEED_LIMIT:
        raise ValueError(f'angular_speed_deg_s must be between 1 and {MAX_ANGULAR_SPEED_LIMIT:g} deg/s')
    times, path_length, requested_duration = time_samples(xyz, rotations, requested, accel, angular)
    keep = control_period_mask(s, times, [e['s'] for e in events if e.get('hold_s')])
    s, xyz, requested, times = s[keep], xyz[keep], requested[keep], times[keep]
    rotations = rotations[np.flatnonzero(keep)]
    matrices = rotations.as_matrix()
    stops=[]
    for e in events:
        i=int(np.searchsorted(s,e['s'],side='left'))
        e['time_s'] = float(np.interp(e['s'],s,times))
        dwell=e.get('hold_s',0.)
        if dwell:
            times[i+1:] += dwell
            times=np.insert(times,i+1,times[i]+dwell)
            s=np.insert(s,i+1,s[i]);xyz=np.insert(xyz,i+1,xyz[i],axis=0)
            matrices=np.insert(matrices,i+1,matrices[i],axis=0)
            requested=np.insert(requested,i+1,0.);requested[i]=0.
            stops.extend([i,i+1]);e['hold_end_time_s']=float(times[i+1])
    hold=float(spec.get('start_hold_s',0.))
    if not np.isfinite(hold) or not 0<=hold<=60:
        raise ValueError('start_hold_s must be between 0 and 60 seconds')
    if hold:
        times=np.r_[0.,times+hold];s=np.r_[0.,s];xyz=np.vstack([xyz[:1],xyz])
        matrices=np.concatenate([matrices[:1],matrices])
        requested=np.r_[0.,requested];stops=[1]+[i+1 for i in stops]
        for event in events:
            if event['s']>0:event['time_s']+=hold
            if 'hold_end_time_s' in event:event['hold_end_time_s']+=hold
    return {'s': s, 'xyz': xyz, 'rotation_matrices': matrices, 'orientation_mode':mode, 'time_s': times,
            'requested_speed_m_s': requested, 'requested_duration_s': requested_duration,
            'path_length_m': path_length, 'gripper_events': events,
            'stop_indices':stops,'start_hold_s':hold,'accel_m_s2':accel,'angular_speed_deg_s':angular,
            'approach_config': {'speed_m_s':approach_speed,'clearance_m':clearance},
            'air_track': bool(spec.get('air_track', False))}


def compile_arms(specs, state):
    result = {}
    for arm, spec in specs.items():
        try:
            result[arm] = compile_arm(spec, state[arm])
        except ValueError as exc:
            name = {'left':'左臂', 'right':'右臂'}.get(arm, arm)
            raise ValueError(f'{name}：{exc}') from exc
    return result
