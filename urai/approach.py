"""Explicit low-speed approach segments before the user's own trajectory."""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .trajectory import compile_arm


def needs_approach(task, state):
    angle = (Rotation.from_euler('xyz', state['rpy_deg'], degrees=True).inv()*
             Rotation.from_matrix(task['rotation_matrices'][0])).magnitude()
    return np.linalg.norm(task['xyz'][0]-state['xyz']) > .0005 or (task.get('orientation_mode')!='free' and angle > np.deg2rad(.1))


def with_approaches(compiled, start, routes, prepared=None):
    approaches = {}
    for arm, route in routes.items():
        task = compiled[arm]; state = start[arm]
        if route == 'joint':
            approaches[arm] = copy.deepcopy(prepared[arm])
            continue
        p0 = np.asarray(state['xyz']); goal = task['xyz'][0]
        config = task.get('approach_config', {})
        if route == 'direct':
            points = [p0, goal]
        elif route == 'lift':
            z = max(p0[2], goal[2])+config.get('clearance_m', .12)
            points = [p0, [p0[0], p0[1], z], [goal[0], goal[1], z], goal]
        else:
            raise ValueError('Unknown approach route')
        target_rpy = Rotation.from_matrix(task['rotation_matrices'][0]).as_euler('xyz', degrees=True)
        orient = {'mode':'free'} if task.get('orientation_mode')=='free' else {'mode':'keyframes',
            'points': [[0, *state['rpy_deg']], [1, *target_rpy]]}
        approaches[arm] = compile_arm({'path': {'mode':'waypoints', 'points':np.asarray(points).tolist()},
            'speed': config.get('speed_m_s', .02), 'orientation':orient}, state)
        approaches[arm]['route'] = route
    offset = max((p['time_s'][-1] for p in approaches.values()), default=0.)
    if offset == 0:
        return copy.deepcopy(compiled)
    result = {}
    for arm, original in compiled.items():
        task = copy.deepcopy(original)
        task['time_s'] += offset
        for event in task['gripper_events']:
            event['time_s'] += offset
            if 'hold_end_time_s' in event:
                event['hold_end_time_s'] += offset
        approach = approaches.get(arm)
        stops = []
        if approach:
            # When an arm arrives early, a repeated pose creates an explicit hold until both are ready.
            count = len(approach['s']) if approach['time_s'][-1] < offset-1e-9 else len(approach['s'])-1
            prefix = {k: approach[k][:count] for k in ('xyz','rotation_matrices','time_s','requested_speed_m_s')}
            if count == len(approach['s']):
                stops.append(count-1)
        else:
            prefix = {'xyz': np.asarray([start[arm]['xyz']]),
                      'rotation_matrices': Rotation.from_euler('xyz',[start[arm]['rpy_deg']],degrees=True).as_matrix(),
                      'time_s': np.array([0.]), 'requested_speed_m_s':np.array([0.])}
            count = 1
        stops.append(count)
        stops.extend(count+i for i in task.get('stop_indices',[]))
        combined = copy.deepcopy(task)
        for key in ('xyz','rotation_matrices','time_s','requested_speed_m_s'):
            combined[key] = np.concatenate([prefix[key], task[key]])
        combined.update(s=np.linspace(0,1,len(combined['xyz'])), user_trajectory=task,
                        task_offset_s=float(offset), stop_indices=stops, approach=approach)
        if approach and 'source_joints_deg' in approach:
            combined['approach_joints_deg'] = approach['source_joints_deg'].copy()
        combined['path_length_m'] += approach['path_length_m'] if approach else 0.
        result[arm] = combined
    return result
