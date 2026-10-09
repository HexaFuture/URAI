"""Preview all placement phases, dispatch each only after measured completion.

Planning uses predicted boundary states. The live robot must remain at the
original preview start throughout planning; execution validates each real boundary.
"""
import copy
import numpy as np
from scipy.spatial.transform import Rotation
from .trajectory import compile_arms
from .backend import Cancelled


def end_state(backend, plan, start):
    result=copy.deepcopy(start)
    for arm,p in plan['arms'].items():
        state=result[arm]
        if 'joints_deg' in p:
            q=np.asarray(p['joints_deg'][-1]);state['joints_deg']=q.tolist()
            pose=backend.models[arm].fk_tcp_world(q)
            state['xyz']=pose[:3,3].tolist()
            state['rpy_deg']=Rotation.from_matrix(pose[:3,:3]).as_euler('xyz',degrees=True).tolist()
        else:
            state['xyz']=np.asarray(p['xyz'][-1]).tolist()
            state['rpy_deg']=Rotation.from_matrix(p['rotation_matrices'][-1]).as_euler('xyz',degrees=True).tolist()
            if backend.mode!='simulation':raise ValueError('Real placement plan requires joint endpoints')
            state['joints_deg']=(np.asarray(state['xyz'])*100).tolist()
        for event in sorted(p['gripper_events'],key=lambda e:e['time_s']):
            state['gripper_mm']=float(event['opening_mm'])
    return result


def display_arms(phases, start):
    """Joined display traces only; these are never dispatched as a combined motion."""
    result={};offset=0.;state=copy.deepcopy(start)
    for phase in phases:
        duration=float(phase['duration_s'])
        for arm in start:
            if arm not in result:
                result[arm]={'xyz':[],'rotation_matrices':[],'time_s':[],
                             'requested_speed_m_s':[],'gripper_events':[],'path_length_m':0.}
            out=result[arm];p=phase['arms'].get(arm)
            if p:
                times=np.asarray(p['time_s']);xyz=np.asarray(p['xyz']);rot=np.asarray(p['rotation_matrices']);speed=np.asarray(p['requested_speed_m_s'])
                out['path_length_m']+=p['path_length_m']
                for event in p['gripper_events']:
                    event=copy.deepcopy(event);event['time_s']+=offset
                    if 'hold_end_time_s' in event:event['hold_end_time_s']+=offset
                    out['gripper_events'].append(event)
            else:
                times=np.array([0.,duration]);xyz=np.tile(state[arm]['xyz'],(2,1))
                rot=np.tile(Rotation.from_euler('xyz',state[arm]['rpy_deg'],degrees=True).as_matrix(),(2,1,1));speed=np.zeros(2)
            for t,x,r,v in zip(times,xyz,rot,speed):
                if out['time_s'] and t+offset<=out['time_s'][-1]+1e-10:continue
                out['time_s'].append(float(t+offset));out['xyz'].append(x);out['rotation_matrices'].append(r);out['requested_speed_m_s'].append(v)
        state=phase['end_state'];offset+=duration
    for p in result.values():
        for k in ('xyz','rotation_matrices','time_s','requested_speed_m_s'):p[k]=np.asarray(p[k])
        p['s']=p['time_s']/offset;p['orientation_mode']='keyframes';p['start_hold_s']=0.;p['stop_indices']=[]
    return result


def plan(backend, draft, start, metadata, tolerance, profile, progress=None):
    holder=metadata['holder'];worker=metadata['worker']
    # Future-phase planning must not mistake its virtual start for live feedback.
    # Every validation still checks that the real robot has not left the original start.
    planner=copy.copy(backend)
    planner.validate_start=lambda _predicted: backend.validate_start(start)
    backend.validate_start(start)
    phases=[];state=copy.deepcopy(start)
    def append(label, value, spec):
        nonlocal state
        backend.validate_start(start)
        value.update(label=label,start=copy.deepcopy(state),input_draft=spec)
        state=end_state(backend,value,state);value['end_state']=copy.deepcopy(state)
        phases.append(value)
    options={'pose_tolerance':tolerance,'progress':progress,'motion_profile':profile}
    retreat={'arms':{worker:copy.deepcopy(draft['arms'][worker])}}
    append('worker_retract',planner.plan(compile_arms(retreat['arms'],state),state,**options),retreat)
    append('worker_home',planner.plan_home(state,[worker],open_grippers=False,speed_m_s=.2,**options),{'home':{'arms':[worker]}})
    placement={'arms':{holder:copy.deepcopy(draft['arms'][holder])}}
    placement['arms'][holder]['start_hold_s']=0.
    append('holder_place',planner.plan(compile_arms(placement['arms'],state),state,**options),placement)
    duration=sum(float(p['duration_s']) for p in phases)
    if duration>180:raise ValueError('完整放置顺序超过 180 秒，请调整路径')
    return {'duration_s':duration,'arms':display_arms(phases,start),'sequence':phases,
            'sequence_worker':worker,'sequence_holder':holder,
            'phase_order':[p['label'] for p in phases],
            'approach_duration_s':sum(p.get('approach_duration_s',0.) for p in phases),
            'retiming_factor':1.,'pose_tolerance':tolerance,'motion_limits':phases[0]['motion_limits'],
            'notes':['拧盖臂先松爪退开并完整回收纳位；实测确认到位后，持瓶臂才开始放置。']}


def execute(backend, plan, cancel, progress, phase_changed=None):
    elapsed=0.;results=[];total=plan['duration_s']
    for phase in plan['sequence']:
        if cancel.is_set():raise Cancelled()
        backend.validate_start(phase['start'])
        if phase_changed:phase_changed(phase['label'])
        def report(fraction,seconds):progress((elapsed+fraction*phase['duration_s'])/total,elapsed+seconds)
        result=backend.execute(phase,phase['start'],cancel,report)
        if cancel.is_set():raise Cancelled()
        if not result.get('completed'):raise ValueError('前序动作未完成，已停止后续放置：'+phase['label'])
        actual=backend.state()
        for arm in phase['end_state']:
            expected=phase['end_state'][arm];now=actual[arm]
            if np.max(np.abs(np.asarray(now['joints_deg'])-expected['joints_deg']))>.5:
                raise ValueError('前序动作关节尚未到位，已保持持瓶：'+phase['label'])
            pe=np.linalg.norm(np.asarray(now['xyz'])-expected['xyz'])
            re=(Rotation.from_euler('xyz',now['rpy_deg'],degrees=True).inv()*Rotation.from_euler('xyz',expected['rpy_deg'],degrees=True)).magnitude()
            if pe>.003 or re>np.deg2rad(3):raise ValueError('前序动作位姿尚未到位，已保持持瓶：'+phase['label'])
            if expected['gripper_mm']>=60 and now['gripper_mm']<60:
                raise ValueError('机械臂夹爪尚未松开，已停止后续动作')
        results.append({k:v for k,v in result.items() if k!='trace'})
        elapsed+=phase['duration_s']
    return {'completed':True,'phase_results':results,'elapsed_s':elapsed}
