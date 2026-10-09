"""Upright bottle preparation and world-axis cap turning; ordinary preview/executor required."""
import copy
import uuid
import numpy as np
from scipy.spatial.transform import Rotation
from .registry import register,number,integer
from .motion import side_grasp_rotation,open_event,close_event
from .bottle_cap import _point,_probe,_spec,_compensation,_rotation,planning_session

def cap_route(center,direction,compensation,angle,cycles,stand_off,effort,pitch=0,lift=0):
    center=np.asarray(center,float);base=side_grasp_rotation(np.asarray(direction,float),pitch)
    def tcp(r):return center+r[:,1]*compensation
    points=[tcp(base)-base[:,2]*stand_off,tcp(base)];rots=[base,base];events=[(0,'open')]
    for cycle in range(cycles):
        events.append((len(points)-1,'close'))
        for a in np.linspace(angle/6,angle,6):
            r=Rotation.from_rotvec([0,0,np.deg2rad(a)]).as_matrix()@base
            points.append(tcp(r));rots.append(r)
        if lift and cycle==cycles-1:
            points.append(tcp(r)+[0,0,lift]);rots.append(r)
            break
        events.append((len(points)-1,'open'))
        for a in np.linspace(angle*5/6,0,6):
            r=Rotation.from_rotvec([0,0,np.deg2rad(a)]).as_matrix()@base
            points.append(tcp(r));rots.append(r)
    if not lift:
        points.append(tcp(base)-base[:,2]*stand_off);rots.append(base)
    n=len(points)-1
    return points,rots,[open_event(i/n,hold_s=.6) if k=='open' else close_event(i/n,effort=effort) for i,k in events]

@register(name='bottle_upright_prepare',label='装水瓶① · 竖直持瓶',group='双臂拧瓶盖',stroke='point',
 summary='点瓶盖；侧抓瓶身并保持竖直移至双臂操作区。',hint='瓶子须竖直，抓取点需与瓶身对应。',
 inputs={'body_offset_mm':number('盖到抓点',120,220,140,5,' mm'),
 'body_inset_mm':number('夹入深度',0,20,15,1,' mm'),
 'grip_effort':integer('持瓶力矩',100,2000,1000),
 'speed_m_s':number('速度',.01,.15,.08,.01,' m/s')})
def prepare(ctx,ink,body_offset_mm,body_inset_mm,grip_effort,speed_m_s,body_reference=False):
    cap=_point(ctx,ink)
    if body_reference:cap=cap+[0,0,body_offset_mm/1000]
    body=cap-[0,0,body_offset_mm/1000];ctx.speed_m_s=speed_m_s
    if body[2]<ctx.table_z+.025:raise ValueError('瓶身抓点过低')
    d=body[:2]-ctx.base_xy;d/=np.linalg.norm(d);R=side_grasp_rotation(d)
    offset=np.array([0,_compensation(ctx),body_inset_mm/1000]);grip=body+R@offset
    other=ctx.other_arm();middle=(ctx.base_xy+other.base_xy)/2
    goal=np.r_[middle+[.35,0],grip[2]+.020]
    points=[grip-R[:,2]*.0525+[0,0,.025],grip,grip+[0,0,.06],goal+[0,0,.04],goal]
    rots=[R]*len(points);_probe(ctx,points,rots)
    cap_local=R.T@(cap-grip);cap_after=goal+R@cap_local
    direction=cap_after[:2]-other.base_xy;direction/=np.linalg.norm(direction)
    trial,rr,_=cap_route(cap_after-[0,0,.010],direction,_compensation(other),60,1,.07,1500);_probe(other,trial,rr)
    spec=_spec(points,rots,[open_event(0),close_event(.25,effort=grip_effort)],ctx,15)
    spec['bottle_cap']={'stage':'prepare','id':uuid.uuid4().hex,'holder':ctx.arm,'mode':'upright',
      'axis':[0,0,1],'axis_local':(R.T@np.array([0,0,1.])).tolist(),'cap_local':cap_local.tolist(),
      'bottle_height_m':float(cap[2]-ctx.table_z),'body_tool_offset':offset.tolist(),
      'upright_rotations':[R.tolist()],'upright_positions':[goal.tolist()]}
    return {'arm':spec,'cap_xyz':cap_after.tolist(),'summary':'竖直持瓶；完成后重新观察瓶盖'}

@register(name='bottle_upright_twist',label='装水瓶② · 侧夹竖轴拧盖',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='竖直持瓶，侧夹瓶盖绕世界竖直轴拧转。',hint='①完成后重新点瓶盖中心；逐轮检查开盖情况。',
 inputs={'turn_deg':number('每轮角度',10,60,60,5,' °'),'cycles':integer('轮数',1,6,1),
 'grip_effort':integer('拧盖力矩',100,1500,1500),'cap_inset_mm':number('盖侧夹持深度',3,15,10,1,' mm'),
 'pitch_deg':number('接近俯角',0,40,0,5,' °'),'lift_mm':number('末轮夹持上提',0,5,0,1,' mm'),'approach_offset_deg':number('侧面接近偏角',-60,60,0,5,' °'),'speed_m_s':number('速度',.01,.15,.06,.01,' m/s')})
def twist(ctx,ink,turn_deg,cycles,grip_effort,cap_inset_mm,approach_offset_deg,speed_m_s,pitch_deg,lift_mm):
    session,relocalized=planning_session(ctx)
    if session.get('mode')!='upright':raise ValueError('需要竖直持瓶记录')
    worker=ctx.other_arm() if ctx.arm==session['holder'] else ctx;worker.speed_m_s=speed_m_s
    held=ctx.current[session['holder']];R=_rotation(held['rpy_deg']);axis=R@np.asarray(session['axis_local'])
    if np.linalg.norm(axis-[0,0,1])>.035:raise ValueError('瓶子偏离竖直，需要重新定位')
    expected=np.asarray(held['xyz'])+R@np.asarray(session['cap_local']);surface=_point(worker,ink)
    if np.linalg.norm(surface-expected)>.04:raise ValueError('瓶盖观测偏离预测超过4cm')
    center=surface-[0,0,cap_inset_mm/1000];direction=center[:2]-worker.base_xy;direction/=np.linalg.norm(direction)
    a=np.deg2rad(approach_offset_deg);direction=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@direction
    pts,rots,events=cap_route(center,direction,_compensation(worker),turn_deg,cycles,.07,grip_effort,pitch_deg,lift_mm/1000)
    now=ctx.current[worker.arm];rr=_rotation(now['rpy_deg']);nn=len(pts)-1
    pts=[np.asarray(now['xyz']),np.asarray(now['xyz'])+[0,0,.06]]+pts
    rots=[rr,rr]+rots
    for e in events:e['s']=(e['s']*nn+2)/(nn+2)
    _probe(worker,pts,rots)
    if relocalized:ctx.service.bottle_cap=copy.deepcopy(session)
    spec=_spec(pts,rots,events,worker,10)
    spec['bottle_cap']={'stage':'twist','id':session['id'],'holder':session['holder'],'worker':worker.arm,'cycles':cycles}
    return {'arms':{worker.arm:spec},'summary':'侧夹瓶盖竖轴拧转，持瓶臂保持竖直'}

@register(name='bottle_upright_retract',label='装水瓶 · 拧盖臂退开',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='松开拧盖爪，沿当前夹爪接近方向后退；持瓶臂保持。',hint='用于重新观测瓶盖。',
 inputs={'retreat_mm':number('退开距离',30,100,70,5,' mm')})
def upright_retract(ctx,ink,retreat_mm):
    session=getattr(ctx.service,'bottle_cap',None)
    if not session or session.get('mode')!='upright':raise ValueError('需要竖直持瓶记录')
    worker=ctx.other_arm() if ctx.arm==session['holder'] else ctx
    now=ctx.current[worker.arm];R=_rotation(now['rpy_deg']);start=np.asarray(now['xyz'])
    points=[start,start-R[:,2]*retreat_mm/1000];rots=[R,R]
    _probe(worker,points,rots);worker.speed_m_s=.03
    spec=_spec(points,rots,[open_event(0,hold_s=.6)],worker,10)
    spec['bottle_cap']={'stage':'retract','id':uuid.uuid4().hex,'holder':session['holder'],'worker':worker.arm}
    return {'arms':{worker.arm:spec},'summary':'松爪沿接近方向退开，保持持瓶'}


@register(name='bottle_upright_release',label='装水瓶 · 托稳后松爪',group='双臂拧瓶盖',stroke='point',
 summary='仅在人工已托稳瓶子后使用：原地松开持瓶爪，不移动机械臂。',hint='必须先托稳瓶子。',inputs={})
def upright_release(ctx,ink):
    session=getattr(ctx.service,'bottle_cap',None)
    if not session or ctx.arm!=session.get('holder'):raise ValueError('请选择当前持瓶臂')
    now=ctx.current[ctx.arm];R=_rotation(now['rpy_deg']);p=np.asarray(now['xyz'])
    spec=_spec([p,p],[R,R],[open_event(0,hold_s=1.)],ctx,5.)
    spec['bottle_cap']={'stage':'release','id':uuid.uuid4().hex,'holder':ctx.arm}
    return {'arm':spec,'summary':'原地松爪，等待人工放稳瓶子'}


@register(name='bottle_body_recenter',label='装水瓶 · 瓶身定位重抓',group='双臂拧瓶盖',stroke='point',
 summary='点可见瓶身侧面，竖直抓回操作区；完成后必须重新定位瓶盖。',hint='点不透明标签上有可靠深度的区域。',inputs={})
def body_recenter(ctx,ink):
    return prepare(ctx,ink,180,15,1000,.04,body_reference=True)
