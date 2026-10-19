"""Upright bottle preparation and world-axis cap turning; ordinary preview/executor required."""
import copy,uuid
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .registry import register,number,integer,boolean
from .motion import side_grasp_rotation,open_event,close_event
from .bottle_cap import _point,_probe,_spec,_compensation,_rotation,planning_session
from ..upright_cap import cap_geometry, measured_tool_offset, aligned_turn_route

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
 inputs={'body_offset_mm':number('盖到抓点',120,220,180,5,' mm'),
 'body_inset_mm':number('夹入深度',0,25,23,1,' mm'),
 'body_pitch_deg':number('持瓶接近俯角',0,30,20,5,' °'),
 'grip_effort':integer('持瓶力矩',100,2000,2000),
 'approach_speed_m_s':number('接近速度',.01,.15,.05,.01,' m/s'),
 'speed_m_s':number('速度',.01,.15,.05,.01,' m/s')})
def prepare(ctx,ink,body_offset_mm,body_inset_mm,grip_effort,speed_m_s,body_reference=False,body_pitch_deg=20,approach_speed_m_s=.05):
    cap=_point(ctx,ink)
    if body_reference:cap=cap+[0,0,body_offset_mm/1000]
    body=cap-[0,0,body_offset_mm/1000];ctx.speed_m_s=speed_m_s
    if body[2]<ctx.table_z+.025:raise ValueError('瓶身抓点过低')
    d=body[:2]-ctx.base_xy;d/=np.linalg.norm(d);R=side_grasp_rotation(d,body_pitch_deg)
    offset=np.array([0,_compensation(ctx),body_inset_mm/1000]);grip=body+R@offset
    if grip[2]<ctx.table_z+.025:raise ValueError('降低后的持瓶抓点太靠近桌面，请减小盖到抓点距离')
    other=ctx.other_arm();middle=(ctx.base_xy+other.base_xy)/2
    toward_holder=ctx.base_xy-middle;toward_holder/=np.linalg.norm(toward_holder)
    goal=np.r_[middle+[.33,0]+toward_holder*.020,grip[2]+.020]
    points=[grip-R[:,2]*.07,grip,grip+[0,0,.07],goal+[0,0,.04],goal]
    rots=[R]*len(points);_probe(ctx,points,rots)
    cap_local=R.T@(cap-grip);cap_after=goal+R@cap_local
    spec=_spec(points,rots,[open_event(0),close_event(.25,effort=grip_effort)],ctx,15)
    spec['bottle_cap']={'stage':'prepare','id':uuid.uuid4().hex,'holder':ctx.arm,'mode':'upright',
      'axis':[0,0,1],'axis_local':(R.T@np.array([0,0,1.])).tolist(),'cap_local':cap_local.tolist(),
      'bottle_height_m':float(cap[2]-ctx.table_z),'body_offset_m':body_offset_mm/1000,'body_tool_offset':offset.tolist(),
      'pickup_positions':[p.tolist() for p in points],'pickup_rotations':[r.tolist() for r in rots],
      'upright_rotations':[R.tolist()],'upright_positions':[goal.tolist()]}
    return {'arm':spec,'cap_xyz':cap_after.tolist(),'summary':'竖直持瓶；完成后重新观察瓶盖'}

def _upright_context(ctx,ink,inset_mm,speed):
    original=getattr(ctx.service,'bottle_cap',None) or {}
    last_frame=original.get('validated_observation_id',original.get('observation_id'))
    if ctx.frame.id==last_frame:raise ValueError('请重新观测并点选当前瓶盖')
    session,_=planning_session(ctx)
    if session.get('mode')!='upright':raise ValueError('需要竖直持瓶记录')
    if session.get('cap_removed'):raise ValueError('瓶盖已取下，请开始新的持瓶任务')
    worker=ctx.other_arm() if ctx.arm==session['holder'] else ctx;worker.speed_m_s=speed
    held=ctx.current[session['holder']];R=_rotation(held['rpy_deg']);surface=_point(worker,ink)
    expected=np.asarray(held['xyz'])+R@np.asarray(session['cap_local'])
    if np.linalg.norm(surface-expected)>.04:raise ValueError('瓶盖观测偏离预测超过4cm，请重新定位')
    center,axis=cap_geometry(surface,held['xyz'],R,session['body_tool_offset'],inset_mm/1000)
    session['cap_local']=(R.T@(surface-held['xyz'])).tolist();session['axis_local']=(R.T@axis).tolist()
    return worker,session,center,axis


def _held_draft(ctx,session,worker,points,rotations,events,phase,**metadata):
    _probe(worker,points,rotations)
    spec=_spec(points,rotations,events,worker,20)
    spec['bottle_cap']={'stage':'twist','id':session['id'],'holder':session['holder'],
        'worker':worker.arm,'active_holder':True,'upright_phase':phase,**metadata}
    h=ctx.current[session['holder']]
    support={'path':{'mode':'waypoints','points':[h['xyz'],h['xyz']]},
      'orientation':{'mode':'keyframes','points':[[0,*h['rpy_deg']],[1,*h['rpy_deg']]]},
      'speed':.01,'gripper_events':[],'approach':{'speed_m_s':.01,'clearance_m':.02}}
    ctx.service.bottle_cap=copy.deepcopy(session)
    return {'arms':{worker.arm:spec,session['holder']:support}}


@register(name='bottle_upright_grasp_cap',label='装水瓶② · 定位夹盖',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='重新点瓶盖，从外侧稍向下接近并夹住；保持持瓶臂。',hint='夹紧后重新观测，再执行③对心拧盖。',
 inputs={'cap_inset_mm':number('盖顶到夹点',3,15,8,1,' mm'),
 'grip_depth_mm':number('夹入深度',0,20,12,1,' mm'),
 'pitch_deg':number('接近俯角',0,40,20,5,' °'),
 'approach_offset_deg':number('外侧接近偏角',-60,60,25,5,' °'),
 'grip_effort':integer('拧盖力矩',100,2500,2500),
 'approach_speed_m_s':number('接近速度',.01,.10,.02,.01,' m/s'),
 'speed_m_s':number('移动速度',.01,.10,.02,.01,' m/s')})
def grasp_cap(ctx,ink,cap_inset_mm,grip_depth_mm,pitch_deg,approach_offset_deg,grip_effort,speed_m_s,approach_speed_m_s=.02):
    worker,session,center,axis=_upright_context(ctx,ink,cap_inset_mm,speed_m_s)
    direction=center[:2]-worker.base_xy;direction/=np.linalg.norm(direction)
    a=np.deg2rad(approach_offset_deg)*(1 if worker.arm=='right' else -1)
    direction=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@direction
    R=side_grasp_rotation(direction,pitch_deg);offset=np.array([0,_compensation(worker),grip_depth_mm/1000])
    target=center+R@offset;points=[target-R[:,2]*.085,target];rotations=[R,R]
    now=ctx.current[worker.arm];start=np.asarray(now['xyz']);current_R=_rotation(now['rpy_deg'])
    if np.linalg.norm(start-center)<.09:
        points=[start,start-current_R[:,2]*.04]+points
        rotations=[current_R,current_R]+rotations
    result=_held_draft(ctx,session,worker,points,rotations,
        [open_event(0,opening_mm=50),close_event(1,effort=grip_effort)],'grasp',
        upright_reference={'base_rotation':R.tolist()})
    return {**result,'cap_center':center.tolist(),'bottle_axis':axis.tolist(),'summary':'夹盖后重新观测瓶盖中心，再执行③'}


@register(name='bottle_upright_twist',label='装水瓶③ · 对心拧盖',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='按夹紧后新观测的瓶盖中心拧转，补偿完整夹入偏移；末轮保持夹盖。',
 hint='必须先完成②并重新点瓶盖；空抓时回到②重新定位。',
 inputs={'turn_deg':number('每轮角度',5,60,30,5,' °'),'cycles':integer('轮数',1,6,6),
 'grip_effort':integer('拧盖力矩',100,2500,2500),'cap_inset_mm':number('盖顶到夹点',3,15,8,1,' mm'),
 'approach_speed_m_s':number('接近速度',.01,.10,.03,.01,' m/s'),
 'speed_m_s':number('速度',.01,.10,.03,.01,' m/s')})
def twist(ctx,ink,turn_deg,cycles,grip_effort,cap_inset_mm,speed_m_s,approach_speed_m_s=.03):
    worker,session,center,axis=_upright_context(ctx,ink,cap_inset_mm,speed_m_s)
    reference=session.get('upright_reference')
    if not reference:raise ValueError('请先执行②定位夹盖')
    now=ctx.current[worker.arm]
    if not 1.5<now['gripper_mm']<67:raise ValueError('当前夹爪未夹住瓶盖，请重新执行②')
    current=_rotation(now['rpy_deg']);offset=measured_tool_offset(now['xyz'],current,center)
    base=np.asarray(reference['base_rotation'])
    points,rots,events=aligned_turn_route(center,axis,offset,current,base,turn_deg,cycles,effort=grip_effort)
    result=_held_draft(ctx,session,worker,points,rots,events,'turn',cycles=cycles,
        upright_reference={'base_rotation':base.tolist(),'axis':axis.tolist(),'tool_offset':offset.tolist()})
    return {**result,'cap_center':center.tolist(),'bottle_axis':axis.tolist(),'tool_offset':offset.tolist(),
        'summary':'完成本批拧转后保持夹盖；重新观察，确认松脱后再取盖'}


@register(name='bottle_upright_take_cap',label='装水瓶④ · 夹盖提离',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='保持夹盖沿瓶轴小幅提起；确认已脱离后再提离退开。',
 hint='默认先提 5 mm。观察到瓶口与瓶盖分离后，再确认脱离并退开。',
 inputs={'lift_mm':number('沿瓶轴提起',0,20,5,1,' mm'),
 'retreat_mm':number('夹盖退开',0,100,0,5,' mm'),
 'cap_detached':boolean('已观察到瓶盖脱离',False),
 'approach_speed_m_s':number('接近速度',.005,.05,.01,.005,' m/s'),
 'speed_m_s':number('速度',.005,.05,.01,.005,' m/s')})
def take_cap(ctx,ink,lift_mm,retreat_mm,cap_detached,speed_m_s,approach_speed_m_s=.01):
    session,_=planning_session(ctx)
    if session.get('mode')!='upright' or not session.get('upright_reference'):
        raise ValueError('请先完成竖直夹盖和拧转')
    if not cap_detached and (lift_mm>5 or retreat_mm>0):raise ValueError('先小幅提起并观察脱离，再退开')
    worker=ctx.other_arm() if ctx.arm==session['holder'] else ctx;worker.speed_m_s=speed_m_s
    now=ctx.current[worker.arm];h=ctx.current[session['holder']]
    if not 1.5<now['gripper_mm']<67:raise ValueError('当前夹爪未夹住瓶盖')
    axis=_rotation(h['rpy_deg'])@np.asarray(session['axis_local']);R=_rotation(now['rpy_deg']);p=np.asarray(now['xyz'])
    points=[p,p+axis*lift_mm/1000]
    if retreat_mm:points.append(points[-1]-R[:,2]*retreat_mm/1000)
    return {**_held_draft(ctx,session,worker,points,[R]*len(points),[],'take',cap_removed=bool(cap_detached)),
        'summary':'保持瓶身和瓶盖夹持；提离后由相机确认瓶口敞开'}


@register(name='bottle_upright_place_cap',label='装水瓶⑤ · 放下瓶盖',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='点空桌面，将已分离的瓶盖低位放下并退开；持瓶臂保持。',
 hint='完成后先让拧盖臂回位，再执行⑥沿原路放瓶。',
 inputs={'release_clearance_mm':number('盖中心离桌高度',20,60,35,5,' mm'),
 'approach_speed_m_s':number('接近速度',.01,.10,.04,.01,' m/s'),
 'speed_m_s':number('速度',.01,.10,.04,.01,' m/s')})
def place_cap(ctx,ink,release_clearance_mm,speed_m_s,approach_speed_m_s=.04):
    session,_=planning_session(ctx)
    if session.get('mode')!='upright' or not session.get('cap_removed'):
        raise ValueError('请先确认瓶盖已分离并提离')
    if session.get('cap_placed'):raise ValueError('瓶盖已经放下，请继续放瓶')
    worker=ctx.other_arm() if ctx.arm==session['holder'] else ctx;worker.speed_m_s=speed_m_s
    now=ctx.current[worker.arm]
    if not 1.5<now['gripper_mm']<67:raise ValueError('当前夹爪未夹住瓶盖')
    target=_point(worker,ink)
    if abs(target[2]-ctx.table_z)>.04:raise ValueError('请选择空桌面作为放盖位置')
    R=_rotation(now['rpy_deg']);start=np.asarray(now['xyz'])
    offset=np.asarray(session['upright_reference']['tool_offset'])
    center=target+[0,0,release_clearance_mm/1000]
    direction=target[:2]-worker.base_xy;direction/=np.linalg.norm(direction)
    candidates=[R]+[side_grasp_rotation(direction,pitch) for pitch in (20,30,10,0)]
    last_error=None
    for goal in candidates:
        contact=center+goal@offset;hover=contact.copy();hover[2]=max(start[2],contact[2]+.08)
        fractions=np.linspace(0,1,7)
        rots=list(Slerp([0,1],Rotation.from_matrix([R,goal]))(fractions).as_matrix())
        points=[(1-t)*start+t*hover for t in fractions]
        points.extend([contact,hover]);rots.extend([goal,goal])
        try:
            result=_held_draft(ctx,session,worker,points,rots,
                [open_event((len(points)-2)/(len(points)-1),opening_mm=50,hold_s=.8)],
                'place_cap',cap_placed=True)
            return {**result,'release_center':center.tolist(),
                'summary':'瓶盖已放下后，收回拧盖臂并沿原路放瓶'}
        except ValueError as exc:last_error=exc
    raise ValueError('放盖位置不可达：'+str(last_error))


@register(name='bottle_upright_return',label='装水瓶⑥ · 原路放瓶回位',group='双臂拧瓶盖',stroke='point',
 summary='倒放本次持瓶路线，在原抓取位置放下瓶子，退开并自动回位。',
 hint='先完成⑤放盖并收回拧盖臂；原来的瓶底位置必须空着。',
 inputs={'approach_speed_m_s':number('接近速度',.01,.10,.04,.01,' m/s'),
 'speed_m_s':number('放瓶速度',.01,.10,.04,.01,' m/s')})
def return_bottle(ctx,ink,speed_m_s,approach_speed_m_s=.04):
    session,_=planning_session(ctx)
    if session.get('mode')!='upright' or not session.get('cap_placed'):
        raise ValueError('请先放下瓶盖并收回拧盖臂')
    holder=ctx if ctx.arm==session['holder'] else ctx.other_arm();holder.speed_m_s=speed_m_s
    positions=session.get('pickup_positions');rotations=session.get('pickup_rotations')
    if not positions or len(positions)!=5 or not rotations or len(rotations)!=5:
        raise ValueError('缺少本次抓瓶路线，请重新定位放瓶位置')
    now=ctx.current[holder.arm];start=np.asarray(now['xyz']);R=_rotation(now['rpy_deg'])
    if np.linalg.norm(start-np.asarray(positions[-1]))>.025:
        raise ValueError('当前持瓶位置已偏离原路线，请重新定位放瓶位置')
    points=[start]+[np.asarray(p) for p in positions[-2:0:-1]]
    rots=[R]+[np.asarray(r) for r in rotations[-2:0:-1]]
    release=len(points)-1
    points.extend([np.asarray(positions[0]),np.asarray(positions[0])+[0,0,.07]])
    rots.extend([np.asarray(rotations[0])]*2)
    _probe(holder,points,rots,['原路放瓶']*len(points))
    spec=_spec(points,rots,[open_event(release/(len(points)-1),hold_s=1.)],holder,15)
    spec['bottle_cap']={'stage':'place','id':session['id'],'holder':holder.arm,'upright_phase':'return'}
    ctx.service.bottle_cap=copy.deepcopy(session)
    return {'arm':spec,'summary':'原路放回瓶子，松爪退开后自动回收纳位'}

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
