"""Two-stage bottle opening. Planning only; the ordinary executor owns all motion.

A point on the upright cap locates a body grip. After horizontal holding has
completed, a fresh observation supplies the second cap point. The holder never
homes between these stages. Two turns are an operation, not proof of an open cap.
"""
import copy
import uuid
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .registry import register, number, integer, choice
from .motion import draft, open_event, close_event, rpy_deg, side_grasp_rotation
from ..grasp import grasp_terms

PRECISION = {'position_mm': 3., 'orientation_deg': 3.}


def _rotation(pose):
    return Rotation.from_euler('xyz', pose, degrees=True).as_matrix()


def _spec(points, rotations, events, ctx, angular):
    n=len(points)-1
    return draft(points, [[i/n,*rpy_deg(r)] for i,r in enumerate(rotations)], events,
                 min(ctx.speed_m_s,.15), ctx.clearance_m,
                 approach_speed_m_s=min(ctx.speed_m_s,.15), angular_speed_deg_s=angular)


class _PoseError(ValueError):
    def __init__(self, message, stage, residual=float('inf')):
        super().__init__(message)
        self.stage = stage
        self.residual = residual


def _probe(ctx, points, rotations, labels=None):
    seed=ctx._seed
    for i,(p,r) in enumerate(zip(points,rotations)):
        if ctx.model is None:continue
        pe,re,seed=ctx.backend.pose_error(ctx.arm,p,r,seed,PRECISION)
        if pe>3. or re>3.:
            stage=labels[i] if labels is not None else '拧盖姿态'
            message=f'{stage}第 {i+1} 点不可达（{pe:.1f} mm / {re:.1f}°）'
            if labels is None:message+='；请调整瓶子位置'
            raise _PoseError(message,stage,max(pe/3.,re/3.))
        if abs(float(seed[5]))>115.:
            stage=labels[i] if labels is not None else '拧盖姿态'
            message=f'{stage}腕部 j6 行程不足' if labels is not None else '拧盖腕部 j6 行程不足，请减小单次转角或调整瓶子方向'
            raise _PoseError(message,stage)


def _compensation(ctx):
    return float(grasp_terms(ctx.arm,ctx.model,ctx.backend.table_checks).get('pinch_compensation_m',0.))


def _point(ctx, ink):
    p=np.asarray(ctx.frame.pick(*ink[0],mode='surface'),dtype=float)
    if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError('瓶盖点需要有效三维坐标，请重新点选')
    return p


@register(name='bottle_cap_prepare',label='拧瓶盖① · 夹瓶身并横置',group='双臂拧瓶盖',stroke='point',
 summary='点竖直瓶子的瓶盖，侧抓瓶身，抬起并水平持瓶；保持姿态等待第二次点盖。',
 hint='选择持瓶臂，点瓶盖中心。瓶盖到瓶身抓点的距离必须按实物设置；完成后重新观测，用另一只臂执行②。',
 limits='要求瓶子初始竖直、瓶身可夹住。不会自动回位；普通塑料瓶，未验证紧盖的扭矩。',
 inputs={'speed_m_s':number('移动速度',.01,.15,.10,.01,' m/s'),
         'approach_speed_m_s':number('接近速度',.01,.15,.10,.01,' m/s'),
         'body_offset_mm':number('瓶盖到瓶身抓点',120,220,140,5,' mm'),
         'body_inset_mm':number('瓶身夹入深度',0,20,15,1,' mm',hint='沿侧抓接近方向深入，避免只用指尖夹持。'),
         'body_diameter_mm':number('瓶身直径',15,65,50,1,' mm'),
         'hold_height_m':number('水平持瓶高度',.20,.45,.28,.01,' m'),
         'hold_reach_m':number('共同操作区前伸',.30,.48,.40,.01,' m'),
         'grip_effort':integer('持瓶夹持力矩',100,2000,1000,' ‰N·m'),
         'angular_speed_deg_s':number('横置角速度',5,45,20,1,' °/s')})
def prepare(ctx, ink, body_offset_mm, body_diameter_mm, body_inset_mm, hold_height_m, hold_reach_m,grip_effort,angular_speed_deg_s,speed_m_s,approach_speed_m_s):
    ctx.speed_m_s=speed_m_s
    cap=_point(ctx,ink);offset=body_offset_mm/1000
    if body_inset_mm>=body_diameter_mm/2:raise ValueError('瓶身夹入深度必须小于瓶身半径')
    tool_offset=np.array([0.,_compensation(ctx),body_inset_mm/1000])
    body=cap-[0,0,offset]
    if body[2]<ctx.table_z+.025:raise ValueError('瓶身抓点过低：请减小瓶盖到瓶身抓点距离，或重新点瓶盖')
    other=ctx.other_arm();axis=np.r_[other.base_xy-ctx.base_xy,0.]
    axis/=np.linalg.norm(axis)
    toward=body[:2]-ctx.base_xy;toward/=np.linalg.norm(toward)
    shared=np.r_[(ctx.base_xy+other.base_xy)/2,hold_height_m]+[hold_reach_m,0,0]
    facing=np.r_[other.base_xy-shared[:2],0.]
    facing/=np.linalg.norm(facing)
    axes=[facing,axis]
    errors=[]
    # Prefer level fingers, then a small downward approach tilt to avoid the
    # wrist limit at low bottle-body grasps. The bottle's world vertical axis,
    # not the gripper tilt, still defines the subsequent 90-degree turn.
    for axis,pitch,roll in [(a,p,r) for p in (0.,10.,20.) for a in axes for r in (0.,30.,-30.,60.,-60.,90.,-90.)]:
        R0=side_grasp_rotation(toward,pitch)
        # Carry upright first, then turn about the body grip in the common workspace.
        turn_axis=np.cross([0.,0.,1.],axis)
        grip=body+R0@tool_offset
        lift=grip+[0,0,max(ctx.clearance_m,body_diameter_mm/1000)]
        middle=np.r_[(ctx.base_xy+other.base_xy)/2,hold_height_m]+[hold_reach_m,0,0]-axis*(offset/2)
        # Enough height for the upright lower bottle section and the swept cylinder.
        middle[2]=max(middle[2],lift[2])
        points=[grip-R0[:,2]*(body_diameter_mm/2000+.02)+[0,0,.025],grip,lift,middle+R0@tool_offset]
        rots=[R0.copy() for _ in points]
        for theta in np.linspace(15,90,6):
            r=(Rotation.from_rotvec(axis*np.deg2rad(roll*theta/90)).as_matrix()
               @Rotation.from_rotvec(turn_axis*np.deg2rad(theta)).as_matrix()@R0)
            points.append(middle+r@tool_offset);rots.append(r)
        try:
            _probe(ctx,points,rots)
            cap_after=middle+axis*(offset-.006)
            trial,trial_rot,_,_=twist_route(cap_after,axis,60.,6,.07,1500,_compensation(other))
            _probe(other,trial,trial_rot)
            break
        except ValueError as exc:
            errors.append(str(exc))
    else:
        raise ValueError("水平侧抓不可达，请调整瓶子位置或共同操作区参数："+"；".join(errors))
    n=len(points)-1
    spec=_spec(points,rots,[open_event(0),close_event(1/n,effort=grip_effort)],ctx,angular_speed_deg_s)
    spec['approach']['speed_m_s']=approach_speed_m_s
    token=uuid.uuid4().hex
    spec['bottle_cap']={'stage':'prepare','id':token,'holder':ctx.arm,'axis':axis.tolist(),
        'upright_rotations':[r.tolist() for r in rots[3:]],
        'upright_positions':[p.tolist() for p in points[3:]],
        'body_tool_offset':tool_offset.tolist(),
        'bottle_height_m':float(cap[2]-ctx.table_z),
        'cap_local':(R0.T@(cap-grip)).tolist(),'axis_local':(R0.T@np.array([0.,0.,1.])).tolist()}
    return {'arm':spec,'summary':'侧抓瓶身并水平持瓶；完成后保持，等待重新点瓶盖。',
            'bottle_axis':axis.tolist(),'bottle_cap_stage':'prepare'}


def held_session(service,current,*,active_hold=False):
    session=getattr(service,'bottle_cap',None)
    if not session:raise ValueError('缺少持瓶定位记录；若已持瓶，请恢复持瓶记录后重新定位瓶盖，勿直接重复抓瓶')
    if session.get('state')=='needs_relocalization':raise ValueError('上次拧盖未完成；持瓶记录已保留，请更新观测并重新点瓶盖')
    if session.get('state')!='waiting_cap':raise ValueError('当前拧盖阶段已结束，需要重新定位瓶盖')
    arm=session['holder'];now=current[arm];saved=session['held_state']
    joint_limit,position_limit,angle_limit=(2.,.008,5.) if active_hold else (.5,.003,2.)
    if (np.max(np.abs(np.asarray(now['joints_deg'])-saved['joints_deg']))>joint_limit
        or np.linalg.norm(np.asarray(now['xyz'])-saved['xyz'])>position_limit
        or np.linalg.norm(Rotation.from_matrix(_rotation(now['rpy_deg'])@_rotation(saved['rpy_deg']).T).as_rotvec())>np.deg2rad(angle_limit)
):
        raise ValueError('持瓶臂已移动或夹爪变化，请重新完成持瓶定位')
    return session


def planning_session(ctx):
    """Re-localize only against a fresh observation; execution keeps strict checks."""
    try:
        return held_session(ctx.service,ctx.current),False
    except ValueError:
        original=getattr(ctx.service,'bottle_cap',None)
        if not original or original.get('state') not in ('waiting_cap','needs_relocalization','completed'):raise
        last=original.get('validated_observation_id',original['observation_id'])
        observed=getattr(ctx.service,'observation_start',None)
        if ctx.frame.id==last or not observed:
            raise ValueError('请更新观测并重新点瓶盖，即可重新规划本轮拧转') from None
        holder=original['holder']
        candidate=copy.deepcopy(original)
        candidate['state']='waiting_cap'
        candidate['held_state']=copy.deepcopy(observed[holder])
        # Same pose/joint/opening limits, now checked against the new image.
        held_session(SimpleNamespace(bottle_cap=candidate),ctx.current)
        candidate['id']=uuid.uuid4().hex
        candidate['validated_observation_id']=ctx.frame.id
        return candidate,True


def direct_cap_session(ctx, ink, holder):
    """Operator-selected holder and fresh cap point replace missing workflow history.

    This does not claim that gripper opening detects a bottle. Geometry is anchored
    to the current pose; normal execution motion/collision guards still apply.
    """
    held=ctx.current[holder]
    center=_point(ctx,ink)
    delta=center-np.asarray(held['xyz'])
    if np.linalg.norm(delta[:2])<.12:
        raise ValueError('瓶盖离持瓶夹爪太近，两臂需要至少 120 mm 间隔')
    axis=np.r_[delta[:2],0.];axis/=np.linalg.norm(axis)
    R=_rotation(held['rpy_deg'])
    return {'id':uuid.uuid4().hex,'state':'waiting_cap','holder':holder,
            'held_state':copy.deepcopy(held),'observation_id':None,
            'validated_observation_id':ctx.frame.id,
            'cap_local':(R.T@delta).tolist(),'axis_local':(R.T@axis).tolist(),
            'source':'operator_cap_point'}


def twist_route(center,axis,angle,cycles,stand_off,effort,compensation,feed_m=0.):
    center=np.asarray(center,dtype=float);axis=np.asarray(axis,dtype=float)
    base=side_grasp_rotation(-axis[:2])
    def tcp(c,r):return c+r[:,1]*compensation
    points=[tcp(center+axis*stand_off,base),tcp(center,base)];rots=[base,base]
    event_indices=[(0,'open')];phases=[]
    for cycle in range(cycles):
        grasp=len(points)-1;event_indices.append((grasp,'close'))
        anchor=center+axis*(feed_m*cycle)
        for theta in np.linspace(angle/6,angle,6):
            r=Rotation.from_rotvec(axis*np.deg2rad(theta)).as_matrix()@base
            c=anchor+axis*(feed_m*theta/angle)
            points.append(tcp(c,r));rots.append(r)
        turned=len(points)-1;event_indices.append((turned,'open'))
        # Open before reversing: never tighten the cap during the wrist reset.
        for theta in np.linspace(angle*5/6,0,6):
            r=Rotation.from_rotvec(axis*np.deg2rad(theta)).as_matrix()@base
            points.append(tcp(center+axis*feed_m*(cycle+1),r));rots.append(r)
        phases.append({'grasp':grasp,'turned':turned,'reset':len(points)-1})
    points.append(tcp(center+axis*(feed_m*cycles+stand_off),base));rots.append(base)
    n=len(points)-1
    events=[open_event(i/n,hold_s=.6) if action=='open' else close_event(i/n,effort=effort) for i,action in event_indices]
    return np.array(points),rots,events,phases


@register(name='bottle_cap_twist',label='拧瓶盖② · 点盖并拧六轮',group='双臂拧瓶盖',stroke='point',
 summary='水平持瓶后重新点瓶盖；另一臂接近，合爪拧、松爪复位，默认重复六轮后退开。',
 hint='已持瓶可直接点瓶盖端面中心，无需重做①。没有记录时选择持瓶臂；另一臂拧盖。瓶子应保持水平。',
 limits='动作完成仅代表动作完成，不保证瓶盖已完全拧开；持瓶臂保持，拧盖臂松开退回。',arms='dual',
 inputs={'speed_m_s':number('移动速度',.01,.15,.08,.01,' m/s'),
         'approach_speed_m_s':number('接近速度',.01,.15,.08,.01,' m/s'),
         'holder_arm':choice('持瓶臂（无记录时需选）',[('auto','沿用定位；无记录用当前所选臂'),('left','左臂持瓶'),('right','右臂持瓶')],'auto'),
         'turn_deg':number('每轮拧转角',10,90,60,5,' °'),
         'cycles':integer('拧转轮数',1,12,6,' 轮'),
         'direction':choice('方向',[('loosen','松盖 · 外侧看逆时针'),('tighten','拧紧 · 外侧看顺时针')],'loosen'),
         'angular_speed_deg_s':number('拧转角速度',2,30,10,1,' °/s'),
         'grip_effort':integer('瓶盖夹持力矩',100,1500,1500,' ‰N·m'),
         'cap_inset_mm':number('从瓶盖端面向内夹入',0,20,6,1,' mm'),
         'stand_off_mm':number('接近及退开距离',40,120,70,5,' mm'),
         'feed_mm':number('每轮沿瓶轴外移',0,2,0,.1,' mm',hint='默认原地拧；仅已知螺距时设置。')})
def twist(ctx,ink,holder_arm,turn_deg,cycles,direction,angular_speed_deg_s,grip_effort,stand_off_mm,feed_mm,cap_inset_mm,speed_m_s,approach_speed_m_s):
    ctx.speed_m_s=speed_m_s
    if holder_arm!='auto' or not getattr(ctx.service,'bottle_cap',None):
        session=direct_cap_session(ctx,ink,ctx.arm if holder_arm=='auto' else holder_arm)
        relocalized=True
    else:
        session,relocalized=planning_session(ctx)
    if ctx.frame.id==session['observation_id']:raise ValueError('持瓶后需要重新观测，再点瓶盖')
    worker=ctx.other_arm() if ctx.arm==session['holder'] else ctx
    center=_point(worker,ink);held=ctx.current[session['holder']]
    R=_rotation(held['rpy_deg']);axis=R@np.asarray(session['axis_local']);axis/=np.linalg.norm(axis)
    if abs(axis[2])>.06:raise ValueError('瓶轴没有保持水平，请重新定位持瓶')
    expected=np.asarray(held['xyz'])+R@np.asarray(session['cap_local'])
    if np.linalg.norm(center-expected)>.04:raise ValueError('新瓶盖点偏离持瓶时预测位置超过 4 cm，请核对瓶盖点及瓶身偏移')
    if np.linalg.norm(center-np.asarray(held['xyz']))<.12:raise ValueError('瓶盖离持瓶夹爪太近，两臂需要至少 120 mm 间隔')
    # A clicked RGB-D surface is not the cap axis (typically its upper surface).
    # Keep only its axial displacement; the held bottle defines the radial center.
    surface=center.copy()
    center=expected+axis*float(np.dot(surface-expected,axis))-axis*(cap_inset_mm/1000)
    if np.linalg.norm(center-np.asarray(held['xyz']))<.12:
        raise ValueError('修正后的瓶盖夹持点离持瓶夹爪不足 120 mm')
    angle=turn_deg if direction=='loosen' else -turn_deg
    points,rots,events,phases=twist_route(center,axis,angle,cycles,stand_off_mm/1000,grip_effort,_compensation(worker),feed_mm/1000)
    _probe(worker,points,rots)
    spec=_spec(points,rots,events,worker,angular_speed_deg_s)
    spec['approach']['speed_m_s']=approach_speed_m_s
    # Only accept the new reference after cap geometry and the whole IK route pass.
    if relocalized:ctx.service.bottle_cap=copy.deepcopy(session)
    spec['bottle_cap']={'stage':'twist','id':session['id'],'holder':session['holder'],'worker':worker.arm,'cycles':cycles}
    return {'arms':{worker.arm:spec},'summary':f'{cycles} 轮合爪拧转、松爪复位；拧盖臂退开，持瓶臂保持。',
            'bottle_cap_stage':'twist','phases':phases,'cap_xyz':center.tolist(),'bottle_axis':axis.tolist(),'cap_surface_xyz':surface.tolist(),
            'cap_axis_expected_xyz':expected.tolist()}


@register(name='bottle_cap_retract',label='拧瓶盖 · 松爪退开重试',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='抓盖失败后，拧盖臂松爪并向瓶盖外侧退开，持瓶臂保持。',
 hint='选择持瓶臂，点画面生成退开轨迹；退开后重新选择②并点瓶盖。',
 limits='仅退开拧盖臂，不重复抓瓶，也不自动重试接触。',
 inputs={'holder_arm':choice('持瓶臂',[('left','左臂持瓶'),('right','右臂持瓶')],'left'),
         'retreat_mm':number('退开距离',20,100,70,5,' mm')})
def retract(ctx,ink,holder_arm,retreat_mm):
    worker=ctx.other_arm() if ctx.arm==holder_arm else ctx
    now=ctx.current[worker.arm];held=ctx.current[holder_arm]
    axis=np.asarray(now['xyz'])-np.asarray(held['xyz']);axis[2]=0.
    if np.linalg.norm(axis)<.03:raise ValueError('两臂过近，无法确定退开方向')
    axis/=np.linalg.norm(axis)
    existing=getattr(ctx.service,'bottle_cap',None)
    if existing and existing.get('holder')==holder_arm and existing.get('axis_local') is not None:
        stored_axis=_rotation(held['rpy_deg'])@np.asarray(existing['axis_local'])
        if np.dot(stored_axis,axis)<.5:raise ValueError('当前退开方向与瓶轴不一致，请重新定位')
        axis=stored_axis/np.linalg.norm(stored_axis)
    start=np.asarray(now['xyz']);rotation=_rotation(now['rpy_deg'])
    points=[start,start+axis*retreat_mm/1000];rots=[rotation,rotation]
    _probe(worker,points,rots)
    worker.speed_m_s=.02
    spec=_spec(points,rots,[open_event(0,hold_s=.6)],worker,5.)
    spec['bottle_cap']={'stage':'retract','id':uuid.uuid4().hex,'holder':holder_arm,'worker':worker.arm}
    return {'arms':{worker.arm:spec},'summary':'拧盖臂松爪退开；重新点瓶盖即可重试。'}


@register(name='bottle_cap_place',label='拧瓶盖③ · 竖直放置',group='双臂拧瓶盖',stroke='point',arms='dual',
 summary='点桌面放置位置；拧盖臂先松爪退开并回收纳位，随后持瓶臂转正放下。',
 hint='更新观测后点桌面空位。旧持瓶记录需核对瓶子总高度；仅适用于瓶子没有在夹爪内滑动的情况。',
 limits='必须先持瓶；确认瓶盖臂已退开。不可达时不会放开瓶子。',
 inputs={'speed_m_s':number('放置移动速度',.01,.15,.08,.01,' m/s'),
         'approach_speed_m_s':number('下降速度',.01,.10,.04,.01,' m/s'),
         'angular_speed_deg_s':number('转正角速度',2,30,10,1,' °/s'),
         'worker_retreat_mm':number('拧盖臂离盖距离',70,220,170,10,' mm',hint='先退到距瓶盖至少此距离，再回收纳位；已退开的距离会计入。'),
         'bottle_height_mm':number('瓶子总高度（旧记录）',100,350,210,5,' mm')})
def place(ctx,ink,speed_m_s,approach_speed_m_s,angular_speed_deg_s,bottle_height_mm,worker_retreat_mm):
    session,relocalized=planning_session(ctx)
    if ctx.frame.id==session['observation_id']:raise ValueError('请更新观测再选择放置位置')
    holder=ctx if ctx.arm==session['holder'] else ctx.other_arm()
    holder.speed_m_s=speed_m_s
    target=_point(holder,ink)
    if abs(target[2]-holder.table_z)>.04:raise ValueError('请选择桌面空位作为瓶底放置点')
    h=float(session.get('bottle_height_m',bottle_height_mm/1000))
    local_cap=np.asarray(session['cap_local']);local_axis=np.asarray(session['axis_local'])
    bottom_local=local_cap-local_axis*h
    now=ctx.current[holder.arm];R=_rotation(now['rpy_deg']);tcp=np.array(now['xyz'])
    # The bottle must be upright, but its yaw is free. Search side approach
    # headings instead of binding placement to the radial grasp orientation.
    stored=session.get('upright_rotations')
    nominal=np.array(stored[0]) if stored else side_grasp_rotation(target[:2]-holder.base_xy)
    mapped=nominal@local_axis
    nominal=Rotation.align_vectors([[0.,0.,1.]],[mapped])[0].as_matrix()@nominal
    center_local=local_cap-local_axis*h/2
    center=tcp+R@center_local
    center[2]=max(center[2],target[2]+ctx.clearance_m+h/2+.03)
    errors=[]
    for yaw in (0.,30.,-30.,60.,-60.,90.,-90.,120.,-120.,180.):
        goal=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@nominal
        if stored:
            # Reverse the grasp-pivot trajectory, not a new bottle-center orbit.
            positions=session.get('upright_positions')
            if positions and len(positions)==len(stored):
                points=[tcp]+[np.asarray(p) for p in reversed(positions)]
            else:
                # Legacy sessions stored only rotations. Recover transverse TCP
                # offset from cap geometry and anchor the pivot at the live pose.
                offset=np.asarray(session.get('body_tool_offset',
                    local_axis*np.dot(local_cap,local_axis)-local_cap))
                pivot=tcp-R@offset
                points=[tcp]+[pivot+np.asarray(r)@offset for r in reversed(stored)]
            rots=[R]+[np.asarray(r) for r in reversed(stored)]
        else:
            rotations=list(Slerp([0,1],Rotation.from_matrix([R,nominal]))(np.linspace(0,1,13)).as_matrix())
            points=[tcp,center-R@center_local];rots=[R,R]
            for r in rotations[1:]:points.append(center-r@center_local);rots.append(r)
        if abs(float((goal@local_axis)[2])-1)>.002:
            raise ValueError('无法确认瓶子转正方向，请重新定位持瓶')
        labels=['持瓶转正']*len(points)
        contact=target-goal@bottom_local
        hover=contact+[0,0,ctx.clearance_m]
        # Yaw is free for an upright bottle. Turn during transport toward the
        # destination, rather than rotating in place at the extended hold pose.
        departure=np.array(points[-1]);initial=rots[-1]
        transfer_start=len(points)-1
        fractions=np.linspace(0,1,7)[1:]
        for t,r in zip(fractions,Slerp([0,1],Rotation.from_matrix([initial,goal]))(fractions).as_matrix()):
            points.append((1-t)*departure+t*hover);rots.append(r);labels.append('移向落点')
        points.append(contact);rots.append(goal);labels.append('下降落桌');release=len(points)-1
        # After release retrace the already reachable descent and carry route
        # with open jaws. A new low sideways retreat can hit the wrist limit
        # even though both the placement and its incoming path are reachable.
        retreat_points=[np.array(p) for p in reversed(points[transfer_start:release])]
        retreat_rots=[np.array(r) for r in reversed(rots[transfer_start:release])]
        points.extend(retreat_points);rots.extend(retreat_rots);labels.extend(['松爪退开']*len(retreat_points))
        try:
            _probe(holder,points,rots,labels)
            break
        except _PoseError as exc:errors.append(exc)
    else:
        order={'持瓶转正':0,'移向落点':1,'下降落桌':2,'松爪退开':3}
        error=max(errors,key=lambda e:(order.get(e.stage,0),-e.residual))
        raise ValueError('放置路径不可达：'+str(error)+'；已尝试多个水平朝向，请重新选择落点')
    spec=_spec(points,rots,[open_event(release/(len(points)-1),hold_s=1.)],holder,angular_speed_deg_s)
    spec['approach']['speed_m_s']=approach_speed_m_s
    # Use the lower speed for the whole placement to include the final descent.
    spec['speed']=[[0,min(speed_m_s,approach_speed_m_s)],[1,min(speed_m_s,approach_speed_m_s)]]
    # Both arms share one trajectory clock: hold the bottle until the cap hand
    # has opened and completed its outward retreat. Full preview checks both arms.
    worker=holder.other_arm();ws=ctx.current[worker.arm];wr=_rotation(ws['rpy_deg'])
    axis=R@local_axis;axis/=np.linalg.norm(axis)
    start=np.asarray(ws['xyz'])
    cap=tcp+R@local_cap
    axial_gap=float(np.dot(start-cap,axis))
    clear=start+axis*max(0.,worker_retreat_mm/1000-axial_gap)
    if np.dot(start-np.asarray(now['xyz']),axis)<.03:
        raise ValueError('拧盖臂不在瓶盖外侧，请先使用松爪退开重试')
    _probe(worker,[start,clear],[wr,wr])
    worker.speed_m_s=.02
    retreat=_spec([start,clear],[wr,wr],[open_event(0,hold_s=.6)],worker,5.)
    from ..trajectory import compile_arm
    retreat_time=float(compile_arm(retreat,ws)['time_s'][-1])
    spec['start_hold_s']=retreat_time+.25
    if relocalized:ctx.service.bottle_cap=copy.deepcopy(session)
    spec['bottle_cap']={'stage':'place','id':session['id'],'holder':holder.arm,'worker':worker.arm}
    return {'arms':{worker.arm:retreat,holder.arm:spec},'bottle_cap_stage':'place','release_index':release,
        'bottle_bottom_xyz':target.tolist(),'place_yaw_deg':yaw,'summary':'拧盖臂先松爪退开并回收纳位；确认到位后持瓶臂反向转正、放瓶回位。'}



def stage_metadata(plan):
    arms=(plan.get('input_draft') or {}).get('arms',{})
    entries=[(arm,spec['bottle_cap']) for arm,spec in arms.items() if 'bottle_cap' in spec]
    if not entries:return None
    if len(entries)!=1:raise ValueError('拧瓶盖阶段不能与其他轨迹合并执行')
    arm,m=entries[0]
    allowed={m.get('holder'),m.get('worker')} if m.get('stage') in ('place','twist') and m.get('worker') and (m.get('stage')=='place' or m.get('active_holder')) else {arm}
    if set(arms)!=allowed:raise ValueError('拧瓶盖阶段不能与其他轨迹合并执行')
    if m.get('stage') not in ('prepare','twist','place','retract','release','inspect') or m.get('holder') not in ('left','right'):
        raise ValueError('无效的拧瓶盖阶段')
    if (m['stage'] in ('prepare','place','release','inspect') and arm!=m['holder']) or (m['stage'] in ('twist','retract') and arm==m['holder']):
        raise ValueError('拧瓶盖持瓶臂与拧盖臂不能相同')
    return m


def validate_stage(service,plan,current):
    m=stage_metadata(plan)
    if m and m['stage'] in ('twist','place'):
        session=held_session(service,current)
        if m['id']!=session['id']:raise ValueError('拧盖会话已改变，请重新规划')
        if m['stage']=='twist' and m.get('active_holder'):
            # Command the holder throughout the shared clock; only a stationary,
            # closed-jaw reference is accepted as an active cap-turning support.
            spec=plan['input_draft']['arms'][m['holder']];now=current[m['holder']]
            points=np.asarray(spec.get('path',{}).get('points',[]),float)
            keys=np.asarray(spec.get('orientation',{}).get('points',[]),float)
            if points.shape!=(2,3) or not np.allclose(points,now['xyz'],atol=.001,rtol=0):
                raise ValueError('协同拧盖时持瓶臂必须保持当前姿态')
            if keys.shape!=(2,4) or not np.allclose(keys[:,1:],now['rpy_deg'],atol=.2,rtol=0):
                raise ValueError('协同拧盖时持瓶臂必须保持当前姿态')
            if any('opening_mm' in e for e in spec.get('gripper_events',[])):
                raise ValueError('协同拧盖时持瓶臂不得松爪或重新夹持')
    return m


def complete_stage(service,plan):
    m=stage_metadata(plan)
    if not m:
        session=getattr(service,'bottle_cap',None)
        if session and session.get('state')=='needs_relocalization' and session['holder'] not in plan.get('arms',{}):
            return
        service.bottle_cap=None
        return
    if m['stage']=='prepare':
        service.bottle_cap={**copy.deepcopy(m),'state':'waiting_cap',
            'held_state':copy.deepcopy(service.backend.state()[m['holder']]),
            'observation_id':plan['observation_id']}
    elif m['stage']=='retract':
        if service.bottle_cap:
            service.bottle_cap={**service.bottle_cap,'state':'needs_relocalization',
                'validated_observation_id':plan['observation_id']}
    elif m['stage'] in ('place','release','inspect'):
        service.bottle_cap=None
    else:
        service.bottle_cap={**service.bottle_cap,'state':'completed','cycles':m.get('cycles',6),
            'validated_observation_id':plan['observation_id']}


def failed_stage(service,plan,reason):
    session=getattr(service,'bottle_cap',None)
    m=stage_metadata(plan)
    if session and m and m['stage']=='twist':
        service.bottle_cap={**copy.deepcopy(session),'state':'needs_relocalization',
            'last_error':str(reason),'validated_observation_id':service.frame.id if service.frame else None}
    else:
        service.bottle_cap=None
