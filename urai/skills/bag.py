"""Observed flap-edge pinch and peel. Planning never labels a closed bag as open."""
import numpy as np
from scipy.spatial.transform import Rotation
from .registry import register, number, integer
from .motion import draft, down_rotation, jaw_yaw_near
from ..grasp import grasp_terms
from ..pour import rpy_deg

PRECISION = {'position_mm': 3., 'orientation_deg': 3.}


def edge_geometry(frame, ink, offset_mm):
    """First two pixels bound the free lip; last pixel identifies the flap/hinge side."""
    ink = np.asarray(ink, float)
    if len(ink) < 3:
        raise ValueError('标注包口边缘两端，再点上层布片内侧；不能只点包底或提手')
    if len(ink) > 3:
        baseline = ink[-1]-ink[0]
        norm = np.linalg.norm(baseline)
        if norm < 12:
            raise ValueError('请沿包口画到拐点，再转向布片内侧')
        distances = np.abs(np.cross(ink-ink[0], baseline))/norm
        corner = int(np.argmax(distances))
        if corner in (0,len(ink)-1) or distances[corner] < 12:
            raise ValueError('需要包口边缘到布片内侧的清晰拐点')
        ink = ink[[0,corner,-1]]
    mid = (ink[0]+ink[1])/2
    tangent = ink[1]-ink[0]
    if np.linalg.norm(tangent) < 12:
        raise ValueError('包口边缘两端过近')
    normal = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
    side = float((ink[-1]-mid) @ normal)
    if abs(side) < 12:
        raise ValueError('第三点应在上层布片内侧，不能与包口边缘共线')
    normal *= np.sign(side)
    # Measure on the upper fabric, not across the depth discontinuity at the lip.
    sample = mid + normal*5
    surface = np.asarray(frame.pick(*sample, mode='surface'), float)
    ends = np.array([frame.pick(*p, mode='plane', z=float(surface[2])) for p in ink[:2]])
    edge = ends[1,:2]-ends[0,:2]
    edge /= np.linalg.norm(edge)
    inside = np.array([-edge[1], edge[0]])
    hinge = np.asarray(frame.pick(*ink[-1], mode='plane', z=float(surface[2])))
    centre = np.asarray(frame.pick(*mid, mode='plane', z=float(surface[2])))
    if (hinge[:2]-centre[:2]) @ inside < 0:
        inside *= -1
    centre[:2] += inside*offset_mm/1000
    return centre, inside, sample


def checked_spec(ctx, points, rotations, events, speed=.15):
    seed = ctx.current[ctx.arm]['joints_deg']
    for i, (point, rotation) in enumerate(zip(points, rotations)):
        if ctx.has_kinematics:
            pe, re, seed = ctx.backend.pose_error(ctx.arm, point, rotation, seed, PRECISION)
            if pe > 3 or re > 3:
                raise ValueError(f'开包姿态第 {i+1} 点不可达（{pe:.1f} mm / {re:.1f}°）；请换抓取位置')
    keys = [[i/(len(points)-1), *rpy_deg(r)] for i,r in enumerate(rotations)]
    return draft(points, keys, events, min(ctx.speed_m_s,speed), ctx.clearance_m,
                 approach_speed_m_s=.12, angular_speed_deg_s=35.)


@register(name='bag_flap_grasp', label='开包① · 浅夹上层试提', group='柔性物体', stroke='stroke',
 summary='沿自由边缘浅夹上层布片，垂直试提后保持；不把试提视为开包成功。',
 hint='沿自由边缘画到另一端，再转向上层布片内侧，形成折线。夹爪跨过边缘。',
 inputs={'speed_m_s':number('试提速度',.01,.15,.15,.01,' m/s'),
         'approach_speed_m_s':number('接近速度',.01,.3,.12,.01,' m/s'),
         'edge_offset_mm':number('沿边缘向布片内侧偏移',1,12,5,1,' mm'),
         'pinch_depth_mm':number('浅夹下探',0,8,4,1,' mm'),
         'test_lift_mm':number('试提高度',10,30,20,1,' mm'),
         'opening_mm':number('初始开口',12,40,24,1,' mm'),
         'grip_effort':integer('夹持力矩',100,1500,800)})
def grasp(ctx, ink, edge_offset_mm, pinch_depth_mm, test_lift_mm, opening_mm, grip_effort, approach_speed_m_s=.12, speed_m_s=.15):
    contact, inside, sample = edge_geometry(ctx.frame, ink, edge_offset_mm)
    yaw = jaw_yaw_near(np.degrees(np.arctan2(inside[1],inside[0])), ctx.preferred_yaw_deg)
    R = down_rotation(yaw)  # No base-dependent yaw or tilted side-grasp fallback for cloth.
    terms = grasp_terms(ctx.arm,ctx.model,ctx.backend.table_checks)
    target = contact + R[:,1]*terms['pinch_compensation_m'] - [0,0,pinch_depth_mm/1000]
    floor = terms['min_tcp_z_m']
    if floor is not None and target[2] < floor:
        raise ValueError('上层夹持点低于接触边界，请重新选上层自由边缘')
    points = [target+[0,0,.06],target,target+[0,0,test_lift_mm/1000]]
    events = [{'s':0.,'opening_mm':opening_mm,'pregrasp_open':True},
              {'s':.5,'opening_mm':0,'wait_for_arrival':True,'hold_s':.6,'ramp_s':.5,'effort':grip_effort}]
    # Thin cloth need not exceed the rigid-object minimum jaw opening. Visual verification follows.
    spec = checked_spec(ctx,points,[R]*3,events)
    meta = {'phase':'probe','arm':ctx.arm,'inside_xy':inside.tolist(),
            'contact_xyz':contact.tolist(),'probe_xyz':points[-1].tolist(),
            'observation_id':ctx.frame.id,'bag_open_verified':False}
    spec['bag_flap'] = meta
    return {'arm':spec,'bag_flap':meta,'sample_pixel':sample.tolist(),
            'jaw_yaw_deg':yaw,'pinch_depth_mm':pinch_depth_mm,
            'summary':'上层浅夹并试提后保持。重新观察上下层分离情况后才能继续掀开。'}


def plan_peel(ctx, probe_report, upper_pixel, lower_pixel, lift_mm=80., pull_mm=60., tilt_deg=30.):
    """Agent supplies fresh RGB-identified upper/lower lip points after the probe.

    The depth gate checks separation only; visual identity and lower-panel movement must also be checked
    by the caller. It is not a detector or a claim of successful packing.
    """
    meta = probe_report['bag_flap']
    if meta['arm'] != ctx.arm or meta['phase'] != 'probe':
        raise ValueError('需使用同一只臂的试提结果')
    if meta['observation_id'] == ctx.frame.id:
        raise ValueError('试提后需重新观察上层和下层，不能复用旧观测')
    if not np.isfinite([lift_mm,pull_mm,tilt_deg]).all() or not (20<=lift_mm<=120 and 0<=pull_mm<=100 and 0<=tilt_deg<=45):
        raise ValueError('掀起参数超出范围')
    upper = np.asarray(ctx.frame.pick(*upper_pixel,mode='surface'))
    lower = np.asarray(ctx.frame.pick(*lower_pixel,mode='surface'))
    if upper[2]-lower[2] < .012:
        raise ValueError('尚未看到上下层分离，请放回并改抓上层，不能继续整包抬起')
    h=ctx.current[ctx.arm]; p=np.asarray(h['xyz']); R=Rotation.from_euler('xyz',h['rpy_deg'],degrees=True).as_matrix()
    if np.linalg.norm(p-np.asarray(meta['probe_xyz'])) > .01:
        raise ValueError('试提后机械臂位置已改变，请重新定位')
    inside=np.r_[meta['inside_xy'],0.]; axis=np.cross(inside,[0,0,1])
    # Lift first; then peel toward the flap's hinge, rotating about the observed lip tangent.
    start=p+[0,0,lift_mm/1000]; points=[p,start]; rotations=[R,R]
    for t in np.linspace(0,1,5)[1:]:
        points.append(start+inside*pull_mm/1000*t)
        rotations.append(Rotation.from_rotvec(axis*np.deg2rad(tilt_deg)*t).as_matrix() @ R)
    spec=checked_spec(ctx,points,rotations,[])
    spec['bag_flap']={**meta,'phase':'peel','bag_open_verified':False}
    return {'arm':spec,'summary':'掀开后保持，待 RGB 确认主仓开口及投放区域；尚未确认开包成功。'}


def stage_metadata(plan):
    arms=(plan.get('input_draft') or {}).get('arms',{})
    marked=[(a,s['bag_flap']) for a,s in arms.items() if 'bag_flap' in s]
    if not marked:return None
    if len(marked)!=1 or len(arms)!=1:
        raise ValueError('试提/掀包阶段只执行持包臂')
    arm,meta=marked[0]
    if not isinstance(meta,dict) or meta.get('arm')!=arm or meta.get('phase') not in ('probe','peel','face_other','panel_lift'):
        raise ValueError('开包阶段元数据无效')
    return meta


def plan_face_other(ctx, target_xyz=None):
    """Turn the held flap toward the other arm at a caller-chosen, previewed holding point.

    Translation must be selected from the current bag geometry, never silently copied from a prior run.
    This planner does not execute or confirm that the compartment is open.
    """
    from scipy.spatial.transform import Slerp
    h=ctx.current[ctx.arm]; start=np.asarray(h['xyz'],float)
    target=start.copy() if target_xyz is None else np.asarray(target_xyz,float)
    if target.shape!=(3,) or not np.isfinite(target).all() or np.linalg.norm(target-start)>.18:
        raise ValueError('持包转向点需在当前位置 180 mm 内')
    other='right' if ctx.arm=='left' else 'left'
    direction=np.asarray(ctx.current[other]['xyz'],float)[:2]-target[:2]
    if np.linalg.norm(direction)<.08:
        raise ValueError('持包点过于接近另一只手，请重新选择')
    direction/=np.linalg.norm(direction)
    R=Rotation.from_euler('xyz',h['rpy_deg'],degrees=True).as_matrix()
    ts=np.linspace(0,1,13); points=[start+(target-start)*t for t in ts]
    for tilt in [45,30,60,20]:
        z=np.r_[direction*np.sin(np.deg2rad(tilt)),-np.cos(np.deg2rad(tilt))]
        x=np.cross([0,0,1],z);x/=np.linalg.norm(x)
        basis=np.column_stack([x,np.cross(z,x),z])
        for twist in [0,180,90,-90,45,-45,135,-135]:
            G=basis@Rotation.from_euler('z',twist,degrees=True).as_matrix()
            rotations=Slerp([0,1],Rotation.from_matrix([R,G]))(ts).as_matrix()
            try:spec=checked_spec(ctx,points,rotations,[],speed=.12)
            except ValueError:continue
            spec['bag_flap']={'phase':'face_other','arm':ctx.arm,'bag_open_verified':False}
            return {'arm':spec,'facing_arm':other,'forward_xy':G[:2,2].tolist(),
                    'summary':'保持夹持并转向另一只手；确认主仓开口后再装物。'}
    raise ValueError('转向另一只手的路径不可达，请调整持包点后重新规划')
