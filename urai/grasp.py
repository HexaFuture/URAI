"""Convert a two-finger grasp annotation to the shared trajectory schema."""
import numpy as np
from .transfer import GRIP_RAMP_S, grasp_orientation, leveled_cloud

# Thin cloth needs light support contact before the jaws close. Bound that contact independently of the
# operator's inset, so depth noise plus a 10 mm inset cannot drive the TCP far through the tabletop.
TABLE_PRESS_M = .003
# The physical pinch centre is offset from the model TCP by the finger and tool geometry. This is a
# tool-geometry compensation, not camera or base calibration, and must stay separate from it. The wrist
# tools are mirror-mounted, so the same signed correction moves the right pinch centre outward; hence the
# mirrored sign.
#
# The 10 mm magnitude was measured on the development platform for its fingers and tool mounts; re-measure
# it on your own platform, after the arm bases are calibrated, so that a base extrinsic error is not
# absorbed into this value.
PINCH_COMPENSATION_M = {'left': .010, 'right': -.010}


def grasp_terms(arm, model, table_checks):
    """The calibration a grasp line is planned with: fingertip floor, table height, pinch offset.

    Several callers need exactly these - the grasp endpoint, the skills and the task queue - and the
    numbers have to agree between them or the same stroke plans two different motions: a pinch offset
    applied by one caller and not another moves the answer by a degree of wrist and a millimetre of reach.
    """
    table_z_m = None if model is None else float(model.table_z_mm)/1000
    floor = None
    if model is not None:
        if table_checks and abs(float(model.t_world_base[2, 2])-1.) < 1e-9:
            # An upright calibrated base: keep a 1 mm margin above the fingertip floor.
            floor = float(model.t_world_base[2, 3])+(model.table_z_mm-model.fingertip_below_table_mm
                                                     -model.fingertip_bias_mm+1.)/1000
        elif not table_checks:
            floor = table_z_m-TABLE_PRESS_M
    return {'table_z_m': table_z_m, 'min_tcp_z_m': floor,
            'pinch_compensation_m': PINCH_COMPENSATION_M[arm]}


def grasp_from_line(frame, pixels, inset_mm=10., clearance_m=.12, speed_m_s=.8, endpoint_action='grasp',
                    *, preferred_yaw_deg=0., min_tcp_z_m=None, base_xy=None, grip_effort=None, table_z_m=None,
                    pinch_compensation_m=0., top_down=False, surface_xyz=None, reach_probe=None, seed_joints=None, pose_tolerance=(10.,3.)):
    if endpoint_action not in ('grasp','release','none'):
        raise ValueError('终点动作需为 grasp、release 或 none')
    points=np.asarray(pixels,dtype=float)
    if points.shape!=(2,2) or not np.isfinite(points).all() or (endpoint_action=='grasp' and np.linalg.norm(points[1]-points[0])<2):
        raise ValueError('请跨过物体画一条短线，线两端表示两指位置')
    inset_mm,clearance_m,speed_m_s=map(float,(inset_mm,clearance_m,speed_m_s))
    if not np.isfinite([inset_mm,clearance_m,speed_m_s]).all() or not 0<=inset_mm<=50 or not .02<=clearance_m<=.30 or not .001<=speed_m_s<=1.:
        raise ValueError('下探深度需为 0–50 mm，上方距离 0.02–0.30 m，速度 0.001–1.0 m/s')
    # The XY stays where the ray through the drawn midpoint meets the surface; only Z is re-levelled,
    # against the table actually fitted in this frame. Replacing the whole point with a fixed-plane
    # intersection would shift an angled-camera target sideways.
    #
    # The point is still taken as measured however low it reads. A midpoint is not refused for reading
    # below the table: such a check compares against one global constant while the same real table can
    # span 35 mm across a frame, and 42.6% of genuine table points tripped it in a measured frame.
    # Levelling is the part that helps - a local median instead of one pixel - and it refuses nothing.
    center=(frame.pick(*points.mean(axis=0),mode='surface') if surface_xyz is None
            else np.asarray(surface_xyz,dtype=float).copy())
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('抓取位置需为三个有限世界坐标')
    if table_z_m is not None and surface_xyz is None:
        try:
            leveled, _ = leveled_cloud(frame, float(table_z_m))
        except (ValueError, RuntimeError):
            pass        # no usable table fit in this frame: the single measured pixel is all there is
        else:
            u0, v0 = np.rint(points.mean(axis=0)).astype(int)
            patch = leveled[max(0, v0-2):v0+3, max(0, u0-2):u0+3]
            valid_z = patch[..., 2][np.isfinite(patch[..., 2])]
            if valid_z.size:
                center[2] = float(np.median(valid_z))
    if endpoint_action != 'grasp':
        # A held object needs a release location, not a new pinch axis or a post-grasp lift.
        # Keep the jaws closed through travel; the ordinary position-only planner checks
        # reachability/collisions and chooses a continuous wrist pose before opening.
        target=center+np.array([0.,0.,.02])
        hover=target+np.array([0.,0.,clearance_m])
        events=[]
        if endpoint_action=='release':
            event={'s':1.,'opening_mm':70,'hold_s':.6,'wait_for_arrival':True,'ramp_s':GRIP_RAMP_S}
            if grip_effort is not None:
                if not float(grip_effort).is_integer() or not 50<=grip_effort<=5000:
                    raise ValueError('夹持力矩需为 50–5000 的整数（单位 0.001 N·m）')
                event['effort']=int(grip_effort)
            events.append(event)
        return {'observation_id':frame.id,'surface_xyz':center.tolist(),
                'command_center_xyz':target.tolist(),'grasp_mode':'position_only',
                'release_clearance_mm':20.,'actual_inset_mm':0.,'table_limited':False,
                'width_mm':0.,'drawn_width_mm':0.,'width_clamped':False,
                'arm':{'grasp_line':False,'path':{'mode':'waypoints','points':[hover.tolist(),target.tolist()]},
                       'orientation':{'mode':'free'},'speed':speed_m_s,'start_hold_s':0.,
                       'gripper_events':events,'approach':{'speed_m_s':.8,'clearance_m':clearance_m}}}
    # Endpoints can lie on the background; intersect both rays with the
    # horizontal plane at the object's measured height to infer jaw direction.
    ends=np.array([frame.pick(*p,mode='plane',z=float(center[2])) for p in points])
    delta=ends[1]-ends[0];width=float(np.linalg.norm(delta[:2]))
    if width < .003:
        raise ValueError(f'抓取线对应宽度 {width*1000:.1f} mm；请画在两指抓取位置之间（至少 3 mm）')
    drawn_width=width
    width=min(width,.07)
    yaw=float(np.rad2deg(np.arctan2(delta[1],delta[0])))
    preferred_yaw_deg=float(preferred_yaw_deg)
    pinch_compensation_m=float(pinch_compensation_m)
    if (not np.isfinite([preferred_yaw_deg,pinch_compensation_m]).all()
            or abs(pinch_compensation_m)>.03
            or (min_tcp_z_m is not None and not np.isfinite(min_tcp_z_m))):
        raise ValueError('Grasp orientation and table height must be finite')
    # Parallel jaws define an unsigned closing axis. Reversing the stroke must
    # not demand a 180-degree wrist turn; prefer the equivalent current yaw.
    yaw=preferred_yaw_deg+(yaw-preferred_yaw_deg+90.)%180.-90.
    # The calibrated robot TCP is not the physical pinch centre: the finger
    # geometry puts the actual contact point about 10 mm toward the cuff
    # interior.  Apply this in the grasp frame so it follows the drawn jaw
    # direction, while keeping the measured surface XYZ available for logs.
    surface_center=center.copy()
    yaw_rad=np.deg2rad(yaw)
    compensation=np.array([np.sin(yaw_rad),-np.cos(yaw_rad),0.])*pinch_compensation_m
    center=center+compensation
    target=center-np.array([0,0,inset_mm/1000]);hover=center+[0,0,clearance_m]
    # Two real cases pull opposite ways, so this is a choice, not a default.
    #
    # Leaning toward the base is what a grasp line 0.6 m out needs: an upright tool only reaches about
    # 0.33 m from a PiPER base, and a straight-down line that far solved 8 mm off and 20 degrees tilted
    # (bread on a plate). Staying vertical is what a garment needs: a 40-degree approach puts one
    # finger down first and the pinch misses the fabric.
    #
    # Neither can be inferred from the picture, so the caller says which. Ordinary IK, joint and
    # collision checks apply to whichever pose comes out, and an unreachable vertical grasp is refused
    # there rather than quietly leaned into reach.
    pose=[180,0,yaw] if (top_down or base_xy is None) else grasp_orientation(target,yaw,base_xy)
    table_limited=min_tcp_z_m is not None and target[2]<min_tcp_z_m
    if table_limited:
        if min_tcp_z_m>center[2]+1e-9:
            # A thin surface may level a fraction below the conservative table
            # plane.  Raising it to the bounded contact floor is safe; a floor
            # above the calibrated table remains a configuration error.
            if table_z_m is None or min_tcp_z_m>table_z_m+1e-9:
                raise ValueError('观测表面低于桌面允许边界，请检查深度和桌面标定')
            center[2]=min_tcp_z_m
        target[2]=min_tcp_z_m
    solution = None
    if endpoint_action=='grasp' and reach_probe is not None:
        from .grasp_solver import solve_line_grasp
        # The contact center stays anchored to the annotation. Rotate the calibrated
        # tool-local pinch correction with the new pose instead of shifting fingertips.
        contact = target-compensation
        solution = solve_line_grasp(contact,yaw,base_xy,clearance_m,pinch_compensation_m,
                                    reach_probe,seed_joints,pose_tolerance,min_tcp_z_m)
        target,hover,pose=solution['target'],solution['pre'],solution['pose']
        compensation=solution['rotation'][:,1]*pinch_compensation_m
        center=surface_center+compensation
    events = [{'s':0,'opening_mm':70,'pregrasp_open':True}] if endpoint_action=='grasp' else []
    if endpoint_action!='none':
        # The jaws are walked to their target over half a second instead of snapping shut: closing a soft
        # object in one jump at the block torque squeezed a bread half straight out of the fingers
        # (measured 0.1 mm after the close). The torque limit is what actually stops the jaws.
        end={'s':1,'opening_mm':0 if endpoint_action=='grasp' else 70,
             'hold_s':.6,'wait_for_arrival':True,'ramp_s':GRIP_RAMP_S}
        if grip_effort is not None:
            if not float(grip_effort).is_integer() or not 50<=grip_effort<=5000:
                raise ValueError('夹持力矩需为 50–5000 的整数（单位 0.001 N·m）')
            end['effort']=int(grip_effort)
        events.append(end)
    return {'observation_id':frame.id,'surface_xyz':surface_center.tolist(),
            **({} if solution is None else {'grasp_mode':solution['mode'],'grasp_tilt_deg':solution['tilt_deg'],
                'lift_xyz':solution['lift'].tolist(),'grasp_solver_candidates':solution['candidates']}),
            'command_center_xyz':center.tolist(),
            'pinch_compensation_mm':(compensation*1000).tolist(),'width_mm':width*1000,'drawn_width_mm':drawn_width*1000,'width_clamped':drawn_width>.07,
            'grip_effort':None if grip_effort is None else int(grip_effort),
            'actual_inset_mm':float((center[2]-target[2])*1000),'table_limited':bool(table_limited),
            'arm':{'grasp_line':True,'path':{'mode':'waypoints','points':[hover.tolist(),target.tolist()]},
                   'orientation':{'mode':'keyframes','points':[[0,*pose],[1,*pose]]},
                   'speed':speed_m_s,'start_hold_s':.6 if endpoint_action=='grasp' else 0.,
                   'gripper_events':events,
                   'approach':{'speed_m_s':.8,'clearance_m':clearance_m}}}


def line_object(frame, pixels, **kwargs):
    """Use the operator's line directly; no raised-object segmentation."""
    result = grasp_from_line(frame, pixels, **kwargs)
    center = np.asarray(result['surface_xyz'])
    ends = np.array([frame.pick(*p, mode='plane', z=float(center[2])) for p in pixels])
    axis = ends[1,:2]-ends[0,:2]
    axis /= np.linalg.norm(axis)
    return {'center': center[:2], 'top': float(center[2]),
            'width': result['width_mm']/1000, 'points': ends,
            'axis_deg': float((np.degrees(np.arctan2(axis[1],axis[0]))+90.)%180.-90.),
            'line_world': ends.tolist(),
            'manual_grasp': np.asarray(result['arm']['path']['points'][-1]),
            'manual_hover': np.asarray(result['arm']['path']['points'][0])}
