"""Fresh visual landmarks -> bounded pour actions. No model or motion on import."""
import copy
import json
import time
import uuid
import warnings
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from .pour_alignment import fit_rim, plan_step
from .pour_rgb import rgb_rim, triangulated_rim

PRECISION = {'position_mm': 1., 'orientation_deg': .5}

def median_depth_frames(depths):
    values=np.stack(depths).astype(float)
    valid=np.isfinite(values)&(values>0)
    masked=np.where(valid,values,np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        result=np.nanmedian(masked,axis=0)
        spread=np.nanmax(masked,axis=0)-np.nanmin(masked,axis=0)
    supported=(valid.sum(axis=0)>=len(depths)//2+1)&(spread<=.012)
    return np.where(supported,result,0.)

def plain(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def number(data,name,default,low,high):
    value=float(data.get(name,default))
    if not np.isfinite(value) or not low<=value<=high:raise ValueError(f'{name} must be {low}..{high}')
    return value

def same_pose(a,b):
    return (np.linalg.norm(np.asarray(a['xyz'])-b['xyz'])<=.002 and
            np.max(np.abs(np.asarray(a['joints_deg'])-b['joints_deg']))<=.5 and
            (Rotation.from_euler('xyz',a['rpy_deg'],degrees=True).inv()*Rotation.from_euler('xyz',b['rpy_deg'],degrees=True)).magnitude()<=np.deg2rad(1))

def held_opening(state):
    opening=float(state.get('gripper_mm',float('nan')))
    if not np.isfinite(opening) or not 1.<opening<69.:
        raise ValueError('Held bottle requires finite gripper feedback between closed and fully open')
    return opening

def rim_from_pixels(frame,pixels,hint):
    p=np.asarray(pixels,float)
    if p.ndim!=2 or p.shape[1]!=2 or not 4<=len(p)<=16 or not np.isfinite(p).all():
        raise ValueError('Provide 4..16 visible rim pixels, not a transparent opening center')
    # Depth must come from the annotated material, not arbitrary table-plane fallback.
    points=[]
    for u,v in p:
        if not (0<=u<frame.depth.shape[1] and 0<=v<frame.depth.shape[0]):raise ValueError('Rim pixel outside frame')
        x,y=int(u),int(v);patch=frame.depth[max(0,y-2):y+3,max(0,x-2):x+3]
        valid=patch[np.isfinite(patch)&(patch>0)]
        if len(valid)<max(3,patch.size//2) or np.ptp(np.quantile(valid,[.1,.9]))>.012:
            raise ValueError('Rim depth is missing or crosses foreground/background; re-observe')
        points.append(frame.pick(u,v,mode='surface'))
    return fit_rim(points,normal_hint=hint)

def horizontal_target_rim(frame,pixels,table_z):
    """A visually horizontal metal rim: fit its RGB contour at a conservative depth bound.

    This is opt-in for an upright receiving vessel, never for the moving bottle.
    Report the height interval separately; the circle residual is not a 3D depth accuracy claim.
    """
    p=np.asarray(pixels,float)
    if p.ndim!=2 or p.shape[1]!=2 or not 6<=len(p)<=16 or not np.isfinite(p).all():
        raise ValueError('Horizontal receiver needs 6..16 material-rim pixels')
    if len(np.unique(p,axis=0))!=len(p):raise ValueError('Receiver landmarks must be distinct')
    measured_pixels=[]
    for u,v in p:
        if not (2<=u<frame.depth.shape[1]-2 and 2<=v<frame.depth.shape[0]-2):
            raise ValueError('Receiver rim pixel outside frame')
        patch=frame.depth[int(v)-2:int(v)+3,int(u)-2:int(u)+3]
        if np.count_nonzero(np.isfinite(patch)&(patch>0))>=20:
            measured_pixels.append((u,v))
    if len(measured_pixels)<6:raise ValueError('Receiver rim has insufficient measured depth')
    measured=np.array([frame.pick(u,v,mode='surface') for u,v in measured_pixels])
    if not np.isfinite(measured).all():raise ValueError('Receiver depth is not finite')
    z=measured[:,2]
    if z.min()<table_z+.005 or z.max()>table_z+.25 or np.ptp(z)>.05:
        raise ValueError('Receiver depth range does not support a tabletop horizontal rim')
    bounds=[float(z.min()-.010),float(z.max()+.010)]
    fits=[fit_rim([frame.pick(u,v,mode='plane',z=h) for u,v in p],[0,0,1]) for h in bounds]
    uncertainty=float(np.linalg.norm(fits[1]['center'][:2]-fits[0]['center'][:2]))
    usable_radius=min(f['radius'] for f in fits)-uncertainty
    if usable_radius<.030:
        raise ValueError('Receiver opening is too small for its depth uncertainty')
    # Use only the intersection guaranteed to be inside the opening at every
    # measured height, instead of treating the projected center as exact.
    conservative={**fits[1],'radius':usable_radius}
    return conservative,{'method':'horizontal_rgb_contour_with_depth_bound',
                    'height_interval_m':bounds,'center_xy_uncertainty_m':uncertainty,
                    'physical_radius_m':float(fits[1]['radius']),'usable_radius_m':float(usable_radius),
                    'circle_residual_m':float(fits[1]['residual_m'])}

def verify_compiled_mouth(spec,state,source,target,options):
    """Validate the actual PCHIP/SLERP samples, not just their waypoints."""
    from .trajectory import compile_arm
    from .pour_alignment import lowest_lip
    c=compile_arm(spec,state);R0=Rotation.from_euler('xyz',state['rpy_deg'],degrees=True).as_matrix()
    m=R0.T@(np.asarray(source['center'])-state['xyz']);axis=R0.T@source['normal']
    rotations=c['rotation_matrices'];centers=c['xyz']+np.einsum('nij,j->ni',rotations,m)
    normals=np.einsum('nij,j->ni',rotations,axis);direction=np.asarray(options['pour_direction_xy'])
    top=target['center'][2]+target['radius']*np.sqrt(max(0.,1-target['normal'][2]**2))
    worst=0.
    for center,normal in zip(centers,normals):
        if np.linalg.norm(normal[:2])<1e-7:lip=center+np.r_[direction*source['radius'],0.]
        else:lip=lowest_lip(center,normal,source['radius'])
        error=float(np.linalg.norm(lip[:2]-target['center'][:2]));worst=max(worst,error)
        body_low=center[2]-options['bottle_length_m']*max(normal[2],0)-options['bottle_radius_m']*np.sqrt(max(0.,1-normal[2]**2))
        if error>.004 or lip[2]<top+options['clearance_m']-.001 or body_low<top+.018:
            raise ValueError('Compiled pour interpolation leaves the observed mouth/clearance bounds')
    return {'max_lip_xy_error_m':worst,'samples':len(centers)}

class PourAgent:
    def __init__(self,service):
        self.service=service;self.session=None;self.observation=None;self.pending=None;self.rgb_reference=None

    def status(self):
        if self.session is None:return None
        s=self.session
        return {k:plain(s[k]) for k in ('id','arm','state','steps','goal_tilt_deg','inspection_tilt_deg','artifact_dir')}

    def record(self,event):
        if self.session is None:return
        directory=Path(self.session['artifact_dir']);directory.mkdir(parents=True,exist_ok=True)
        with (directory/'events.jsonl').open('a') as f:f.write(json.dumps(plain({'time':time.time(),**event}),ensure_ascii=False)+'\n')

    def start(self,data):
        self.service._editable()
        arm=data.get('arm','left')
        if arm not in ('left','right'):raise ValueError('Choose left or right')
        st=copy.deepcopy(self.service.backend.state()[arm]);held_opening(st);identifier=uuid.uuid4().hex
        root=self.service.log_dir or Path('/tmp/urai-pour-agent')
        self.session={'id':identifier,'arm':arm,'state':'observe','steps':0,'initial':st,'history':[st],'undo':[],
          'goal_tilt_deg':number(data,'tilt_deg',130,60,150),'inspection_tilt_deg':number(data,'inspection_tilt_deg',20,0,30),
          'bottle_length_m':number(data,'bottle_length_m',.35,.05,.45),'bottle_radius_m':number(data,'bottle_radius_m',.06,.01,.10),
          'clearance_m':number(data,'mouth_clearance_m',.04,.03,.10),'speed':number(data,'speed_m_s',.8,.001,1.),
          'hold_s':number(data,'hold_s',.5,.1,1.),'direction':data.get('pour_direction_xy'),
          'epoch':self.service.cancel_epoch,'artifact_dir':str(root/('pour-agent-'+identifier)),
          'normal_local':None,'created':time.monotonic(),'distance':0.}
        if self.session['direction'] is not None:
            d=np.asarray(self.session['direction'],float)
            if d.shape!=(2,) or not np.isfinite(d).all() or np.linalg.norm(d)<1e-6:raise ValueError('Invalid pour direction')
            self.session['direction']=(d/np.linalg.norm(d)).tolist()
        self.observation=None;self.pending=None;self.rgb_reference=None;self.record({'event':'start','session':self.status(),'initial':st})
        return {**self.status(),'next_tool':'POST /api/pour-agent/observe','holding_required':True}

    def require(self,data=None,*,recovery=False):
        s=self.session
        if s is None or (data and data.get('session_id')!=s['id']):raise ValueError('Missing or changed pour session')
        if self.service.cancel_epoch!=s['epoch']:raise ValueError('Pour session cancelled; start a new session')
        if not recovery and (s['steps']>=80 or time.monotonic()-s['created']>1800):raise ValueError('Pour loop budget exhausted; holding position')
        return s

    def observe(self,data,*,recovery=False):
        s=self.require(data,recovery=recovery);self.service._editable();arm=s['arm']
        count=data.get('depth_frames',1)
        if type(count) is not int or count not in (1,3):raise ValueError('depth_frames must be 1 or 3')
        before=copy.deepcopy(self.service.backend.state()[arm]);depths=[];exposures=[]
        for index in range(count):
            if depths:time.sleep(.05)
            if index==count-1:
                picture=self.service.observe();f=self.service.frame
            else:
                f=self.service.backend.capture()
            depths.append(f.depth.copy());exposures.append((f.sensor_sequence,f.sensor_timestamp))
        after=copy.deepcopy(self.service.backend.state()[arm]);frame=self.service.frame
        if not same_pose(before,after):raise ValueError('Capture a settled bottle before locating its rim')
        stamp=getattr(frame,'sensor_timestamp',None);seq=getattr(frame,'sensor_sequence',None)
        if self.service.backend.mode=='real':
            if stamp is None or seq is None or not 0<=time.monotonic()-stamp<=1.:
                raise ValueError('Missing or stale camera exposure metadata')
            if self.observation and self.observation.get('sensor_sequence')==seq:raise ValueError('Camera exposure was reused')
            if count>1 and (len({v[0] for v in exposures})!=count or
                    any(v[1] is None or not 0<=time.monotonic()-v[1]<=1. for v in exposures)):
                raise ValueError('Temporal depth needs distinct exposures less than one second old')
        if count>1:frame.depth=median_depth_frames(depths)
        token=uuid.uuid4().hex
        self.observation={'token':token,'frame_id':frame.id,'state':after,'created':time.monotonic(),
                          'sensor_sequence':seq,'depth_exposures':exposures,'used':False}
        self.pending=None;s['state']='awaiting_landmarks'
        folder=Path(s['artifact_dir']);folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(frame.rgb).save(folder/(token+'.jpg'))
        np.savez_compressed(folder/(token+'.npz'),depth=frame.depth,k=frame.k,t_world_camera=frame.t)
        self.record({'event':'observe',**self.observation})
        return {'session':self.status(),'observation_token':token,'observation_id':frame.id,
                'image':picture['image'],'width':picture['width'],'height':picture['height'],'tcp':after,
                'landmark_schema':{'source_rim_pixels':'4..16 visible bottle-mouth rim [u,v] pixels',
                                   'target_rim_pixels':'4..16 receiving-cup INNER rim [u,v] pixels','confidence':'0..1'},
                'next_tool':'POST /api/pour-agent/plan'}

    def check_observation(self,data,*,recovery=False):
        s=self.require(data,recovery=recovery);o=self.observation
        if not o or o['used'] or data.get('observation_token')!=o['token'] or self.service.frame.id!=o['frame_id']:
            raise ValueError('Fresh rim observation required after every executed motion')
        if time.monotonic()-o['created']>90:raise ValueError('Rim observation expired')
        current=self.service.backend.state()[s['arm']]
        if not same_pose(current,o['state']):raise ValueError('Bottle moved after rim observation')
        if abs(held_opening(current)-held_opening(o['state']))>3.:raise ValueError('Grip changed after rim observation; locate the mouth again')
        return s,o,current

    def plan(self,data):
        self.service._editable();s,o,st=self.check_observation(data)
        confidence=number(data,'confidence',0,0,1)
        if confidence<.8:raise ValueError('Rim confidence too low; request another view')
        R=Rotation.from_euler('xyz',st['rpy_deg'],degrees=True).as_matrix()
        hint=R@np.asarray(s['normal_local']) if s['normal_local'] is not None else np.array([0,0,1.])
        source_evidence=None;target_evidence=None
        if data.get('perception')=='rgb_probe':
            pixels=np.asarray(data.get('source_rim_pixels'),float)
            if pixels.ndim!=2 or pixels.shape[1]!=2 or not 6<=len(pixels)<=16 or not np.isfinite(pixels).all():raise ValueError('RGB probe needs visible bottle rim landmarks')
            shift=np.asarray(data.get('translation_m'),float)
            if shift.shape!=(3,) or not np.isfinite(shift).all() or not .010<=np.linalg.norm(shift)<=.015001:raise ValueError('RGB probe translation must be 10..15 mm')
            self.rgb_reference={'state':copy.deepcopy(st),'pixels':pixels,'token':o['token']}
            spec={'path':{'mode':'waypoints','points':[st['xyz'],(np.asarray(st['xyz'])+shift).tolist()]},'orientation':{'mode':'hold'},'speed':min(s['speed'],.04),'angular_speed_deg_s':8.,'gripper_events':[],'approach':{'speed_m_s':.04,'clearance_m':.03}}
            return self.commit(spec,'align',o,st,{'perception':'rgb_probe'})
        if data.get('perception')=='rgb_parallax':
            ref=self.rgb_reference
            if ref is None or data.get('reference_token')!=ref['token']:raise ValueError('RGB parallax needs its executed probe reference')
            if (R.T@Rotation.from_euler('xyz',ref['state']['rpy_deg'],degrees=True).as_matrix()).trace()<1+2*np.cos(np.deg2rad(.5)):raise ValueError('RGB parallax probe rotated the bottle')
            source,source_evidence=triangulated_rim(self.service.frame,ref['pixels'],data.get('source_rim_pixels'),np.asarray(st['xyz'])-ref['state']['xyz'],hint)
            target,target_evidence=rgb_rim(self.service.frame,data.get('target_rim_pixels'),number(data,'target_radius_m',0,.008,.10),[0,0,1],horizontal=True)
        elif data.get('perception')=='rgb':
            source,source_evidence=rgb_rim(self.service.frame,data.get('source_rim_pixels'),number(data,'source_radius_m',0,.008,.10),hint,fixed_normal=s['normal_local'] is not None)
            target,target_evidence=rgb_rim(self.service.frame,data.get('target_rim_pixels'),number(data,'target_radius_m',0,.008,.10),[0,0,1],horizontal=True)
        else:
            source=rim_from_pixels(self.service.frame,data.get('source_rim_pixels'),hint)
        if data.get('perception') in ('rgb','rgb_parallax'):pass
        elif data.get('target_rim_horizontal') is True:
            model=self.service.backend.models[s['arm']]
            target,target_evidence=horizontal_target_rim(self.service.frame,data.get('target_rim_pixels'),model.table_z_mm/1000)
        else:
            target=rim_from_pixels(self.service.frame,data.get('target_rim_pixels'),[0,0,1])
        offset=np.linalg.norm(source['center']-st['xyz'])
        if not .015<=offset<=.5:raise ValueError('Observed mouth is inconsistent with held-object reach')
        if s['normal_local'] is None and source['normal'][2]<np.cos(np.deg2rad(35)):raise ValueError('Begin the pour loop with the bottle upright')
        if s['normal_local'] is None and np.dot(source['normal'],source['center']-st['xyz'])<=0:
            raise ValueError('Initial bottle mouth must lie above the grip along the bottle axis; reject background depth')
        if s['direction'] is None:
            d=source['normal'][:2] if np.linalg.norm(source['normal'][:2])>1e-6 else np.asarray(target['center'][:2])-source['center'][:2]
            if np.linalg.norm(d)<1e-6:raise ValueError('Provide pour_direction_xy for an already centered bottle')
            s['direction']=(d/np.linalg.norm(d)).tolist()
        kwargs=dict(clearance_m=s['clearance_m'],bottle_length_m=s['bottle_length_m'],bottle_radius_m=s['bottle_radius_m'],pour_direction_xy=s['direction'])
        step=plan_step(st['xyz'],st['rpy_deg'],source,target,mode='align',tolerance_m=.001,**kwargs)
        mode='align';events=[]
        if step['aligned']:
            tilt=float(step['tilt_deg'])
            if tilt<s['inspection_tilt_deg']-.25:
                mode='inspect';step=plan_step(st['xyz'],st['rpy_deg'],source,target,mode='tilt',tilt_step_deg=min(5.,s['inspection_tilt_deg']-tilt),**kwargs)
            else:
                # The model is never awaited at the flowing pose: one checked
                # transaction pours briefly, then retraces to the inspection pose.
                mode='pour';forward=[list(st['xyz'])];rotations=[list(st['rpy_deg'])];virtual=copy.deepcopy(source)
                p=st['xyz'];r=st['rpy_deg'];angle=tilt
                for _ in range(60):
                    if angle>=s['goal_tilt_deg']-.05:break
                    delta=min(5.,s['goal_tilt_deg']-angle)
                    if angle<90.-1e-5:delta=min(delta,90.-angle)
                    nxt=plan_step(p,r,virtual,target,mode='tilt',tilt_step_deg=delta,**kwargs)
                    if nxt['delta_tilt_deg']<.01:raise ValueError('Cannot advance the compensated pour arc')
                    # Preserve the dense arc around the horizontal-body clearance kink.
                    # Elsewhere bounded step endpoints already meet the compiled 4 mm bound.
                    if angle<=100. and float(nxt['tilt_deg'])>=80.:
                        for pp,kk in zip(nxt['path']['points'][1:],nxt['orientation']['points'][1:]):
                            if np.linalg.norm(np.asarray(pp)-forward[-1])>1e-9 or (Rotation.from_euler('xyz',kk[1:],degrees=True).inv()*Rotation.from_euler('xyz',rotations[-1],degrees=True)).magnitude()>1e-9:
                                forward.append(plain(pp));rotations.append(plain(kk[1:]))
                    else:
                        forward.append(plain(nxt['xyz']));rotations.append(plain(nxt['rpy_deg']))
                    p=plain(nxt['xyz']);r=plain(nxt['rpy_deg']);angle=float(nxt['tilt_deg'])
                    virtual={**virtual,'center':np.asarray(nxt['source_center']),'normal':np.asarray(nxt['source_normal'])}
                else:raise ValueError('Pour arc exceeds waypoint budget')
                if angle<s['goal_tilt_deg']-.05:raise ValueError('Pour angle not reached')
                pts=forward+forward[-2::-1];rr=rotations+rotations[-2::-1]
                if len(pts)>128:raise ValueError('Pour arc exceeds waypoint budget')
                step={**step,'path':{'mode':'waypoints','points':pts},'orientation':{'mode':'keyframes','points':[[i/(len(pts)-1),*r] for i,r in enumerate(rr)]}}
                events=[{'s':.5,'hold_s':s['hold_s'],'wait_for_arrival':True}]
        s['normal_local']=(R.T@source['normal']).tolist()
        self.record({'event':'landmarks','observation_token':o['token'],'source':source,'target':target,'target_evidence':target_evidence,'source_evidence':source_evidence,'input':data,'planned':step,'mode':mode})
        spec={'path':plain(step['path']),'orientation':plain(step['orientation']),'speed':s['speed'],
              'angular_speed_deg_s':8.,'gripper_events':events,'approach':{'speed_m_s':s['speed'],'clearance_m':.03}}
        if mode in ('inspect','pour'):
            verify_compiled_mouth(spec,st,source,target,kwargs)
        return self.commit(spec,mode,o,st,{'alignment_error_mm':float(step['error_m'])*1000,'tilt_deg':float(step['tilt_deg']),'source_evidence':source_evidence,'target_evidence':target_evidence})

    def commit(self,spec,mode,o,st,report):
        s=self.session;nonce=uuid.uuid4().hex
        spec['pour_agent']={'phase':mode,'arm':s['arm'],'session_id':s['id'],'observation_token':o['token'],'nonce':nonce}
        distance=float(np.linalg.norm(np.diff(np.asarray(spec['path']['points']),axis=0),axis=1).sum())
        if mode!='return' and s['distance']+distance>2.:raise ValueError('Cumulative pour correction budget exceeded')
        self.pending={'spec':copy.deepcopy(spec),'nonce':nonce,'mode':mode,'distance':distance,'start':copy.deepcopy(st)}
        d=self.service.set_draft({'observation_id':o['frame_id'],'arms':{s['arm']:spec}})
        s['state']='ready';self.record({'event':'planned','mode':mode,'revision':d['revision'],'report':report})
        return {'session':self.status(),'mode':mode,'revision':d['revision'],**report,'next_tool':'POST /api/preview then /api/execute; re-observe afterward'}

    def return_upright(self,data):
        s=self.require(data,recovery=True);self.service._editable()
        if s['state'] not in ('poured','holding','awaiting_landmarks','ready'):raise ValueError('Return is available after pouring or a completed correction')
        self.observe(data,recovery=True);st=self.observation['state']
        boundary=s['history'][-1]
        if not same_pose(st,boundary) or abs(held_opening(st)-held_opening(boundary))>3.:
            raise ValueError('Arm or grip moved outside the pour session; cannot replay an old return path')
        # Reverse one executed correction at a time. Reversing its original PCHIP
        # knots preserves the path; rebuilding from endpoint history rounds corners.
        if s['undo']:
            spec=copy.deepcopy(s['undo'][-1])
            spec.pop('pour_agent',None)
            spec['path']['points'].reverse()
            keys=spec['orientation']['points']
            spec['orientation']['points']=[[1-k[0],*k[1:]] for k in reversed(keys)]
            spec['gripper_events']=[]
        else:
            spec={'path':{'mode':'waypoints','points':[st['xyz'],st['xyz']]},
                  'orientation':{'mode':'hold'},'speed':s['speed'],'angular_speed_deg_s':8.,'gripper_events':[],
                  'approach':{'speed_m_s':s['speed'],'clearance_m':.03}}
        result=self.commit(spec,'return',self.observation,st,{})
        result['return_steps_remaining']=max(0,len(s['undo'])-1)
        return result

    def check_grip(self,tag):
        if tag['phase']=='prepare':return
        current=self.service.backend.state()[tag['arm']]
        if self.pending is None or abs(held_opening(current)-held_opening(self.pending['start']))>8.:
            raise ValueError('Gripper changed during pour; hold position and reacquire bottle mouth')

    def validate(self,plan,current):
        arms=(plan.get('input_draft') or {}).get('arms',{})
        tagged=[(a,p) for a,p in arms.items() if 'pour_agent' in p]
        if not tagged:return None
        if len(tagged)!=1 or len(arms)!=1:raise ValueError('Pour alignment is a single-arm held-object action')
        arm,spec=tagged[0];tag=spec['pour_agent']
        if tag.get('arm')!=arm:raise ValueError('Invalid pour arm')
        if tag.get('phase')=='prepare':return tag
        s,o,st=self.check_observation({'session_id':tag.get('session_id'),'observation_token':tag.get('observation_token')},recovery=tag.get('phase')=='return')
        if not self.pending or spec!=self.pending['spec'] or tag.get('nonce')!=self.pending['nonce']:
            raise ValueError('Pour draft changed; rebuild from fresh landmarks')
        if any('opening_mm' in e for e in spec.get('gripper_events',[])):raise ValueError('Pour alignment must preserve grip')
        return tag

    def completed(self,tag):
        if tag['phase']=='prepare':return
        s=self.session;pending=self.pending
        st=copy.deepcopy(self.service.backend.state()[s['arm']]);s['history'].append(st)
        if tag['phase']=='return':
            if s['undo']:s['undo'].pop()
        elif tag['phase'] in ('align','inspect'):s['undo'].append(copy.deepcopy(pending['spec']))
        s['steps']+=1;s['distance']+=pending['distance'];self.observation['used']=True;self.pending=None
        s['state']='returned' if tag['phase']=='return' and not s['undo'] else 'poured' if tag['phase']=='pour' else 'holding'
        self.record({'event':'completed','phase':tag['phase'],'state':st,'session':self.status()})

    def failed(self,error):
        if self.session:
            self.session['state']='error';self.pending=None
            if self.observation:self.observation['used']=True
            self.record({'event':'error','error':str(error)})
