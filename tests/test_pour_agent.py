import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from conftest import make_client
from urai.backend import SimulationBackend
from urai.service import Service
from urai.geometry import Frame
from urai.pour_agent import rim_from_pixels


def configured(tmp_path,monkeypatch,inspection=20):
    s=Service(SimulationBackend(),log_dir=tmp_path)
    s.backend.positions['left']=np.array([.30,0,.20]);s.backend.openings['left']=42.
    c=make_client(s)
    start=c.post('/api/pour-agent/start',json={'arm':'left','pour_direction_xy':[1,0],'inspection_tilt_deg':inspection}).json()
    o=c.post('/api/pour-agent/observe',json={'session_id':start['id']}).json()
    source={'center':np.array([.28,0,.49]),'normal':np.array([0.,0.,1.]),'radius':.02,'residual_m':0.}
    target={'center':np.array([.30,0,.12]),'normal':np.array([0.,0.,1.]),'radius':.04,'residual_m':0.}
    def rims(frame,pixels,hint):return copy.deepcopy(source if pixels[0][0]==1 else target)
    monkeypatch.setattr('urai.pour_agent.rim_from_pixels',rims)
    data={'session_id':start['id'],'observation_token':o['observation_token'],
          'source_rim_pixels':[[1,1]]*6,'target_rim_pixels':[[2,2]]*6,'confidence':.95}
    return s,c,data,source,target


def test_agent_plan_is_not_motion_and_never_opens_gripper(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    before=copy.deepcopy(s.backend.state());r=c.post('/api/pour-agent/plan',json=d)
    assert r.status_code==200,r.text
    assert r.json()['mode']=='inspect'
    assert s.backend.state()==before
    spec=s.draft['arms']['left']
    assert not any('opening_mm' in e for e in spec['gripper_events'])
    assert len(spec['path']['points'])>2


def test_unexecuted_alignment_cannot_advance_tilt(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch);src['center'][0]-=.05
    r=c.post('/api/pour-agent/plan',json=d)
    assert r.status_code==200,r.text
    assert r.json()['mode']=='align'
    assert np.linalg.norm(np.array(s.draft['arms']['left']['path']['points'][-1])-s.backend.positions['left'])<=.015001


def test_executed_step_preserves_grip_no_home_and_consumes_observation(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    r=c.post('/api/pour-agent/plan',json=d);assert r.status_code==200,r.text
    p=s.preview(r.json()['revision']);s.execute(p['id']);s.worker.join(10)
    assert s.execution['state']=='completed',s.execution
    assert 'return_home' not in s.execution['result']
    assert s.backend.openings['left']==42.
    assert c.post('/api/pour-agent/plan',json=d).status_code==409
    assert s.pour_agent.status()['steps']==1


def test_burst_returns_to_inspection_pose_without_model_wait(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch,inspection=0)
    r=c.post('/api/pour-agent/plan',json=d);assert r.status_code==200,r.text
    assert r.json()['mode']=='pour'
    a=s.draft['arms']['left'];pts=np.array(a['path']['points']);keys=np.array(a['orientation']['points'])
    np.testing.assert_allclose(pts[0],pts[-1]);np.testing.assert_allclose(keys[0,1:],keys[-1,1:])
    assert 3<len(pts)<=128 and len(a['gripper_events'])==1
    assert a['gripper_events'][0]['s']==.5
    p=s.preview(r.json()['revision'])
    def immediate(plan,start,cancel,progress):
        from scipy.spatial.transform import Rotation
        for arm,curve in plan['arms'].items():
            s.backend.positions[arm]=curve['xyz'][-1].copy()
            s.backend.rotations[arm]=Rotation.from_matrix(curve['rotation_matrices'][-1]).as_euler('xyz',degrees=True).tolist()
        progress(1.,plan['duration_s']);return {'completed':True,'mode':'simulation'}
    monkeypatch.setattr(s.backend,'execute',immediate)
    s.execute(p['id']);s.worker.join(10)
    assert s.execution['state']=='completed',s.execution
    assert s.pour_agent.status()['state']=='poured'
    ret=c.post('/api/pour-agent/return',json={'session_id':d['session_id']});assert ret.status_code==200,ret.text
    p=s.preview(ret.json()['revision']);s.execute(p['id']);s.worker.join(10)
    assert s.execution['state']=='completed'
    assert s.pour_agent.status()['state']=='returned'
    assert s.backend.openings['left']==42.


@pytest.mark.parametrize('change',['expired','token','frame','pose','confidence'])
def test_invalid_landmarks_never_create_motion(tmp_path,monkeypatch,change):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    if change=='expired':s.pour_agent.observation['created']-=100
    if change=='token':d['observation_token']='old'
    if change=='frame':s.observe()
    if change=='pose':s.backend.positions['left'][0]+=.01
    if change=='confidence':d['confidence']=.2
    r=c.post('/api/pour-agent/plan',json=d);assert r.status_code==409,r.text
    assert s.execution['state']!='running'


def test_preview_does_not_rebind_stale_rim_geometry(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    r=c.post('/api/pour-agent/plan',json=d);s.backend.positions['left'][0]+=.01
    with pytest.raises(ValueError,match='rebound'):s.preview(r.json()['revision'])


def test_draft_tampering_rejected_before_preview(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    r=c.post('/api/pour-agent/plan',json=d)
    s.draft['arms']['left']['path']['points'][-1][0]+=.1
    with pytest.raises(ValueError,match='draft changed'):s.preview(r.json()['revision'])


def test_cancel_invalidates_session_even_after_recover(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch);s.cancel();s.recover()
    assert c.post('/api/pour-agent/observe',json={'session_id':d['session_id']}).status_code==409


def test_rim_depth_uses_world_transform_not_transparent_center():
    k=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    t=np.eye(4);t[:3,:3]=np.diag([1,-1,-1]);t[:3,3]=[0,0,1]
    f=Frame(np.zeros((480,640,3),dtype=np.uint8),np.full((480,640),np.nan),k,t)
    ring=np.array([[.1+.03*np.cos(a),.04+.03*np.sin(a),.35] for a in np.linspace(0,2*np.pi,8,endpoint=False)])
    px=f.project(ring)
    for u,v in px:f.depth[int(v)-2:int(v)+3,int(u)-2:int(u)+3]=.65
    rim=rim_from_pixels(f,px,[0,0,1]);np.testing.assert_allclose(rim['center'],[.1,.04,.35],atol=1e-6)
    assert rim['radius']==pytest.approx(.03)
    f.depth[:]=np.nan
    with pytest.raises(ValueError,match='depth'):rim_from_pixels(f,px,[0,0,1])


def test_real_capture_requires_actual_exposure_metadata(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch);s.backend.mode='real'
    r=c.post('/api/pour-agent/observe',json={'session_id':d['session_id']});assert r.status_code==409
    assert 'exposure' in r.text


def test_prepare_skill_registered_and_does_not_plan_a_pour(tmp_path):
    from test_skills import scene_service,CUP_PIXEL
    from urai.skills import plan,skill_context
    s=scene_service();result=plan('pour_agent_prepare',skill_context(s,'left'),{'pixels':[CUP_PIXEL]})
    assert result['arm']['pour_agent']['phase']=='prepare'
    assert len(result['arm']['path']['points'])==3
    assert result['mouth_geometry_measured'] is False


def test_agent_driver_reobserves_and_returns_without_extra_model_call():
    path=Path(__file__).parents[1]/'scripts/pour_agent.py';spec=importlib.util.spec_from_file_location('pour_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class Robot:
        def __init__(self):self.calls=[];self.n=0
        def post(self,path,data):
            self.calls.append(path)
            if path.endswith('start'):return {'id':'s'}
            if path.endswith('observe'):self.n+=1;return {'observation_token':str(self.n),'observation_id':str(self.n)}
            if path.endswith('plan'):return {'revision':self.n,'mode':'inspect' if self.n==1 else 'pour'}
            return {'revision':9}
        def execute(self,revision):self.calls.append('execute')
    class Agent:
        def __init__(self):self.calls=0
        def locate(self,o):self.calls+=1;return {'visible':True,'confidence':.9,'source_rim_pixels':[[1,1]]*6,'target_rim_pixels':[[2,2]]*6},{}
    r,a=Robot(),Agent();result=m.run(r,a,{},execute=True,record=lambda x:None)
    assert result['state']=='returned_upright' and not result['water_transfer_confirmed']
    assert a.calls==2 and r.calls[-2:]==['pour-agent/return','execute']


def test_gripper_change_requires_new_observation_and_guard_stops_loss(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    s.backend.openings['left']=46.
    r=c.post('/api/pour-agent/plan',json=d)
    assert r.status_code==409 and 'Grip changed' in r.text
    s.backend.openings['left']=42.
    r=c.post('/api/pour-agent/plan',json=d);assert r.status_code==200
    s.backend.openings['left']=55.
    with pytest.raises(ValueError,match='Gripper changed'):s.pour_agent.check_grip(s.pour_agent.pending['spec']['pour_agent'])


def test_background_circle_below_grip_is_not_a_bottle_mouth(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch);src['center'][2]=.1
    r=c.post('/api/pour-agent/plan',json=d)
    assert r.status_code==409 and 'background depth' in r.text


def test_return_reverses_each_original_curve_and_ignores_outbound_budget(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    s.pour_agent.plan(d);original=copy.deepcopy(s.draft['arms']['left'])
    tag=original['pour_agent'];last=original['path']['points'][-1]
    s.backend.positions['left']=np.asarray(last)
    s.backend.rotations['left']=original['orientation']['points'][-1][1:]
    s.pour_agent.completed(tag)
    s.pour_agent.session['steps']=80;s.pour_agent.session['distance']=2.1
    ret=s.pour_agent.return_upright({'session_id':d['session_id']})
    reverse=copy.deepcopy(s.draft['arms']['left'])
    assert reverse['path']['points']==list(reversed(original['path']['points']))
    assert ret['return_steps_remaining']==0
    # Compare the interpolators on matching progress, not just matching endpoints.
    from scipy.interpolate import PchipInterpolator
    p=np.array(original['path']['points']);q=np.array(reverse['path']['points']);t=np.linspace(0,1,101)
    np.testing.assert_allclose(PchipInterpolator(np.linspace(0,1,len(p)),p)(t),PchipInterpolator(np.linspace(0,1,len(q)),q)(1-t),atol=1e-10)
    s.preview(ret['revision'])


def test_compiled_pour_path_rejects_off_axis_interpolation(tmp_path,monkeypatch):
    from urai.pour_agent import verify_compiled_mouth
    s,c,d,src,tgt=configured(tmp_path,monkeypatch,inspection=0)
    s.pour_agent.plan(d);spec=copy.deepcopy(s.draft['arms']['left'])
    st=s.backend.state()['left'];opts=dict(bottle_length_m=.35,bottle_radius_m=.06,clearance_m=.04,pour_direction_xy=[1,0])
    report=verify_compiled_mouth(spec,st,src,tgt,opts)
    assert report['max_lip_xy_error_m']<=.004 and report['samples']>100
    spec['path']['points'][10][1]+=.02
    with pytest.raises(ValueError,match='interpolation'):verify_compiled_mouth(spec,st,src,tgt,opts)


def test_agent_returns_after_lost_rim_without_waiting_for_more_model_calls():
    path=Path(__file__).parents[1]/'scripts/pour_agent.py';spec=importlib.util.spec_from_file_location('pour_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    class Robot:
        def __init__(self):self.calls=[];self.n=0
        def post(self,path,data):
            self.calls.append(path)
            if path.endswith('start'):return {'id':'s'}
            if path.endswith('observe'):self.n+=1;return {'observation_token':str(self.n),'observation_id':str(self.n)}
            if path.endswith('plan'):return {'revision':1,'mode':'inspect'}
            return {'revision':9,'return_steps_remaining':0}
        def execute(self,revision):self.calls.append('execute')
    class Agent:
        def locate(self,o):return {'visible':o['observation_id']=='1','confidence':.9,'source_rim_pixels':[[1,1]]*6,'target_rim_pixels':[[2,2]]*6},{}
    r=Robot();result=m.run(r,Agent(),{},execute=True,record=lambda x:None)
    assert result['state']=='needs_view' and result['returned_upright']
    assert r.calls[-2:]==['pour-agent/return','execute']


def test_return_rejects_arm_moved_outside_session(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    s.pour_agent.plan(d);spec=s.pour_agent.pending['spec']
    s.backend.positions['left']=np.asarray(spec['path']['points'][-1])
    s.backend.rotations['left']=spec['orientation']['points'][-1][1:]
    s.pour_agent.completed(spec['pour_agent'])
    s.backend.positions['left'][0]+=.04
    with pytest.raises(ValueError,match='outside the pour session'):
        s.pour_agent.return_upright({'session_id':d['session_id']})


def test_prepare_uses_clicked_raw_surface_not_truncated_body_height(monkeypatch):
    from test_skills import scene_service,CUP_PIXEL
    from urai.skills import plan,skill_context
    s=scene_service();ctx=skill_context(s,'left')
    def forbid_locate(*args):raise AssertionError('Truncated body segmentation cannot define this grasp')
    monkeypatch.setattr(ctx,'locate',forbid_locate)
    surface=ctx.frame.pick(*CUP_PIXEL,mode='surface')
    result=plan('pour_agent_prepare',ctx,{'pixels':[CUP_PIXEL],'grasp_inset_mm':15})
    np.testing.assert_allclose(result['observed_grasp_surface_xyz'],surface)
    from scipy.spatial.transform import Rotation
    from urai.skills.bottle_cap import _compensation
    R=Rotation.from_euler('xyz',result['arm']['orientation']['points'][0][1:],degrees=True).as_matrix()
    np.testing.assert_allclose(result['grasp_xyz'],surface+R@np.array([0,_compensation(ctx),.015]))


def test_rgb_plan_never_uses_depth_and_preserves_motion_contract(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    def no_depth(*args,**kwargs):raise AssertionError('RGB must not read rim depth')
    monkeypatch.setattr('urai.pour_agent.rim_from_pixels',no_depth)
    def rgb(frame,pixels,radius,hint,*,horizontal=False,fixed_normal=False):
        return copy.deepcopy(tgt if horizontal else src),{'depth_used':False,'method':'calibrated_rgb_known_radius'}
    monkeypatch.setattr('urai.pour_agent.rgb_rim',rgb)
    d.update(perception='rgb',source_radius_m=.02,target_radius_m=.04)
    before=copy.deepcopy(s.backend.state());r=c.post('/api/pour-agent/plan',json=d)
    assert r.status_code==200,r.text
    assert r.json()['mode']=='inspect'
    assert r.json()['source_evidence']['depth_used'] is False
    assert s.backend.state()==before
    assert not any('opening_mm' in x for x in s.draft['arms']['left']['gripper_events'])


def test_rgb_probe_is_bounded_and_requires_normal_execution(tmp_path,monkeypatch):
    s,c,d,src,tgt=configured(tmp_path,monkeypatch)
    d.update(perception='rgb_probe',translation_m=[.015,0,0])
    before=copy.deepcopy(s.backend.state());r=c.post('/api/pour-agent/plan',json=d)
    assert r.status_code==200,r.text
    assert s.backend.state()==before
    spec=s.draft['arms']['left']
    assert spec['orientation']['mode']=='hold'
    assert spec['gripper_events']==[]
    assert np.linalg.norm(np.diff(spec['path']['points'],axis=0))<=.015001
    d['translation_m']=[.04,0,0]
    assert c.post('/api/pour-agent/plan',json=d).status_code==409
