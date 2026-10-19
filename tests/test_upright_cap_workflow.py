import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.upright_cap import aligned_turn_route, cap_geometry, measured_tool_offset
from urai.skills import plan, skill_context, SKILLS
from urai.skills.bottle_cap import complete_stage, validate_stage
from urai.skills.motion import side_grasp_rotation
from urai.trajectory import compile_arm
from test_skills import scene_service


def ready():
    s=scene_service();s.backend.positions['left']=np.array([.33,-.275,.077])
    s.backend.rotations['left']=[-110,0,-125.65];s.backend.openings['left']=52
    h=s.backend.state()['left'];R=Rotation.from_euler('xyz',h['rpy_deg'],degrees=True).as_matrix()
    body_offset=np.array([0,.01,.023]);cap=np.asarray(h['xyz'])-R@body_offset+[0,0,.18]
    s.bottle_cap={'id':'upright','mode':'upright','state':'waiting_cap','holder':'left',
       'held_state':copy.deepcopy(h),'observation_id':'previous-frame',
       'axis_local':(R.T@np.array([0,0,1])).tolist(),'cap_local':(R.T@(cap-h['xyz'])).tolist(),
       'body_tool_offset':body_offset.tolist()}
    s.frame.pick=lambda *a,**kw:cap.copy()
    return s,cap


def complete_grasp(s):
    result=plan('bottle_upright_grasp_cap',skill_context(s,'right'),{'pixels':[[100,100]]})
    sp=result['arms']['right'];s.backend.positions['right']=np.array(sp['path']['points'][-1])
    s.backend.rotations['right']=sp['orientation']['points'][-1][1:];s.backend.openings['right']=33
    complete_stage(s,{'input_draft':{'arms':result['arms']},'observation_id':s.frame.id})
    return result


@pytest.mark.parametrize('reset',[False,True])
def test_all_rotations_keep_full_measured_contact_center_fixed(reset):
    axis=np.array([-.04,.04,1.]);axis/=np.linalg.norm(axis)
    base=side_grasp_rotation(np.array([.34,.94]),20)
    start=Rotation.from_rotvec(axis*np.deg2rad(30)).as_matrix()@base if reset else base
    center=np.array([.31,-.26,.267]);offset=np.array([.002,-.012,.0173])
    pts,rs,events=aligned_turn_route(center,axis,offset,start,base,30,6)
    for p,r in zip(pts,rs):np.testing.assert_allclose(p-r@offset,center,atol=1e-12)
    assert len(events)<=16
    assert events[-1]['opening_mm']==0 and all(e['wait_for_arrival'] for e in events)
    delta=Rotation.from_matrix(rs[-1]@base.T).as_rotvec()
    np.testing.assert_allclose(delta,axis*np.deg2rad(30),atol=1e-10)
    if reset:assert events[0]['s']==0 and events[0]['opening_mm']==50


def test_tool_offset_includes_insertion_and_transverse_error():
    R=side_grasp_rotation(np.array([.34,.94]),20);center=np.array([.3,-.26,.27]);offset=np.array([.002,-.012,.0173])
    np.testing.assert_allclose(measured_tool_offset(center+R@offset,R,center),offset,atol=1e-12)
    with pytest.raises(ValueError,match='40 mm'):measured_tool_offset(center+[0,0,.05],R,center)


def test_bottle_axis_comes_from_body_and_cap_in_same_frame():
    R=Rotation.from_euler('xyz',[-110,0,-125],degrees=True).as_matrix()
    h=np.array([.33,-.275,.077]);offset=[0,.01,.023];body=h-R@offset
    axis=np.array([.03,.06,1.]);axis/=np.linalg.norm(axis);surface=body+axis*.18
    center,got=cap_geometry(surface,h,R,offset,.008)
    np.testing.assert_allclose(got,axis,atol=1e-12)
    np.testing.assert_allclose(center,surface-axis*.008,atol=1e-12)
    with pytest.raises(ValueError,match='15°'):cap_geometry(body+[.12,0,.10],h,R,offset,.008)


def test_grasp_commands_stationary_holder_and_only_publishes_reference_on_success():
    s,cap=ready();result=plan('bottle_upright_grasp_cap',skill_context(s,'right'),{'pixels':[[100,100]]})
    assert set(result['arms'])=={'left','right'}
    assert result['arms']['right']['approach']['speed_m_s']==.02
    assert 'upright_reference' not in s.bottle_cap
    meta=validate_stage(s,{'input_draft':result},s.backend.state())
    assert meta['active_holder'] and meta['upright_phase']=='grasp'
    assert result['arms']['left']['gripper_events']==[]


def test_turn_requires_new_post_grasp_observation():
    s,cap=ready();complete_grasp(s)
    with pytest.raises(ValueError,match='重新观测'):plan('bottle_upright_twist',skill_context(s,'right'),{'pixels':[[100,100]]})
    s.frame.id='fresh-after-grasp';s.observation_start=copy.deepcopy(s.backend.state())
    result=plan('bottle_upright_twist',skill_context(s,'right'),{'pixels':[[100,100]]})
    sp=result['arms']['right'];compiled=compile_arm(sp,s.backend.state()['right'])
    assert len(compiled['gripper_events'])==10
    assert all(e.get('verify')=='holding' for e in sp['gripper_events'] if e['opening_mm']==0)
    assert sp['gripper_events'][-1]['opening_mm']==0
    center=np.asarray(result['cap_center']);offset=np.asarray(result['tool_offset'])
    for p,key in zip(sp['path']['points'],sp['orientation']['points']):
        R=Rotation.from_euler('xyz',key[1:],degrees=True).as_matrix()
        np.testing.assert_allclose(np.array(p)-R@offset,center,atol=1e-10)


def test_empty_grip_requires_regrasp_instead_of_blind_turning():
    s,cap=ready();complete_grasp(s);s.frame.id='fresh';s.observation_start=copy.deepcopy(s.backend.state())
    s.backend.openings['right']=.2
    with pytest.raises(ValueError,match='未夹住'):plan('bottle_upright_twist',skill_context(s,'right'),{'pixels':[[100,100]]})


def test_cap_retry_retracts_before_changing_contact_point():
    s,cap=ready();complete_grasp(s);s.frame.id='retry';s.observation_start=copy.deepcopy(s.backend.state())
    current=s.backend.state()['right'];R=Rotation.from_euler('xyz',current['rpy_deg'],degrees=True).as_matrix()
    r=plan('bottle_upright_grasp_cap',skill_context(s,'right'),{'pixels':[[100,100]]})['arms']['right']
    np.testing.assert_allclose(r['path']['points'][1],np.array(current['xyz'])-R[:,2]*.04)
    assert r['gripper_events'][0]['opening_mm']==50 and r['gripper_events'][0]['s']==0


def test_removal_keeps_jaws_closed_and_needs_detachment_for_retreat():
    s,cap=ready();complete_grasp(s);s.frame.id='fresh';s.observation_start=copy.deepcopy(s.backend.state())
    with pytest.raises(ValueError,match='先小幅'):plan('bottle_upright_take_cap',skill_context(s,'right'),{'pixels':[[100,100]],'retreat_mm':70})
    r=plan('bottle_upright_take_cap',skill_context(s,'right'),{'pixels':[[100,100]]})
    assert r['arms']['right']['gripper_events']==[]
    s.frame.id='detached';s.observation_start=copy.deepcopy(s.backend.state())
    r=plan('bottle_upright_take_cap',skill_context(s,'right'),{'pixels':[[100,100]],'lift_mm':15,'retreat_mm':70,'cap_detached':True})
    complete_stage(s,{'input_draft':r,'observation_id':s.frame.id})
    assert s.bottle_cap['cap_removed'] and 'upright_reference' in s.bottle_cap
    s.frame.id='removed';s.observation_start=copy.deepcopy(s.backend.state())
    with pytest.raises(ValueError,match='已取下'):plan('bottle_upright_twist',skill_context(s,'right'),{'pixels':[[100,100]]})


def test_lower_body_defaults_preserve_upright_bottle_and_reject_table_contact():
    s,cap=ready();s.bottle_cap=None;s.frame.pick=lambda *a,**kw:np.array([.30,-.215,.254])
    c=skill_context(s,'left');r=plan('bottle_upright_prepare',c,{'pixels':[[100,100]]})
    sp=r['arm'];assert sp['bottle_cap']['body_offset_m']==.18
    assert SKILLS['bottle_upright_prepare'].defaults()['body_pitch_deg']==20
    start_R=Rotation.from_euler('xyz',sp['orientation']['points'][0][1:],degrees=True).as_matrix()
    np.testing.assert_allclose(start_R@np.array(sp['bottle_cap']['axis_local']),[0,0,1],atol=1e-10)
    s.frame.pick=lambda *a,**kw:np.array([.30,-.215,.21])
    with pytest.raises(ValueError,match='桌面|过低'):plan('bottle_upright_prepare',skill_context(s,'left'),{'pixels':[[100,100]]})


def test_cap_is_carried_closed_and_only_released_at_table():
    s,_=ready();complete_grasp(s);s.bottle_cap.update(cap_removed=True,
        upright_reference={'tool_offset':[.002,-.012,.0173]})
    s.frame.id='place-cap';s.observation_start=copy.deepcopy(s.backend.state())
    s.frame.pick=lambda *a,**kw:np.array([.27,-.54,.004])
    result=plan('bottle_upright_place_cap',skill_context(s,'right'),{'pixels':[[100,100]]})
    sp=result['arms']['right'];event=sp['gripper_events'][0]
    assert len(sp['gripper_events'])==1 and event['s']==7/8 and event['wait_for_arrival']
    assert result['arms']['left']['gripper_events']==[]
    assert 'cap_placed' not in s.bottle_cap
    complete_stage(s,{'input_draft':result,'observation_id':s.frame.id})
    assert s.bottle_cap['cap_placed']
    saved=copy.deepcopy(s.bottle_cap)
    complete_stage(s,{'kind':'home','arms':{'right':{}}})
    assert s.bottle_cap==saved
    complete_stage(s,{'kind':'home','arms':{'left':{}}})
    assert s.bottle_cap is None


def test_return_reverses_real_pickup_then_releases_and_retracts():
    s,_=ready();s.bottle_cap=None;s.frame.pick=lambda *a,**kw:np.array([.3,-.215,.254])
    prep=plan('bottle_upright_prepare',skill_context(s,'left'),{'pixels':[[100,100]]})['arm']
    s.backend.positions['left']=np.array(prep['path']['points'][-1])
    s.backend.rotations['left']=prep['orientation']['points'][-1][1:]
    complete_stage(s,{'input_draft':{'arms':{'left':prep}},'observation_id':s.frame.id})
    s.bottle_cap.update(cap_removed=True,cap_placed=True)
    s.frame.id='return';s.observation_start=copy.deepcopy(s.backend.state())
    result=plan('bottle_upright_return',skill_context(s,'left'),{'pixels':[[100,100]]})
    spec=result['arm'];pts=spec['path']['points'];original=prep['path']['points']
    np.testing.assert_allclose(pts[1:4],original[-2:0:-1])
    np.testing.assert_allclose(pts[4],original[0])
    event=spec['gripper_events'][0]
    assert event['s']==3/5 and event['wait_for_arrival']
    assert validate_stage(s,{'input_draft':{'arms':{'left':spec}}},s.backend.state())['stage']=='place'
    complete_stage(s,{'input_draft':{'arms':{'left':spec}},'observation_id':s.frame.id})
    assert s.bottle_cap is None
