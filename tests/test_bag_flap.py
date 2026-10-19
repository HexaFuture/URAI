from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.skills.bag import edge_geometry, grasp, plan_peel, stage_metadata
from urai.skills import get

class Frame:
    id='before'
    def pick(self,u,v,mode='surface',z=None):
        return np.array([u/1000,v/1000,.10 if z is None else z])


def context(arm='left'):
    frame=Frame()
    state={arm:dict(xyz=[.25,.15,.116],rpy_deg=[180,0,90],joints_deg=[0]*6)}
    return SimpleNamespace(frame=frame,arm=arm,model=None,backend=SimpleNamespace(table_checks=True),
        current=state,has_kinematics=False,preferred_yaw_deg=0,clearance_m=.06,speed_m_s=.3)

INK=np.array([[200,150],[300,150],[250,250]])
def probe(ctx):return grasp(ctx,INK,5,4,20,24,800)


def test_endpoint_order_does_not_flip_grasp_or_hinge():
    a=edge_geometry(Frame(),INK,5); b=edge_geometry(Frame(),INK[[1,0,2]],5)
    np.testing.assert_allclose(a[0],b[0]);np.testing.assert_allclose(a[1],b[1])
    np.testing.assert_allclose(a[0],[.25,.155,.1])
    np.testing.assert_allclose(a[1],[0,1])


def test_shallow_grasp_crosses_edge_and_only_lifts_20mm():
    ctx=context();result=probe(ctx);sp=result['arm'];p=np.array(sp['path']['points'])
    R=Rotation.from_euler('xyz',sp['orientation']['points'][0][1:],degrees=True).as_matrix()
    assert abs(R[:,0] @ [0,1,0]) == pytest.approx(1.)
    assert p[1,2]==pytest.approx(.096)
    np.testing.assert_allclose(p[2]-p[1],[0,0,.02])
    assert sp['gripper_events'][0]['opening_mm']==24
    assert result['bag_flap']['bag_open_verified'] is False


def test_mirrored_tool_bias_and_no_arbitrary_yaw():
    a=probe(context('left'));b=probe(context('right'))
    assert a['jaw_yaw_deg']==b['jaw_yaw_deg']
    pa=np.array(a['arm']['path']['points'][1]);pb=np.array(b['arm']['path']['points'][1])
    assert np.linalg.norm(pa-pb)==pytest.approx(.02)


@pytest.mark.parametrize('ink',[INK[:2],np.array([[200,150],[300,150],[250,151]])])
def test_ambiguous_flap_side_rejected(ink):
    with pytest.raises(ValueError):edge_geometry(Frame(),ink,5)


def test_peel_requires_fresh_frame_and_separated_layers():
    ctx=context();r=probe(ctx)
    with pytest.raises(ValueError,match='重新观察'):plan_peel(ctx,r,[250,150],[250,180])
    ctx.frame.id='after'
    with pytest.raises(ValueError,match='上下层分离'):plan_peel(ctx,r,[250,150],[250,180])


def test_peel_lifts_first_then_moves_toward_hinge_without_opening_jaws():
    ctx=context();r=probe(ctx);ctx.frame.id='after'
    ctx.current['left']['xyz']=r['bag_flap']['probe_xyz']
    ctx.frame.pick=lambda u,v,**kw: np.array([u/1000,v/1000,.12 if v==150 else .08])
    result=plan_peel(ctx,r,[250,150],[250,180]);p=np.array(result['arm']['path']['points'])
    np.testing.assert_allclose(p[1]-p[0],[0,0,.08])
    np.testing.assert_allclose(p[-1]-p[1],[0,.06,0])
    assert not result['arm']['gripper_events']
    ctx.current['left']['xyz']=np.array(ctx.current['left']['xyz'])+[.02,0,0]
    with pytest.raises(ValueError,match='位置已改变'):plan_peel(ctx,r,[250,150],[250,180])


def test_unreachable_topdown_not_silently_replaced_by_side_grasp():
    ctx=context();ctx.has_kinematics=True;ctx.backend.pose_error=lambda *a:(1.,10.,[0]*6)
    with pytest.raises(ValueError,match='不可达'):probe(ctx)


def test_metadata_cannot_hold_unrelated_dual_arm_execution():
    report=probe(context());plan={'input_draft':{'arms':{'left':report['arm']}}}
    assert stage_metadata(plan)['phase']=='probe'
    plan['input_draft']['arms']['right']={}
    with pytest.raises(ValueError):stage_metadata(plan)


def test_registered_defaults_are_shallow():
    assert get('bag_flap_grasp').defaults()['pinch_depth_mm']==4


def test_service_stops_at_hold_and_does_not_claim_bag_open(monkeypatch):
    from urai.backend import SimulationBackend
    from urai.service import Service
    s=Service(SimulationBackend());s.observe()
    report=probe(context());p={'input_draft':{'arms':{'left':report['arm']}},'arms':{'left':{}},'start':s.backend.state(),'id':'bag-test'}
    monkeypatch.setattr(s.backend,'execute',lambda *a:{'completed':True})
    monkeypatch.setattr(s.backend,'plan_home',lambda *a,**kw:pytest.fail('must not home while holding flap'))
    s._run(p)
    assert s.execution['state']=='completed',s.execution
    assert s.execution['result']['holding_position'] is True
    assert s.execution['result']['bag_open_verified'] is False


def test_dense_ui_stroke_preserves_edge_and_inside_direction():
    stroke=np.vstack([np.linspace(INK[0],INK[1],30),np.linspace(INK[1],INK[2],30)[1:]])
    a=edge_geometry(Frame(),stroke,5);b=edge_geometry(Frame(),INK,5)
    np.testing.assert_allclose(a[0],b[0]);np.testing.assert_allclose(a[1],b[1])


def test_peel_points_fingertips_toward_hinge_not_wrist_outward():
    ctx=context();r=probe(ctx);ctx.frame.id='after'
    ctx.current['left']['xyz']=r['bag_flap']['probe_xyz']
    ctx.current['left']['rpy_deg']=r['arm']['orientation']['points'][-1][1:]
    ctx.frame.pick=lambda u,v,**kw:np.array([u/1000,v/1000,.12 if v==150 else .08])
    out=plan_peel(ctx,r,[250,150],[250,180])
    R=Rotation.from_euler('xyz',out['arm']['orientation']['points'][-1][1:],degrees=True).as_matrix()
    assert R[:,2] @ [0,1,0] > .4


def test_facing_tool_points_at_other_arm_from_selected_endpoint():
    from urai.skills.bag import plan_face_other
    ctx=context();ctx.current['right']={'xyz':[.25,-.5,.2]}
    out=plan_face_other(ctx,[.25,.08,.16])
    direction=np.asarray(ctx.current['right']['xyz'])[:2]-[.25,.08]
    direction/=np.linalg.norm(direction)
    forward=np.asarray(out['forward_xy']);forward/=np.linalg.norm(forward)
    np.testing.assert_allclose(forward,direction)
    assert not out['arm']['gripper_events']
    assert stage_metadata({'input_draft':{'arms':{'left':out['arm']}}})['phase']=='face_other'
