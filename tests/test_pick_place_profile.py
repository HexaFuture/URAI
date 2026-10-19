import copy
import numpy as np
import pytest
from urai.pick_place_profile import optimize_pick_place


def inputs():
    r=dict(object_width_mm=25,object_height_mm=18,finger_z_mm=10,pinch=False,top_down=False,
      source_xyz=[.24,-.59,.0135],target_xyz=[.36,-.35,.05],
      arm=dict(path={'mode':'waypoints','points':[[.24,-.59,.14],[.24,-.59,.0135],[.36,-.35,.05]]},
       orientation={'mode':'keyframes','points':[[0,180,0,-90],[1,180,0,-90]]},
       gripper_events=[dict(s=0,opening_mm=70,hold_s=.8,wait_for_arrival=True),dict(s=.2,opening_mm=0,hold_s=1.,verify='holding',wait_for_arrival=True,ramp_s=.5),dict(s=.8,opening_mm=70,hold_s=.8,wait_for_arrival=True)]))
    p=dict(base_xy=[0,-.59],clearance_m=.08,grasp_options=dict(probe=lambda x,R,q:(0.,0.,np.zeros(6)),seed_joints=np.zeros(6),pose_tolerance=(3.,3.),compensation_m=-.01,min_tcp_z_m=.0005))
    return r,p

def test_profile_is_calibrated_and_preserves_feedback_and_input():
    r,p=inputs();before=copy.deepcopy(r);out=optimize_pick_place(r,p)
    assert r==before
    assert out['opening_mm']==39
    assert out['extra_inset_mm']==6
    assert out['finger_z_mm']==4
    assert out['arm']['gripper_events'][1]['verify']=='holding'
    assert all(e['wait_for_arrival'] for e in out['arm']['gripper_events'])
    assert sum(e['hold_s'] for e in out['arm']['gripper_events'])==pytest.approx(1.6)
    assert np.linalg.norm(np.array(out['source_xyz'])-out['source_contact_xyz'])==pytest.approx(.01)

@pytest.mark.parametrize('change',[{'pinch':True},{'top_down':True},{'object_height_mm':90},{'object_width_mm':100}])
def test_other_transfers_keep_existing_profile(change):
    r,p=inputs();r.update(change);assert optimize_pick_place(r,p) is r

def test_no_model_is_unchanged():
    r,p=inputs();p['grasp_options']=None;assert optimize_pick_place(r,p) is r

def test_thin_object_retains_minimum_table_clearance():
    r,p=inputs();r['finger_z_mm']=4;out=optimize_pick_place(r,p)
    assert out['extra_inset_mm']==1 and out['finger_z_mm']==3


def test_far_small_pick_place_solves_only_final_contact(monkeypatch):
    from types import SimpleNamespace
    from urai.transfer import transfer_draft
    import urai.far_transfer as far
    import urai.pick_place_profile as profile
    original=far.refine_transfer; calls=[]
    def once(result,placement,*args,**kwargs):
        calls.append(float(placement['grasp'][2]))
        # An obsolete high contact can be unreachable while the final lower grasp is reachable.
        assert placement['grasp'][2]<.009
        return original(result,placement,*args,**kwargs)
    monkeypatch.setattr(far,'refine_transfer',once)
    monkeypatch.setattr(profile,'refine_transfer',once)
    grasp=np.array([.50,-.445,.012]);place=np.array([.36,-.35,.055])
    placement=dict(grasp=grasp,place=place,hover=grasp+[0,0,.1],arrival=place+[0,0,.06],finger_z=.012,placement='place')
    found=dict(axis_deg=-86.,width=.042,top=.018,pinch=False)
    _,params=inputs()
    result=transfer_draft(SimpleNamespace(id='test'),found,placement,[],'two-point',.0035,.08,.8,-90.,
        base_xy=[0,-.59],grasp_options=params['grasp_options'],compact_pick_place=True)
    assert len(calls)==1
    assert result['pick_place_profile']=='compact_calibrated'
    assert result['arm']['gripper_events'][1]['verify']=='holding'


def test_observed_table_restoration_preserves_local_contact_clearance():
    from urai.transfer import restore_support_frame,plane_height
    plane=np.array([-.012,-.05,-.0066]);table=.0035
    placement={'grasp':np.array([.37,-.68,.0115]),'place':np.array([.49,-.44,.045]),
               'hover':np.array([.37,-.68,.12]),'arrival':np.array([.49,-.44,.12]),'finger_z':.0115}
    before=copy.deepcopy(placement);out=restore_support_frame(placement,plane,table)
    for key in ('grasp','place','hover','arrival'):
        assert out[key][2]-plane_height(plane,out[key][:2])==pytest.approx(before[key][2]-table)
        np.testing.assert_array_equal(placement[key],before[key])
    assert out['grasp'][2]-placement['grasp'][2]>.019
    assert out['finger_z']==placement['finger_z']


def test_observed_table_restoration_rejects_unbounded_offset():
    from urai.transfer import restore_support_frame
    p={key:[.3,-.6,.02] for key in ('grasp','place','hover','arrival')}
    with pytest.raises(ValueError,match='高度差过大'):restore_support_frame(p,[0,0,.2],.0035)
