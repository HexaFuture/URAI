from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.transfer import transfer_draft
from urai.trajectory import compile_arm
from urai.backend import SimulationBackend


def transfer(probe=None, near=False, pinch=False, top_down=False, compensation=.01):
    contact=np.array([.50,-.445,.012]) if not near else np.array([.24,-.59,.012])
    place=np.array([.36,-.35,.055])
    placement=dict(grasp=contact,place=place,hover=contact+[0,0,.1],arrival=place+[0,0,.06],finger_z=.012,placement='place')
    found=dict(axis_deg=-86.,width=.042,top=.018,pinch=pinch)
    options=dict(probe=probe,seed_joints=np.zeros(6),pose_tolerance=(3.,3.),compensation_m=compensation,min_tcp_z_m=.001)
    return transfer_draft(SimpleNamespace(id='test'),found,placement,[],'two-point',.0035,.08,.2,-90.,
        base_xy=[0,-.59],top_down=top_down,grasp_options=None if probe is None else options)


def side_only(x,r,q):
    return (0.,0.,np.zeros(6)) if -r[2,2] <= np.cos(np.deg2rad(49)) else (20.,40.,None)


@pytest.mark.parametrize('compensation',[.01,-.01])
def test_far_transfer_has_level_fingers_and_axial_approach(compensation):
    r=transfer(side_only,compensation=compensation);sp=r['arm'];pts=np.asarray(sp['path']['points'])
    R=Rotation.from_euler('xyz',sp['orientation']['points'][0][1:],degrees=True).as_matrix()
    assert abs(R[2,0])<1e-12
    assert r['grasp_tilt_deg']==50.
    np.testing.assert_allclose(pts[0]-pts[1],-R[:,2]*.08)
    np.testing.assert_allclose(pts[1]-R[:,1]*compensation,r['source_contact_xyz'])
    assert pts[2,2]>pts[1,2]
    close=sp['gripper_events'][1]
    assert close['verify']=='holding' and close['wait_for_arrival']
    plan=compile_arm(sp,SimulationBackend().state()['right'])
    close=plan['gripper_events'][1]
    at=np.argmin(np.abs(plan['time_s']-close['time_s']))
    np.testing.assert_allclose(plan['xyz'][at],pts[1],atol=1e-7)
    assert plan['gripper_events'][-1]['hold_end_time_s']<plan['time_s'][-1]


@pytest.mark.parametrize('kwargs',[{'near':True},{'pinch':True},{'top_down':True}])
def test_near_cloth_and_explicit_vertical_paths_unchanged(kwargs):
    def forbidden(*args):raise AssertionError('must not solve this path')
    old,new=transfer(**kwargs),transfer(forbidden,**kwargs)
    assert old['arm']==new['arm']


def test_unreachable_side_grasp_does_not_fall_back_to_unverified_pose():
    with pytest.raises(ValueError,match='不可达'):
        transfer(lambda x,r,q:(99.,99.,None))


def test_original_radial_pose_reproduces_thin_block_tip_height_error():
    R=Rotation.from_euler('xyz',[-129.65598825251513,15.906790276539974,-86.70408899989498],degrees=True).as_matrix()
    assert abs(R[2,0])*.042>.009
