import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.geometry import Frame
from urai.pour_rgb import rgb_rim


def scene():
    k=np.array([[912.,0,640],[0,912.,360],[0,0,1.]])
    t=np.eye(4);t[:3,:3]=np.diag([1.,-1.,-1.]);t[:3,3]=[0,0,1.]
    return Frame(np.zeros((720,1280,3),np.uint8),np.full((720,1280),np.nan),k,t)


def contour(f,c,n,r):
    u=np.cross(n,[0,1.,0]);u/=np.linalg.norm(u);v=np.cross(n,u)
    a=np.arange(12)*2*np.pi/12
    return f.project(np.asarray(c)+r*(np.cos(a)[:,None]*u+np.sin(a)[:,None]*v))


@pytest.mark.parametrize('angle',[0,15,35,65,115,125])
def test_rgb_known_radius_recovers_pose_with_all_depth_missing(angle):
    f=scene();n=Rotation.from_euler('x',angle,degrees=True).apply([0.,0.,1.]);c=np.array([.1,.04,.45])
    px=contour(f,c,n,.016)
    rim,ev=rgb_rim(f,px,.016,n)
    np.testing.assert_allclose(rim['center'],c,atol=1e-5)
    assert rim['normal']@n> .9999
    assert not ev['depth_used']


def test_receiver_rgb_radius_recovers_horizontal_plane():
    f=scene();c=np.array([.1,0,.06]);px=contour(f,c,[0,0,1.],.085)
    rim,_=rgb_rim(f,px,.085,[0,0,1.],horizontal=True)
    np.testing.assert_allclose(rim['center'],c,atol=1e-6)


def test_bad_rgb_input_rejected():
    f=scene()
    with pytest.raises(ValueError,match='distinct'):rgb_rim(f,[[10,20]]*6,.016,[0,0,1])
    px=contour(f,[.1,.04,.45],[0,0,1.],.016);px[2]+=[20,25]
    with pytest.raises(ValueError,match='reprojection'):rgb_rim(f,px,.016,[0,0,1])


def test_rgb_parallax_recovers_metric_scale_without_radius_or_depth():
    from urai.pour_rgb import triangulated_rim
    f=scene();c=np.array([.1,.04,.45]);n=np.array([0.,0.,1.]);delta=np.array([.015,0,0])
    before=contour(f,c,n,.016);after=contour(f,c+delta,n,.016)
    rim,ev=triangulated_rim(f,before,after,delta,n)
    np.testing.assert_allclose(rim['center'],c+delta,atol=1e-6)
    assert rim['radius']==pytest.approx(.016)
    assert ev['depth_used'] is False
    with pytest.raises(ValueError,match='baseline'):triangulated_rim(f,before,after,delta*.1,n)


def test_fixed_normal_tracks_translated_tilted_rim_without_pose_branch_flip():
    f=scene();n=Rotation.from_euler('x',27,degrees=True).apply([0.,0.,1.])
    for c in ([.1,.04,.45],[.14,-.07,.48]):
        px=contour(f,c,n,.014)
        rim,e=rgb_rim(f,px,.014,n,fixed_normal=True)
        np.testing.assert_allclose(rim['center'],c,atol=1e-6)
        np.testing.assert_allclose(rim['normal'],n,atol=1e-12)
        assert e['normal_fixed']
