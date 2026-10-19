import numpy as np
import pytest
from urai.pour_agent import horizontal_target_rim

class Receiver:
    depth=np.ones((200,200))
    def __init__(self):
        self.pixels=np.array([[100+40*np.cos(a),100+40*np.sin(a)] for a in np.linspace(0,2*np.pi,8,endpoint=False)])
        self.heights=np.array([.025,.030,.045,.038,.027,.031,.042,.033])
    def pick(self,u,v,mode,z=None):
        i=np.argmin(np.linalg.norm(self.pixels-[u,v],axis=1));a=i*np.pi/4
        h=self.heights[i] if mode=='surface' else z
        return np.array([.4+.08*np.cos(a)+.1*h,-.3+.08*np.sin(a),h])

def test_shiny_receiver_uses_upper_depth_bound_and_reports_uncertainty():
    f=Receiver();rim,evidence=horizontal_target_rim(f,f.pixels,.0035)
    assert rim['center'][2]==pytest.approx(.055)
    assert rim['radius']==pytest.approx(.076)
    assert evidence['physical_radius_m']==pytest.approx(.08)
    assert evidence['height_interval_m']==pytest.approx([.015,.055])
    assert evidence['center_xy_uncertainty_m']==pytest.approx(.004)

def test_rejects_background_or_unbounded_receiver_depth():
    f=Receiver();f.heights[3]=.20
    with pytest.raises(ValueError,match='depth range'):horizontal_target_rim(f,f.pixels,.0035)

def test_rejects_non_circular_horizontal_contour():
    f=Receiver();original=f.pick
    def pick(u,v,mode,z=None):
        p=original(u,v,mode,z);p[0]=.4+(p[0]-.4)*.45;return p
    f.pick=pick
    with pytest.raises(ValueError,match='RMS'):horizontal_target_rim(f,f.pixels,.0035)

def test_missing_depth_uses_remaining_six_measured_rim_locations():
    f=Receiver();f.depth=f.depth.copy()
    for u,v in f.pixels[:2].astype(int):f.depth[v-2:v+3,u-2:u+3]=0
    rim,evidence=horizontal_target_rim(f,f.pixels,.0035)
    assert evidence['height_interval_m'][1]==pytest.approx(.055)
    u,v=f.pixels[2].astype(int);f.depth[v-2:v+3,u-2:u+3]=0
    with pytest.raises(ValueError,match='insufficient'):horizontal_target_rim(f,f.pixels,.0035)

def test_temporal_depth_never_invents_unsupported_or_mixed_surface_depth():
    from urai.pour_agent import median_depth_frames
    got=median_depth_frames([np.array([[.500,0,.500,.50]]),np.array([[.502,.50,0,.70]]),np.array([[.501,0,.502,.51]])])
    np.testing.assert_allclose(got,[[.501,0,.501,0]])
