"""The trajectory brush: a screen-space stroke lifted onto a plane or the measured surface."""
import numpy as np
import pytest
from conftest import make_client
from urai.backend import SimulationBackend


def test_brush_lifts_continuous_line_and_preserves_ends_and_bend():
    from urai.stroke import lift_stroke
    frame=SimulationBackend().capture()
    pixels=np.vstack([np.c_[np.linspace(300,400,150),np.full(150,200.)],
                      np.c_[np.full(150,400.),np.linspace(200,320,150)]])
    result=lift_stroke(frame,pixels,mode='plane',z=.24)
    points=np.array(result['points'])
    assert 3 <= len(points) <= 64
    np.testing.assert_allclose(points[0],frame.pick(300,200,z=.24))
    np.testing.assert_allclose(points[-1],frame.pick(400,320,z=.24))
    assert np.min(np.linalg.norm(points-frame.pick(400,200,z=.24),axis=1)) < .001


def test_long_brush_line_is_simplified_to_control_point_budget():
    from urai.stroke import lift_stroke
    frame=SimulationBackend().capture()
    t=np.linspace(0,1,2000);pixels=np.c_[100+600*t,240+70*np.sin(40*np.pi*t)]
    result=lift_stroke(frame,pixels,mode='plane',z=.24)
    assert 2 <= len(result['points']) <= 64
    assert result['source_sample_count']==2000


def test_brush_rejects_missing_depth_and_taps():
    from urai.stroke import lift_stroke
    frame=SimulationBackend().capture();frame.depth[:]=np.nan
    with pytest.raises(ValueError,match='depth'):
        lift_stroke(frame,[[300,200],[400,300]],mode='surface')
    with pytest.raises(ValueError):
        lift_stroke(frame,[[300,200],[300,200]],mode='plane')


def test_brush_api_only_projects_and_requires_current_observation(sim_service):
    s=sim_service;s.observe();client=make_client(s)
    request={'observation_id':s.frame.id,'pixels':[[300,200],[310,205],[320,210]],'mode':'plane','z':.24}
    response=client.post('/api/stroke',json=request)
    assert response.status_code==200
    assert len(response.json()['points'])>=2
    assert s.draft=={} and s.execution['state']=='idle'
    request['observation_id']='old'
    assert client.post('/api/stroke',json=request).status_code==409


def test_surface_brush_cannot_skip_depth_hole_between_pointer_events():
    from urai.stroke import lift_stroke
    frame=SimulationBackend().capture();frame.depth[:,345:376]=np.nan
    with pytest.raises(ValueError,match='depth'):
        lift_stroke(frame,[[300,300],[420,300]],mode='surface')
