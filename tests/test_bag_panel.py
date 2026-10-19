from types import SimpleNamespace
import numpy as np
import pytest
from urai.skills.bag_panel import panel_lift
from urai.skills.bag import stage_metadata
from urai.skills import get


class Frame:
    id = 'fresh-panel'
    width = height = 720
    def pick(self, u, v, mode='surface', z=None):
        return np.array([u/1000, v/1000, .10 if z is None else z])


def context():
    return SimpleNamespace(frame=Frame(), arm='right', model=None,
        backend=SimpleNamespace(table_checks=True), has_kinematics=False,
        current={'right': {'xyz':[.25, -.5, .22], 'rpy_deg':[180,0,0], 'joints_deg':[0]*6}},
        preferred_yaw_deg=0, base_xy=np.array([0., -.59]), clearance_m=.06, speed_m_s=.15)


LINE = np.array([[230, 150], [270, 150]])


def test_continuous_lift_and_close_timing_and_inverse_path():
    out = panel_lift(context(), LINE)
    sp, meta = out['arm'], out['bag_flap']
    points = np.array(sp['path']['points'])
    np.testing.assert_allclose(points[-1]-points[1], [0,0,.1])
    np.testing.assert_allclose(points[2]-points[1], [0,0,.04])
    close = next(e for e in sp['gripper_events'] if not e.get('pregrasp_open'))
    assert close['s'] == pytest.approx(1/(len(points)-1))
    assert sp['gripper_events'][0]['opening_mm'] == 40
    np.testing.assert_allclose(np.array(meta['return_poses'])[:,:3], points[1:][::-1])
    assert meta['grasp_pose'][:3] == list(points[1])
    assert meta['bag_open_verified'] is False
    assert stage_metadata({'input_draft':{'arms':{'right':sp}}})['phase']=='panel_lift'


def test_checked_upper_lift_failure_does_not_emit_partial_grasp(monkeypatch):
    import urai.skills.bag_panel as m
    attempts=[]
    def reject(ctx,points,rotations,events,**kw):
        attempts.append(points)
        raise ValueError('unreachable upper lift')
    monkeypatch.setattr(m,'checked_spec',reject)
    with pytest.raises(ValueError,match='上层抓提路径不可达'):
        panel_lift(context(),LINE)
    assert len(attempts)==5
    assert all(np.isclose(p[-1][2]-p[1][2], .1) for p in attempts)


def test_depth_discontinuity_fails_before_planning():
    ctx=context()
    ctx.frame.pick=lambda u,v,**kw:np.array([u/1000,v/1000,.2 if u>250 else .1])
    with pytest.raises(ValueError,match='深度不一致'):panel_lift(ctx,LINE)


def test_registry_bounds_and_defaults():
    skill=get('bag_panel_lift')
    assert skill.defaults()['lift_mm']==100
    assert skill.defaults()['inset_mm']==18
    with pytest.raises(ValueError):skill.parse({'lift_mm':141})
    with pytest.raises(ValueError):skill.parse({'inset_mm':25})


def test_panel_hold_does_not_auto_home(monkeypatch):
    from urai.backend import SimulationBackend
    from urai.service import Service
    s=Service(SimulationBackend());s.observe()
    out=panel_lift(context(),LINE)
    p={'input_draft':{'arms':{'right':out['arm']}},'arms':{'right':{}},'start':s.backend.state(),'id':'panel-test'}
    monkeypatch.setattr(s.backend,'execute',lambda *a:{'completed':True})
    monkeypatch.setattr(s.backend,'plan_home',lambda *a,**kw:pytest.fail('must hold panel'))
    s._run(p)
    assert s.execution['state']=='completed'
    assert s.execution['result']['holding_position'] is True
    assert s.execution['result']['bag_open_verified'] is False
