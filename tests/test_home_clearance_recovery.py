import numpy as np
import pytest
from urai.home_recovery import HomeClearanceRecovery

def test_planned_retreat_and_reentry():
    g=HomeClearanceRecovery(29.9)
    for c in [29.9,30,31,32,40,31]:g.check(c)
    g.finish()
    with pytest.raises(ValueError,match='re-entered'):g.check(29.9)

@pytest.mark.parametrize('start',[-1,0,27.99,30,float('nan')])
def test_invalid_start(start):
    with pytest.raises(ValueError):HomeClearanceRecovery(start)

def test_worsening_is_rejected():
    g=HomeClearanceRecovery(29.9);g.check(30.1)
    with pytest.raises(ValueError,match='reduce'):g.check(30)

def test_no_drift_or_stuck_recovery():
    g=HomeClearanceRecovery(29.9,feedback=True);g.check(29.7)
    with pytest.raises(ValueError):g.check(29.5)
    with pytest.raises(ValueError):g.finish()

def test_actual_failed_pose_can_retreat_without_relaxing_ordinary_motion(planner):
    from urai.robot.arm_model import capsules_clearance_mm
    from scipy.spatial.transform import Rotation
    b,models=planner,planner.models
    # Recorded joints, without a dependency on a private run artifact. Find the
    # near-margin point with the installed base calibration: recalibration can
    # move the original pair below the recovery's strict 28 mm lower bound.
    joints={'left':np.array([-22.978,65.649,-38.55,-9.721,36.389,7.473]),
            'right':np.array([49.021,78.529,-42.574,-47.384,-3.867,-8.781])}
    def gap(fraction):
        return capsules_clearance_mm(models['left'].body_capsules(joints['left']),
            models['right'].body_capsules(joints['right']*(1-fraction)))[0]
    lo,hi=0.,1.
    target=max(29.,gap(lo)+.01)
    assert target<30. and gap(lo)<target<gap(hi)
    for _ in range(50):
        mid=(lo+hi)/2
        if gap(mid)<target:lo=mid
        else:hi=mid
    joints['right']*=1-hi
    st={}
    for arm,q in joints.items():
        pose=models[arm].fk_tcp_world(q)
        st[arm]={'joints_deg':q.tolist(),'xyz':pose[:3,3].tolist(),
                 'rpy_deg':Rotation.from_matrix(pose[:3,:3]).as_euler('xyz',degrees=True).tolist(),
                 'gripper_mm':50.}
    with pytest.raises(ValueError):b._geometry_check(st['left']['joints_deg'],st['right']['joints_deg'])
    plan=b.plan_home(st,['right'],open_grippers=False)
    assert plan['home_clearance_recovery']['arm']=='right'
    assert getattr(b,'_home_clearance_recovery',None) is None
    assert plan['motion_limits']['profile']=='fine'
    clear=[]
    for t in np.linspace(0,plan['duration_s'],2001):
        q=plan['arms']['right']['curve'](t)
        clear.append(capsules_clearance_mm(models['left'].body_capsules(st['left']['joints_deg']),models['right'].body_capsules(q))[0])
    g=HomeClearanceRecovery(clear[0])
    for c in clear:g.check(c)
    g.finish()
    with pytest.raises(ValueError):b._geometry_check(st['left']['joints_deg'],st['right']['joints_deg'])
    with pytest.raises(ValueError):b.plan_home(st,['right'],open_grippers=True)
    with pytest.raises(ValueError):b.plan_home(st,['left','right'],open_grippers=False)


def test_feedback_restores_normal_margin_as_soon_as_clear():
    g=HomeClearanceRecovery(29.9,feedback=True)
    for c in [29.9,31.35,30.08,32.5]:g.check(c)
    g.finish()
    with pytest.raises(ValueError):g.check(29.99)
