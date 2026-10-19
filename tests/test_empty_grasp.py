import threading
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from urai.empty_grasp import EmptyGraspError,recover_empty_grasp
from urai.backend import Cancelled

def service(opening=0.5):
    state={'right':{'gripper_mm':opening},'left':{'gripper_mm':20}}
    b=SimpleNamespace(state=Mock(return_value=state),prepare=Mock(),plan_home=Mock(return_value={'arms':{'right':{}}}),execute=Mock(return_value={'completed':True}),capture=Mock(return_value=object()),constraint_state={})
    return SimpleNamespace(backend=b,cancel_event=threading.Event(),lock=threading.RLock(),execution={'state':'running','error':'old'},pose_tolerance={'position_mm':3,'orientation_deg':3},motion_profile='throw',home_speed_m_s=.8,revision=4,draft={'old':1},preview_data={'old':1})

def test_only_empty_arm_returns_then_fresh_observation_is_ready():
    s=service();r=recover_empty_grasp(s,{'arms':{'right':{}}},EmptyGraspError('right',.5),lambda *a:None)
    assert r['completed'] is False and r['ready_for_next']
    assert s.execution['state']=='idle' and 'error' not in s.execution
    assert s.draft=={} and s.preview_data is None and s.revision==5
    assert s.backend.plan_home.call_args.args[1]==['right']
    assert s.backend.plan_home.call_args.kwargs['open_grippers'] is True
    s.backend.capture.assert_called_once()

@pytest.mark.parametrize('opening',[2.,70.,float('nan'),-1.])
def test_not_proven_empty_never_moves(opening):
    s=service(opening)
    with pytest.raises(ValueError):recover_empty_grasp(s,{'arms':{'right':{}}},EmptyGraspError('right',.5),lambda *a:None)
    s.backend.execute.assert_not_called()

def test_cancellation_never_moves():
    s=service();s.cancel_event.set()
    with pytest.raises(Cancelled):recover_empty_grasp(s,{'arms':{'right':{}}},EmptyGraspError('right',.5),lambda *a:None)
    s.backend.execute.assert_not_called()

def test_failed_home_does_not_clear_original_state():
    s=service();s.backend.execute.side_effect=ValueError('collision')
    with pytest.raises(ValueError,match='collision'):recover_empty_grasp(s,{'arms':{'right':{}}},EmptyGraspError('right',.5),lambda *a:None)
    s.backend.capture.assert_not_called();assert s.revision==4

def test_service_failure_runs_recovery_and_becomes_ready():
    from urai.service import Service
    from urai.backend import SimulationBackend
    b=SimulationBackend();s=Service(b)
    state=b.state();state['right']['gripper_mm']=.5
    b.state=Mock(return_value=state)
    b.execute=Mock(side_effect=[EmptyGraspError('right',.5),{'completed':True}])
    b.plan_home=Mock(return_value={'arms':{'right':{}},'duration_s':1.})
    s.execution={'state':'running','id':'empty-test'}
    s._run({'arms':{'right':{}},'start':state,'id':'empty-test'})
    assert b.execute.call_count==2
    assert s.execution['state']=='idle'
    assert s.execution['result']['outcome']=='empty_grasp'
    assert s.execution['result']['completed'] is False
