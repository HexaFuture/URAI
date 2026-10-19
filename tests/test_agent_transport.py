"""Public runtime integration: authenticated HTTP and camera clock conversion."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import numpy as np
from conftest import make_client
from urai.backend import PiperBackend, SimulationBackend
from urai.service import Service


def test_head_exposure_preserves_age_across_clock_domains():
    backend = object.__new__(PiperBackend)
    backend.capture_lock = threading.Lock()
    frame = SimpleNamespace(rgb=np.zeros((2,2,3),dtype=np.uint8), depth_m=np.ones((2,2)),
        k=np.eye(3), t_world_cam=np.eye(4), timestamp=time.time()-.25, seq=12)
    backend.runtime = SimpleNamespace(capture=lambda camera: frame)
    captured = backend.capture()
    assert captured.sensor_sequence == 12
    assert .20 <= time.monotonic()-captured.sensor_timestamp <= 1.
    frame.timestamp=0.;frame.seq=-1
    captured = backend.capture()
    assert captured.sensor_timestamp is None and captured.sensor_sequence is None


def test_new_agent_routes_require_the_robot_token():
    service=Service(SimulationBackend())
    client=make_client(service,token='test-robot-token')
    for path in ['/api/skills/pick_place/run','/api/pour-agent/start']:
        response=client.post(path,json={})
        assert response.status_code==401
    assert service.execution['state']=='idle'


def test_pour_driver_sends_robot_auth_for_reads_and_writes(monkeypatch):
    path=Path(__file__).parents[1]/'scripts/pour_agent.py'
    spec=importlib.util.spec_from_file_location('pour_transport',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    seen=[]
    response=SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'ok':True})
    def request(url,**kwargs):
        seen.append((url,kwargs));return response
    monkeypatch.setattr(module.requests,'get',request)
    monkeypatch.setattr(module.requests,'post',request)
    robot=module.Robot('http://127.0.0.1:7860','test-robot-token')
    robot.state();robot.post('pour-agent/start',{'arm':'left'})
    assert len(seen)==2
    assert all(kwargs['headers']=={'Authorization':'Bearer test-robot-token'} for _,kwargs in seen)
