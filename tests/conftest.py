"""Shared fixtures. Everything here is the real thing: the real arm models built from the example calibration,
the real backends (the synthetic one and the PiPER-X backend on the kinematic simulator), and the real HTTP app."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def make_client(service, token=None, host='127.0.0.1'):
    """A TestClient for the real app. Without a token the app answers loopback Host names only."""
    from urai.app import create_app
    return TestClient(create_app(service, token=token), base_url=f'http://{host}')


@pytest.fixture(scope='session')
def example_calibration():
    from urai.robot.calibration import example_calibration_path, load_calibration
    return load_calibration(example_calibration_path())


@pytest.fixture(scope='session')
def arm_models(example_calibration):
    """Both PiPER-X arm models of the example rig (world frame = left arm base)."""
    from urai.robot.rig import RigConfig, build_models
    return build_models(example_calibration, RigConfig())


@pytest.fixture
def planner(arm_models):
    """The planning half of the PiPER-X backend on the example rig's models; nothing touches hardware."""
    from urai.backend import PiperPlanner
    return PiperPlanner(arm_models)


@pytest.fixture
def sim_runtime():
    """The kinematic simulator runtime (simulated arms, synthetic RGB-D scene)."""
    from urai.robot.simulated import build_simulated_runtime
    runtime = build_simulated_runtime()
    yield runtime
    runtime.close()


@pytest.fixture
def piper_backend(sim_runtime):
    """The real PiPER-X backend driving the kinematic simulator."""
    from urai.backend import PiperBackend
    backend = PiperBackend(sim_runtime)
    yield backend
    backend.close()


@pytest.fixture
def sim_service():
    """A service on the synthetic simulation backend."""
    from urai.backend import SimulationBackend
    from urai.service import Service
    return Service(SimulationBackend())


@pytest.fixture
def piper_service(piper_backend):
    from urai.service import Service
    return Service(piper_backend)


# ---------------------------------------------------------------------------- hardware tests
# Tests marked ``hardware`` talk to the real rig (PiPER arms on CAN, RealSense cameras) and are skipped unless
# URAI_HARDWARE_TESTS=1. Tests that also move an arm or a gripper are marked ``motion`` and additionally need
# URAI_HARDWARE_MOTION=1. docs/hardware.md lists the required set-up.

HARDWARE_ENV = 'URAI_HARDWARE_TESTS'
MOTION_ENV = 'URAI_HARDWARE_MOTION'


def pytest_configure(config):
    config.addinivalue_line('markers', f'hardware: needs the real rig; runs only with {HARDWARE_ENV}=1')
    config.addinivalue_line('markers', f'motion: moves an arm or a gripper; also needs {MOTION_ENV}=1')


def pytest_collection_modifyitems(config, items):
    import os
    hardware = os.environ.get(HARDWARE_ENV) == '1'
    motion = os.environ.get(MOTION_ENV) == '1'
    skip_hardware = pytest.mark.skip(reason=f'real rig required: set {HARDWARE_ENV}=1')
    skip_motion = pytest.mark.skip(reason=f'moves the robot: set {MOTION_ENV}=1 as well')
    for item in items:
        if 'hardware' in item.keywords and not hardware:
            item.add_marker(skip_hardware)
        elif 'motion' in item.keywords and not motion:
            item.add_marker(skip_motion)


# ---------------------------------------------------------------------------- planning and skill tests
#: Joints (deg) of both arms reaching out over the table, the way they stand after a grasp. Planning tests start
#: from here and seed their inverse kinematics with it, so the solver's branch choice is reproducible.
WORKING_JOINTS_DEG = {'left': (-34.26, 148.06, -119.95, 13.74, 2.71, 20.6),
                      'right': (21.6, 102.27, -64.78, 57.42, 0., 29.09)}


def arm_state(model, joints_deg, gripper_mm=0.):
    """One arm's entry of ``backend.state()`` at ``joints_deg``, computed with the arm model's forward kinematics."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    joints = np.asarray(joints_deg, dtype=float)
    pose = model.fk_tcp_world(joints)
    return {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
            'joints_deg': joints.tolist(), 'gripper_mm': float(gripper_mm), 'enabled': True, 'feedback_age_s': 0.,
            'ctrl_mode': 'CAN'}


@pytest.fixture
def working_start(arm_models):
    """``backend.state()`` of the example rig with both arms at :data:`WORKING_JOINTS_DEG`."""
    return {arm: arm_state(arm_models[arm], WORKING_JOINTS_DEG[arm]) for arm in ('left', 'right')}


@pytest.fixture
def locked_piper_service():
    """A service on the PiPER-X backend over a kinematic simulator started without motion permission
    (what ``urai-robot`` does without ``--allow-motion``): planning works, every hardware write is refused."""
    from urai.backend import PiperBackend
    from urai.robot.simulated import build_simulated_runtime
    from urai.service import Service
    runtime = build_simulated_runtime(allow_motion=False)
    backend = PiperBackend(runtime)
    yield Service(backend)
    backend.close()
    runtime.close()


# ---------------------------------------------------------------------------- service-level test helpers

def wait_until(predicate, timeout=10., interval=.005):
    """Poll ``predicate`` until it holds; fail the test after ``timeout`` seconds."""
    import time
    deadline = time.monotonic()+timeout
    while not predicate():
        assert time.monotonic() < deadline, 'condition not reached in time'
        time.sleep(interval)


def line_draft(service, arms, offset, speed=.02, **spec):
    """A draft moving each of ``arms`` in a straight line from where it stands now by ``offset`` (m), tool
    orientation held; ``spec`` adds fields to every arm's entry (for example ``gripper_events``)."""
    import numpy as np
    state = service.backend.state()
    return {'observation_id': service.frame.id, 'arms': {arm: {
        'path': {'mode': 'waypoints', 'points': [state[arm]['xyz'], np.add(state[arm]['xyz'], offset).tolist()]},
        'speed': speed, 'orientation': {'mode': 'hold'}, **spec} for arm in arms}}


def stop_midway(service, distance=.02, arm='left'):
    """Execute a slow 5 cm move of ``arm`` and cancel it once the arm has travelled ``distance`` metres.

    The draft of the stopped motion stays in place, as it does after any motion that did not complete.
    """
    import numpy as np

    def tcp():
        return np.asarray(service.backend.state()[arm]['xyz'])

    start = tcp()
    service.set_draft(line_draft(service, [arm], [.05, 0., 0.]))
    service.execute(service.preview()['id'])
    wait_until(lambda: np.linalg.norm(tcp()-start) >= distance)
    service.cancel()
    service.worker.join(10)
    assert service.execution['state'] == 'cancelled'


@pytest.fixture
def client_for():
    """:func:`make_client` as a fixture: once tests/hardware/conftest.py has been imported, the module name
    ``conftest`` no longer refers to this file, so test modules should not import it by name."""
    return make_client
