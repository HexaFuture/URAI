"""Observation while an arm moves, and the hover search of the toss grasp with the PiPER-X inverse kinematics."""
import numpy as np
import pytest
from conftest import wait_until
from scipy.spatial.transform import Rotation

from urai.toss import solve_hover_approach

TOLERANCE = (10., 3.)
GRASPING = [180, 0, 0]          # tool pointing straight down


def test_fixed_camera_observation_accepts_arm_motion(piper_service):
    """The head camera is fixed in the world, so an arm moving during the capture does not void the picture;
    the observation records the joints measured after the capture."""
    s = piper_service
    driver = s.backend.runtime.arms['left']
    target = driver.joints()+[20., 0., 0., 0., 0., 0.]
    driver.command_joints(target)                 # the arm keeps turning for about half a second
    wait_until(lambda: driver.joints()[0] > .5)
    before = driver.joints()[0]
    result = s.observe()
    after = driver.joints()[0]
    recorded = s.observation_start['left']['joints_deg'][0]
    assert result['id'] == s.frame.id and s.execution['state'] == 'idle'
    assert before < recorded <= after < target[0]


@pytest.fixture
def probe(planner):
    """The planner's reach probe for the left arm, recording every pose it is asked about."""
    calls = []

    def reach(xyz, rotation, seed):
        calls.append((np.array(xyz, dtype=float), np.array(rotation, dtype=float)))
        return planner.pose_error('left', xyz, rotation, seed, {'position_mm': TOLERANCE[0], 'orientation_deg': TOLERANCE[1]})
    reach.calls = calls
    return reach


def hover_search(probe, x, hover_z, grasp_z=.03):
    grasp, hover = np.array([x, 0., grasp_z]), np.array([x, 0., hover_z])
    return grasp, hover, solve_hover_approach(grasp, hover, GRASPING, 0., [0., 0.], np.zeros(6), np.zeros(6), probe, TOLERANCE)


def test_a_reachable_hover_keeps_the_grasp_orientation(planner, probe):
    _, hover, (pose, q) = hover_search(probe, .30, .15)
    assert pose == GRASPING and len(probe.calls) == 1
    reached = planner.models['left'].fk_tcp_world(q)
    assert np.linalg.norm(reached[:3, 3]-hover)*1000 <= TOLERANCE[0]


def test_hover_rejects_an_approach_with_an_unreachable_grasp(probe):
    """0.62 m out a tilted hover is reachable, but the vertical grasp below it is not; no hover is accepted
    whose path cannot end in the grasp pose."""
    position, rotation, _ = probe([.62, 0., .03], Rotation.from_euler('xyz', GRASPING, degrees=True).as_matrix(), np.zeros(6))
    assert position > TOLERANCE[0] or rotation > TOLERANCE[1]
    with pytest.raises(ValueError, match='搜索未通过（最佳悬停误差 0 mm / 0°）'):
        hover_search(probe, .62, .12)


def test_hover_tilt_fallback_keeps_the_final_grasp_pose(planner, probe):
    """0.45 m out and 0.15 m up the tool cannot point straight down within tolerance, so the hover tilts; the
    path from the tilted hover down to the grasp is then checked to end in the unchanged vertical grasp pose."""
    grasp, hover, (pose, q) = hover_search(probe, .45, .15)
    target = Rotation.from_euler('xyz', GRASPING, degrees=True).as_matrix()
    tilted = Rotation.from_euler('xyz', pose, degrees=True).as_matrix()
    assert not np.allclose(tilted, target)
    reached = planner.models['left'].fk_tcp_world(q)
    assert np.linalg.norm(reached[:3, 3]-hover)*1000 <= TOLERANCE[0]
    assert np.rad2deg((Rotation.from_matrix(reached[:3, :3]).inv()*Rotation.from_matrix(tilted)).magnitude()) <= TOLERANCE[1]
    np.testing.assert_allclose(probe.calls[-1][0], grasp)
    np.testing.assert_allclose(probe.calls[-1][1], target, atol=1e-10)
