"""A grasp line far from the base leans toward it; the arm can only reach it that way. Nothing moves."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.backend import SimulationBackend
from urai.grasp import grasp_from_line
from urai.transfer import MAX_GRASP_TILT_DEG, grasp_orientation, grasp_tilt_deg

TOLERANCE = {'position_mm': 30., 'orientation_deg': 60.}
EXACT_MM, EXACT_DEG = 3., 3.
LINE = [[380, 220], [410, 220]]


def line_pose(frame, pixels, base_xy):
    result = grasp_from_line(frame, pixels, inset_mm=5, base_xy=base_xy)
    keyframes = result['arm']['orientation']['points']
    assert keyframes[0][1:] == keyframes[1][1:]
    return np.asarray(result['arm']['path']['points'][-1]), keyframes[0][1:]


def test_a_grasp_line_close_to_the_base_still_points_straight_down():
    target, pose = line_pose(SimulationBackend().capture(), LINE, base_xy=(.37, -.21))
    assert grasp_tilt_deg(target[:2], (.37, -.21)) == 0.
    assert pose[0] == 180 and pose[1] == 0


def test_a_grasp_line_far_from_the_base_leans_the_wrist_in():
    target, pose = line_pose(SimulationBackend().capture(), LINE, base_xy=(-.35, 0.))
    lean = grasp_tilt_deg(target[:2], (-.35, 0.))
    assert lean > 0
    direction = np.asarray(target[:2])-np.asarray([-.35, 0.])
    axis = Rotation.from_euler('xyz', pose, degrees=True).as_matrix()[:, 2]
    # The tool axis is no longer vertical: it leans by the reach-derived angle, toward the base.
    assert np.degrees(np.arccos(np.clip(-axis[2], -1, 1))) == pytest.approx(lean, abs=1e-6)
    assert axis[:2] @ (direction/np.linalg.norm(direction)) > 0


def test_the_left_arm_reaches_a_far_grasp_line_only_when_it_leans(planner, working_start):
    """A grasp line 0.62 m from the left base, jaws across it: exact when leaning, far off when upright."""
    target = np.array([.5551, -.2817, .0236])
    base = np.asarray(planner.models['left'].base_xy, dtype=float)
    assert np.linalg.norm(target[:2]-base) > .6 and grasp_tilt_deg(target[:2], base) == MAX_GRASP_TILT_DEG
    seed = working_start['left']['joints_deg']
    leaning = Rotation.from_euler('xyz', grasp_orientation(target, -90., base), degrees=True).as_matrix()
    upright = Rotation.from_euler('xyz', [180., 0., -90.], degrees=True).as_matrix()
    lean_error = planner.pose_error('left', target, leaning, seed, TOLERANCE)
    upright_error = planner.pose_error('left', target, upright, seed, TOLERANCE)
    assert lean_error[0] <= EXACT_MM and lean_error[1] <= EXACT_DEG
    assert upright_error[1] > 15.
