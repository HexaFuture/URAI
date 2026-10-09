"""The planner's IK on the real PiPER-X model: exact angles near the bounds, FK-checked residuals, local-first search."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.backend.model_checks import ik_errors, solve_ik

FAR_SEED = np.array([120., 10., -150., 80., 80., 100.])  # a start from which the local solve cannot reach the target


def pose(xyz, rpy=(180., 0., 0.)):
    target = np.eye(4)
    target[:3, :3] = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
    target[:3, 3] = xyz
    return target


def link6_target(model, target):
    """The link6 pose in the base frame (mm) that the kinematics solver is asked for."""
    base = model.t_base_world @ target
    base[:3, 3] *= 1000.
    return base @ model.t_tcp_link6


def test_valid_ik_near_joint_boundary_keeps_the_solved_angles(arm_models):
    model = arm_models['left']
    q = np.array([0., .075, -.03, 0., 0., 0.])  # j2 and j3 a few hundredths of a degree inside their bounds
    target = model.fk_tcp_world(q)
    solved, pe, re = solve_ik(model, target, np.zeros(6))
    np.testing.assert_allclose(solved, q, atol=1e-6)
    assert pe < 1e-8 and re < 1e-8
    # The arm model's own IK pushes every solution 1 degree inside the bounds, which moves this exact pose.
    pushed, _, _ = model.ik_tcp_world(target, np.zeros(6))
    assert pushed[1] == pytest.approx(1., abs=1e-6) and pushed[2] == pytest.approx(-1., abs=1e-6)


def test_solver_errors_come_from_tcp_forward_kinematics(arm_models):
    model = arm_models['left']
    q = np.array([0., 80., -60., 0., 30., 0.])
    target = model.fk_tcp_world(q)
    target[:3, 3] += [.006, -.008, 0.]
    _, pe, re = ik_errors(model, target, q)
    assert pe == pytest.approx(10., abs=1e-9) and re < 1e-9
    # Beyond reach the best-effort solution is reported with the TCP distance it really has.
    far = pose([.9, 0., .1])
    solved, pe, re = solve_ik(model, far, q)
    assert pe == pytest.approx(np.linalg.norm(model.fk_tcp_world(solved)[:3, 3] - far[:3, 3]) * 1000.)
    assert pe > 100.


def test_solver_preserves_joint_limits(arm_models):
    model = arm_models['left']
    outside = np.array([0., 80., .01, 0., 0., 0.])  # j3 above its 0 degree bound
    with pytest.raises(ValueError, match='joint limit'):
        ik_errors(model, model.fk_tcp_world(np.clip(outside, None, [150, 180, 0, 89, 89, 120])), outside)
    lower = np.array([model.kin.limits_deg[i][0] for i in range(1, 7)])
    upper = np.array([model.kin.limits_deg[i][1] for i in range(1, 7)])
    for xyz in ([.9, 0., .1], [0., 0., .8], [-.5, .3, -.2]):
        solved, _, _ = solve_ik(model, pose(xyz), np.zeros(6))
        assert np.all(solved >= lower) and np.all(solved <= upper)


def test_accepted_local_solution_does_not_search_other_seeds(arm_models):
    model = arm_models['left']
    target = pose([.35, .05, .10])
    local = model.kin.ik_best_effort(link6_target(model, target), seed_deg=FAR_SEED, max_restarts=0)
    assert local[1] > 100.  # the local solve from this start stays far from the target
    # A TCP tolerance that already admits the local result returns it as is, without any restart.
    solved, pe, _ = solve_ik(model, target, FAR_SEED, {'position_mm': 1000., 'orientation_deg': 180.})
    np.testing.assert_array_equal(solved, local[0])
    assert pe > 100.


def test_local_solution_outside_tcp_tolerance_falls_back_to_full_search(arm_models):
    model = arm_models['left']
    target = pose([.35, .05, .10])
    full = model.kin.ik_best_effort(link6_target(model, target), seed_deg=FAR_SEED, max_restarts=6)
    solved, pe, re = solve_ik(model, target, FAR_SEED)
    np.testing.assert_array_equal(solved, full[0])
    assert pe < 1e-6 and re < 1e-6
    np.testing.assert_allclose(model.fk_tcp_world(solved), target, atol=1e-9)


def test_solver_ignores_the_forward_facing_rules(arm_models):
    """An overhand wind-up turns j1 past the side line; the planner's IK must still return it."""
    model = arm_models['left']
    q = np.array([120., 60., -60., 0., 30., 0.])
    assert model.front_violation(q) is not None
    target = model.fk_tcp_world(q)
    solved, pe, re = solve_ik(model, target, q)
    np.testing.assert_allclose(solved, q, atol=1e-6)
    assert pe < 1e-8 and re < 1e-8
    rejected, _, _ = model.ik_tcp_world(target, q)
    assert rejected is None and model.last_reject_reason is not None
