"""The worker pool reproduces the in-process IK chains and geometry sweep of the real arm models bit for bit."""
import os
import pickle
import signal
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.backend import ApproachError, PiperPlanner, TaskError
from urai.backend.model_checks import measured_joint_bounds
from urai.backend.retiming import joint_curve
from urai.parallel import PlannerPool, build_model, model_spec
from urai.robot.arm_model import inter_arm_reason
from urai.trajectory import compile_arms

SAFE_POSE = np.array([0., 40., -60., 0., 20., 0.])
SELF_COLLISION_POSE = np.array([34.6, 69.1, -.5, 85.6, 33., 36.1])  # tool body inside the arm's own base column
REACH_ACROSS = {'left': np.array([-90., 90., -90., 0., 0., 0.]), 'right': np.array([90., 90., -90., 0., 0., 0.])}  # forearms meet
SCENARIOS = {'self-collision': (SELF_COLLISION_POSE, SAFE_POSE),
             'inter-arm': (REACH_ACROSS['left'], REACH_ACROSS['right']),
             'joint-limit': (SAFE_POSE + [0., 160., 0., 0., 0., 0.], SAFE_POSE)}
REASONS = {'self-collision': 'self-collision', 'inter-arm': 'would hit the right arm', 'joint-limit': 'joint limit exceeded'}


@pytest.fixture(scope='module')
def pooled(arm_models):
    """A planner whose IK chains and collision sweep run in three worker processes."""
    planner = PiperPlanner(arm_models, workers=3)
    planner.warm_up()
    yield planner
    planner.close()


def serial_and_pooled(arm_models, pooled, table_checks):
    serial = PiperPlanner(arm_models)
    for planner in (serial, pooled):
        planner.table_checks = table_checks
    return serial, pooled


def limits(models):
    kin = models['left'].kin
    return np.array([kin.limits_deg[i][0] for i in range(1, 7)]), np.array([kin.limits_deg[i][1] for i in range(1, 7)])


def ramp(target, knots=60):
    """A straight joint-space ramp from the safe pose to ``target`` over 11.8 seconds."""
    times = np.linspace(0., .2 * (knots - 1), knots)
    return times, SAFE_POSE + (np.asarray(target) - SAFE_POSE) * np.linspace(0., 1., knots)[:, None]


def gentle_motion(knots=60):
    times = np.linspace(0., .2 * (knots - 1), knots)
    qs = SAFE_POSE + 3. * np.sin(np.linspace(0., np.pi, knots))[:, None] * np.array([1., .5, -.5, 0., 1., 2.])
    return times, qs


def compiled_for(times, left, right, offset):
    return {arm: {'curve': joint_curve(times, qs), 'time_s': times, 'task_offset_s': offset}
            for arm, qs in (('left', left), ('right', right))}


def sweep(planner, compiled, grid, initial_bounds=None):
    start = {arm: {'joints_deg': compiled[arm]['curve'](0.)} for arm in compiled}
    try:
        planner._check_joint_samples(compiled, start, grid, initial_bounds, lambda **detail: None)
    except ValueError as exc:
        return type(exc), str(exc)
    return None


def start_state(models, joints):
    state = {}
    for arm, q in joints.items():
        pose = models[arm].fk_tcp_world(q)
        state[arm] = {'xyz': pose[:3, 3].tolist(), 'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
                      'joints_deg': list(q), 'gripper_mm': 0.}
    return state


def dual_arm_specs(state, reach=.05):
    specs = {}
    for arm in ('left', 'right'):
        x, y, z = state[arm]['xyz']
        specs[arm] = {'path': {'mode': 'waypoints', 'points': [[x + .03, y, z + .02], [x + .03 + reach, y, z + .02]]},
                      'speed': .05, 'gripper_events': [{'s': .5, 'opening_mm': 40., 'hold_s': .6}]}
    return specs


def full_plan(planner, specs, state):
    try:
        return planner.plan(compile_arms(specs, state), state), None
    except ValueError as exc:
        return None, (type(exc), str(exc))


def test_pooled_plan_matches_the_in_process_plan_bitwise(arm_models, pooled):
    serial, pooled = serial_and_pooled(arm_models, pooled, table_checks=True)
    state = start_state(arm_models, {'left': SAFE_POSE, 'right': SAFE_POSE})
    specs = dual_arm_specs(state)
    expected, serial_error = full_plan(serial, specs, state)
    actual, pooled_error = full_plan(pooled, specs, state)
    assert serial_error is None and pooled_error is None
    assert actual['approach_routes'] == expected['approach_routes'] and actual['duration_s'] == expected['duration_s']
    for arm in ('left', 'right'):
        np.testing.assert_array_equal(actual['arms'][arm]['joints_deg'], expected['arms'][arm]['joints_deg'])
        np.testing.assert_array_equal(actual['arms'][arm]['time_s'], expected['arms'][arm]['time_s'])
        assert actual['arms'][arm]['gripper_events'] == expected['arms'][arm]['gripper_events']
    assert actual['notes'] == expected['notes']


def test_pooled_task_failures_carry_the_serial_type_and_message(arm_models, pooled):
    serial, pooled = serial_and_pooled(arm_models, pooled, table_checks=True)
    state = start_state(arm_models, {'left': SAFE_POSE, 'right': SAFE_POSE})
    specs = dual_arm_specs(state, reach=1.2)  # far beyond the workspace: the task segment fails
    expected = full_plan(serial, specs, state)[1]
    assert expected is not None and expected[0] is TaskError
    assert full_plan(pooled, specs, state)[1] == expected


def test_approach_errors_survive_pickling():
    error = pickle.loads(pickle.dumps(ApproachError('left approach: pose error', 'left')))
    assert isinstance(error, ApproachError) and error.arm == 'left' and str(error) == 'left approach: pose error'


def test_rebuilt_models_give_the_same_geometry_answers(arm_models):
    rebuilt = {arm: build_model(model_spec(model)) for arm, model in arm_models.items()}
    lower, upper = limits(arm_models)
    rng = np.random.default_rng(7)
    for _ in range(40):
        qa, qb = rng.uniform(lower, upper, size=(2, 6))
        for arm, q in (('left', qa), ('right', qb)):
            np.testing.assert_array_equal(rebuilt[arm].kin.link_origins(q), arm_models[arm].kin.link_origins(q))
            np.testing.assert_array_equal(rebuilt[arm].fk_tcp_world(q), arm_models[arm].fk_tcp_world(q))
            assert rebuilt[arm].table_check(q) == arm_models[arm].table_check(q)
            assert rebuilt[arm].self_collision_clearance(q) == arm_models[arm].self_collision_clearance(q)
        assert inter_arm_reason(rebuilt['left'], qa, rebuilt['right'], qb) == \
            inter_arm_reason(arm_models['left'], qa, arm_models['right'], qb)


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
@pytest.mark.parametrize('offset,kind', [(0., TaskError), (100., ValueError)])
def test_pool_sweep_matches_the_serial_sweep(arm_models, pooled, scenario, offset, kind):
    serial, pooled = serial_and_pooled(arm_models, pooled, table_checks=False)
    times, left = ramp(SCENARIOS[scenario][0])
    _, right = ramp(SCENARIOS[scenario][1])
    compiled = compiled_for(times, left, right, offset=offset)
    grid = np.linspace(0., times[-1], 500)
    expected = sweep(serial, compiled, grid)
    assert expected is not None and expected[0] is kind
    assert REASONS[scenario] in expected[1]
    assert sweep(pooled, compiled, grid) == expected


def test_pool_sweep_passes_a_clean_trajectory(arm_models, pooled):
    serial, pooled = serial_and_pooled(arm_models, pooled, table_checks=True)
    times, left = gentle_motion()
    compiled = compiled_for(times, left, left, offset=3.)
    grid = np.linspace(0., times[-1], 400)
    assert sweep(serial, compiled, grid) is None
    assert sweep(pooled, compiled, grid) is None


def test_pool_sweep_honours_initial_joint_bounds(arm_models, pooled):
    serial, pooled = serial_and_pooled(arm_models, pooled, table_checks=False)
    times, left = gentle_motion()
    left[:, 1] += 200.  # clearly outside the nominal j2 range
    compiled = compiled_for(times, left, left, offset=3.)
    grid = np.linspace(0., times[-1], 200)
    bounds = {arm: measured_joint_bounds(arm_models[arm], left[0]) for arm in arm_models}
    expected = sweep(serial, compiled, grid, bounds)
    assert expected is not None and 'planned joint limit exceeded' in expected[1]
    assert sweep(pooled, compiled, grid, bounds) == expected


def test_a_dead_worker_is_reported_once_and_the_pool_is_rebuilt(arm_models):
    pool = PlannerPool(arm_models, workers=2)
    pool.warm_up()
    try:
        os.kill(pool.worker_pids()[0], signal.SIGKILL)   # one of this pool's own worker processes
        time.sleep(.2)
        _, left = gentle_motion()
        with pytest.raises(ValueError, match='工作进程'):
            pool.check(left, left, False, None)
        assert pool.check(left, left, False, None) is None
    finally:
        pool.close()
