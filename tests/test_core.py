"""The trajectory description: bounded expressions, metric paths, timing keyframes and picking pixels."""
import numpy as np
import pytest


def test_expression_is_bounded_math_not_python():
    from urai.trajectory import expression
    np.testing.assert_allclose(expression('0.1 + 0.02*sin(pi*s)', np.array([0, .5, 1])), [.1, .12, .1])
    for bad in ['__import__("os").system("id")', 's.__class__', '[s for s in range(2)]', '9**99999', 'NaN',
                'maximum(s, 0.5, s)', 'sin(s,s)']:
        with pytest.raises(ValueError):
            expression(bad, np.linspace(0, 1, 10))


def test_waypoint_and_function_share_metric_path():
    from urai.trajectory import compile_arm
    state = {'xyz': [.2, 0, .2], 'rpy_deg': [180, 0, 0]}
    base = {'speed': '.02', 'orientation': {'mode': 'hold'}}
    a = compile_arm({**base, 'path': {'mode': 'function', 'x': '.2+.1*s', 'y': '0', 'z': '.2'}}, state)
    b = compile_arm({**base, 'path': {'mode': 'waypoints', 'points': [[.2, 0, .2], [.3, 0, .2]]}}, state)
    np.testing.assert_allclose(a['xyz'], b['xyz'], atol=1e-8)
    assert a['requested_duration_s'] == pytest.approx(5, abs=.02)
    assert a['time_s'][-1] >= 5  # endpoint acceleration ramps
    assert np.all(np.diff(a['time_s']) > 0)


def test_orientation_only_action_gets_real_duration():
    from urai.trajectory import compile_arm
    a = compile_arm({'path': {'mode': 'function', 'x': '.2', 'y': '0', 'z': '.2'}, 'speed': '.03',
                     'orientation': {'mode': 'function', 'roll': '0', 'pitch': '0', 'yaw': '90*s'}},
                    {'xyz': [.2, 0, .2], 'rpy_deg': [0, 0, 0]})
    assert a['time_s'][-1] == pytest.approx(3, abs=1e-9)


def test_invalid_speed_and_nonfinite_path_rejected():
    from urai.trajectory import compile_arm
    state = {'xyz': [.2, 0, .2], 'rpy_deg': [0, 0, 0]}
    for speed in ['0', '-1', '1/0', '1000']:
        with pytest.raises(ValueError):
            compile_arm({'path': {'mode': 'function', 'x': '.2+s', 'y': '0', 'z': '.2'}, 'speed': speed}, state)


def test_depth_and_plane_picks_use_same_world_transform():
    from urai.geometry import Frame
    k = np.array([[100, 0, 2], [0, 100, 2], [0, 0, 1.]])
    t = np.eye(4); t[:3, 3] = [.1, .2, .3]
    f = Frame(np.zeros((5, 5, 3), dtype=np.uint8), np.ones((5, 5)), k, t)
    np.testing.assert_allclose(f.pick(2, 2, mode='surface'), [.1, .2, 1.3])
    np.testing.assert_allclose(f.pick(2, 2, mode='plane', z=.8), [.1, .2, .8])
    np.testing.assert_allclose(f.project([[.1, .2, 1.3]]), [[2, 2]])
    f.depth[:] = 0
    with pytest.raises(ValueError, match='depth'):
        f.pick(2, 2, mode='surface')


def test_speed_keyframes_and_orientation_keyframes():
    from urai.trajectory import compile_arm
    a = compile_arm({'path': {'mode': 'waypoints', 'points': [[.2, 0, .2], [.25, 0, .2]]},
                     'speed': [[0, .01], [.5, .03], [1, .01]],
                     'orientation': {'mode': 'keyframes', 'points': [[0, 0, 0, 170], [1, 0, 0, -170]]}},
                    {'xyz': [.2, 0, .2], 'rpy_deg': [0, 0, 170]})
    assert a['requested_speed_m_s'][len(a['s'])//2] == pytest.approx(.03, abs=.001)
    assert np.abs(a['rotation_matrices'][0] - a['rotation_matrices'][-1]).max() < 1


def test_a_long_uneven_waypoint_path_is_refined_instead_of_refused():
    """A 1.7 m stroke through six far-apart points, without a gripper event.

    A fixed sample grid would leave 22 mm between neighbours and the preview would refuse it; bisecting the path
    costs a few hundred samples and keeps every neighbour inside the 10 mm the preview checks.
    """
    import numpy as np
    from urai.trajectory import POSITION_STEP_M, compile_arm
    points = [[.2386, 0., .2292], [.3204, -.0655, .0206], [.3724, -.6401, .0422],
              [.3429, -.0468, .0059], [.3420, -.0672, .0159], [.3759, -.3491, .0048]]
    state = {'xyz': [.2386, 0., .2292], 'rpy_deg': [180, 0, 0]}
    plan = compile_arm({'path': {'mode': 'waypoints', 'points': points}, 'speed': '.03',
                        'orientation': {'mode': 'free'}}, state)
    steps = np.linalg.norm(np.diff(plan['xyz'], axis=0), axis=1)
    assert steps.max() <= POSITION_STEP_M+1e-9 and len(plan['xyz']) > 201
    # Every control point is still on the compiled path: refinement only adds samples between them.
    for point in points:
        assert np.min(np.linalg.norm(plan['xyz']-np.asarray(point), axis=1)) < POSITION_STEP_M


def test_a_function_path_that_cannot_be_resolved_says_by_how_much():
    import pytest
    from urai.trajectory import compile_arm
    state = {'xyz': [.2, 0, .2], 'rpy_deg': [180, 0, 0]}
    with pytest.raises(ValueError, match='相邻采样点最大'):
        compile_arm({'path': {'mode': 'function', 'x': '.2+.1*sin(200*s)', 'y': '0', 'z': '.2'},
                     'speed': '.02', 'orientation': {'mode': 'hold'}}, state)
