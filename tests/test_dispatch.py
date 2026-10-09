"""Dispatch rules of the PiPER-X executor, exercised on the kinematic simulator.

Every test runs the real :class:`urai.backend.PiperBackend` 50 Hz loop on the wall clock against the simulated
arms, which follow the streamed joint reference at the controller speed the plan selects. Plans are built
straight from joint knots with the planner's own interpolant, so each rule is exercised with exact timing. A
low controller speed makes the simulated arm arrive late, the way the real firmware trails a fast reference.
"""
from __future__ import annotations

import threading
import time
from collections import namedtuple
from itertools import pairwise

import numpy as np
import pytest
from scipy.interpolate import make_interp_spline
from scipy.spatial.transform import Rotation

from urai.backend import Cancelled
from urai.backend.retiming import joint_curve, measured_progress
from urai.service import DispatchCancellation
from urai.settings import motion_limits
from urai.trajectory import compile_arm

Sample = namedtuple('Sample', 'wall t joints opening commanded')


def joint_plan(backend, times, knots, events=(), *, arm='left', stops=(), approach_s=0., profile='normal',
               controller_percent=None):
    """An executable plan for ``arm`` through the joint ``knots`` (n x 6, degrees) at ``times``.

    The reference is the planner's own C2 interpolant (:func:`urai.backend.retiming.joint_curve`), at rest at
    ``stops``; the other arm is not part of the plan and must stay where it is.
    """
    times = np.asarray(times, dtype=float)
    knots = np.asarray(knots, dtype=float)
    model = backend.models[arm]
    bounds = np.array([model.kin.limits_deg[i] for i in range(1, 7)])
    limits = motion_limits(profile)
    if controller_percent is not None:
        limits['controller_speed_percent'] = controller_percent
    return {'duration_s': float(times[-1]), 'approach_duration_s': float(approach_s), 'motion_limits': limits,
            'arms': {arm: {'curve': joint_curve(times, knots, stops, bounds), 'time_s': times, 'joints_deg': knots,
                           'gripper_events': [dict(event) for event in events]}}}


def single_joint(backend, values, *, arm='left', joint=0):
    """Knots that move one joint (0-based index) through ``values`` (deg, relative) from the measured pose."""
    start = backend.runtime.joints_deg(arm)
    knots = np.tile(start, (len(values), 1))
    knots[:, joint] += np.asarray(values, dtype=float)
    return knots


def state_from_joints(models, joints, gripper_mm=0.):
    """A start state in the shape of :meth:`PiperBackend.state` for arms standing at ``joints`` (deg, by arm)."""
    state = {}
    for arm, q in joints.items():
        pose = models[arm].fk_tcp_world(np.asarray(q, dtype=float))
        state[arm] = {'xyz': pose[:3, 3].tolist(),
                      'rpy_deg': Rotation.from_matrix(pose[:3, :3]).as_euler('xyz', degrees=True).tolist(),
                      'joints_deg': np.asarray(q, dtype=float).tolist(), 'gripper_mm': float(gripper_mm)}
    return state


class Recorder:
    """Progress callback of :meth:`PiperBackend.execute` that samples the simulated rig once per control cycle."""

    def __init__(self, runtime, arm='left', on_cycle=None):
        self.runtime = runtime
        self.arm = arm
        self.on_cycle = on_cycle
        self.rows = []
        self.began = time.monotonic()

    def __call__(self, fraction, t):
        self.rows.append(Sample(time.monotonic()-self.began, t, self.runtime.joints_deg(self.arm),
                                self.runtime.gripper_mm(self.arm), self.runtime.last_gripper_command[self.arm]))
        if self.on_cycle is not None:
            self.on_cycle(t)

    def first(self, predicate):
        return next((row for row in self.rows if predicate(row)), None)


def run(backend, plan, cancel=None, recorder=None):
    """Execute ``plan`` from the measured state; returns the result and the recorder."""
    recorder = recorder if recorder is not None else Recorder(backend.runtime)
    recorder.began = time.monotonic()
    return backend.execute(plan, backend.state(), cancel or DispatchCancellation(), recorder), recorder


def drift_deg(runtime, arm='left', settle_s=.3):
    """How far ``arm`` moves within ``settle_s`` of now: zero for an arm that is being held."""
    before = runtime.joints_deg(arm)
    time.sleep(settle_s)
    return float(np.abs(runtime.joints_deg(arm)-before).max())


def test_cancel_before_start_sends_nothing(piper_backend):
    rt = piper_backend.runtime
    start = rt.joints_deg('left')
    plan = joint_plan(piper_backend, [0., 1.], single_joint(piper_backend, [0., 20.]), [{'time_s': 0., 'opening_mm': 50.}])
    cancel = DispatchCancellation()
    cancel.set()
    with pytest.raises(Cancelled):
        run(piper_backend, plan, cancel)
    time.sleep(.3)
    np.testing.assert_array_equal(rt.joints_deg('left'), start)
    assert rt.gripper_mm('left') == 0. and rt.last_gripper_command['left'] is None


def test_cancel_during_motion_holds_the_arm_and_never_sends_the_pending_gripper_event(piper_backend):
    rt = piper_backend.runtime
    plan = joint_plan(piper_backend, [0., 2.], single_joint(piper_backend, [0., 30.]), [{'time_s': 1.5, 'opening_mm': 50.}])
    cancel = DispatchCancellation()
    recorder = Recorder(rt, on_cycle=lambda t: t >= .5 and cancel.set())
    with pytest.raises(Cancelled):
        run(piper_backend, plan, cancel, recorder)
    stopped = rt.joints_deg('left')
    assert 0. < stopped[0] < 15.                       # stopped well short of the 30 degree target
    assert drift_deg(rt) < .05                         # and held there
    assert rt.last_gripper_command['left'] is None and rt.gripper_mm('left') == 0.


def test_cancellation_waits_for_a_write_in_progress_and_refuses_every_later_one():
    cancel = DispatchCancellation()
    writing, finish, order = threading.Event(), threading.Event(), []

    def writer():
        with cancel.dispatch():
            writing.set()
            finish.wait(2)
            order.append('write')

    def canceller():
        cancel.set()
        order.append('cancel')

    threads = [threading.Thread(target=writer)]
    threads[0].start()
    assert writing.wait(2)
    threads.append(threading.Thread(target=canceller))
    threads[1].start()
    time.sleep(.1)
    assert not cancel.is_set()                         # set() waits for the hardware write in progress
    finish.set()
    for thread in threads:
        thread.join(2)
    assert order == ['write', 'cancel'] and cancel.is_set()
    with pytest.raises(Cancelled), cancel.dispatch():
        pass


def test_a_held_execution_lock_refuses_execution_and_preparation_without_any_command(piper_backend):
    rt = piper_backend.runtime
    start = piper_backend.state()
    plan = joint_plan(piper_backend, [0., 1.], single_joint(piper_backend, [0., 20.]), [{'time_s': 0., 'opening_mm': 50.}])
    held, release = threading.Event(), threading.Event()

    def other_writer():
        with rt.execution_lock:
            held.set()
            release.wait(5)

    thread = threading.Thread(target=other_writer)
    thread.start()
    assert held.wait(2)
    try:
        with pytest.raises(ValueError, match='busy'):
            piper_backend.execute(plan, start, DispatchCancellation(), lambda *args: None)
        with pytest.raises(ValueError, match='busy'):
            piper_backend.prepare(DispatchCancellation(), lambda **detail: None)
        time.sleep(.3)
        for arm in ('left', 'right'):
            np.testing.assert_array_equal(rt.joints_deg(arm), start[arm]['joints_deg'])
            assert rt.gripper_mm(arm) == 0. and rt.last_gripper_command[arm] is None
    finally:
        release.set()
        thread.join(2)


@pytest.mark.parametrize('controller_percent,arrives', [(5, True), (1, False)])
def test_task_events_wait_for_measured_arrival_at_the_task_start(piper_backend, controller_percent, arrives):
    """The common task clock stops at the end of the approach until the measured joints reach the task start."""
    rt = piper_backend.runtime
    piper_backend.tracking_limit_deg = 0.
    plan = joint_plan(piper_backend, [0., .5, 1.5], single_joint(piper_backend, [0., 10., 10.]),
                      [{'time_s': .5, 'opening_mm': 50.}], stops=(1,), approach_s=.5, controller_percent=controller_percent)
    recorder = Recorder(rt)
    if arrives:
        result, _ = run(piper_backend, plan, recorder=recorder)
        fired = recorder.first(lambda row: row.commanded == 50.)
        assert result['completed'] and result['approach_wait_s'] > .5
        assert fired.wall > 1. and fired.joints[0] >= 10.-.7  # scheduled for 0.5 s, sent once measured at the start
    else:
        with pytest.raises(RuntimeError, match='Approach start pose not reached'):
            run(piper_backend, plan, recorder=recorder)
        assert rt.last_gripper_command['left'] is None and rt.gripper_mm('left') == 0.
        assert drift_deg(rt) < .05


@pytest.mark.parametrize('opening,holds', [(20., True), (0., False), (68., False)])
def test_the_lift_waits_for_arrival_and_a_holding_opening(piper_backend, opening, holds):
    """A verified grip passes on a measured opening between 1.5 and 67 mm; closed shut or still open stops the carry."""
    rt = piper_backend.runtime
    grip = {'time_s': .5, 'opening_mm': opening, 'hold_s': .8, 'hold_end_time_s': 1.3, 'wait_for_arrival': True,
            'verify': 'holding'}
    plan = joint_plan(piper_backend, [0., .5, 1.3, 1.8], single_joint(piper_backend, [0., 10., 10., 20.]), [grip],
                      stops=(1, 2))
    recorder = Recorder(rt)
    if holds:
        result, _ = run(piper_backend, plan, recorder=recorder)
        gripped = recorder.first(lambda row: row.commanded == opening)
        lifted = recorder.first(lambda row: row.joints[0] > 10.5)
        assert result['completed'] and gripped.joints[0] >= 10.-.7
        assert lifted.wall >= gripped.wall+.8              # the lift waits out the whole grip hold
    else:
        began = time.monotonic()
        with pytest.raises(RuntimeError, match='夹爪疑似空抓'):
            run(piper_backend, plan, recorder=recorder)
        assert time.monotonic()-began > 1.3+3.             # the check is given its full three seconds
        assert max(row.joints[0] for row in recorder.rows) < 10.5   # and the arm never lifted
        assert drift_deg(rt) < .05


def test_a_final_grip_check_cannot_let_the_motion_finish_early(piper_backend):
    grip = {'time_s': .1, 'opening_mm': 0., 'hold_s': .1, 'hold_end_time_s': .2, 'wait_for_arrival': True,
            'verify': 'holding'}
    plan = joint_plan(piper_backend, [0., .1, .2], single_joint(piper_backend, [0., 0., 0.]), [grip], stops=(1,))
    began = time.monotonic()
    with pytest.raises(RuntimeError, match='夹爪'):
        run(piper_backend, plan)
    assert time.monotonic()-began > 3.


def test_an_opening_is_commanded_and_waited_out_never_gated_on_the_measured_width(piper_backend):
    """Fully open jaws report 62-69 mm depending on what sits between the fingers, so an opening is never verified."""
    rt = piper_backend.runtime
    release = {'time_s': .2, 'opening_mm': 70., 'hold_s': .1, 'hold_end_time_s': .3, 'wait_for_arrival': True}
    plan = joint_plan(piper_backend, [0., .2, .3], single_joint(piper_backend, [0., 0., 0.]), [release], stops=(1,))
    result, recorder = run(piper_backend, plan)
    assert result['completed'] and rt.last_gripper_command['left'] == 70.
    assert recorder.rows[-1].opening < 65.             # the plan finished while the jaws were still opening
    spec = {'path': {'mode': 'waypoints', 'points': [[.2, 0, .2], [.3, 0, .2]]}, 'speed': .1,
            'gripper_events': [{'s': 1., 'opening_mm': 70, 'hold_s': .5, 'wait_for_arrival': True, 'verify': 'open'}]}
    with pytest.raises(ValueError, match='verification'):
        compile_arm(spec, piper_backend.state()['left'])


@pytest.mark.parametrize('controller_percent,arrives', [(5, True), (1, False)])
def test_an_endpoint_release_waits_for_the_measured_end_pose(piper_backend, controller_percent, arrives):
    rt = piper_backend.runtime
    piper_backend.tracking_limit_deg = 0.
    release = {'time_s': .3, 'opening_mm': 70., 'hold_s': .6, 'hold_end_time_s': .9, 'wait_for_arrival': True}
    plan = joint_plan(piper_backend, [0., .3, .9], single_joint(piper_backend, [0., 10., 10.]), [release], stops=(1,),
                      controller_percent=controller_percent)
    recorder = Recorder(rt)
    if arrives:
        result, _ = run(piper_backend, plan, recorder=recorder)
        fired = recorder.first(lambda row: row.commanded == 70.)
        assert result['completed'] and fired.wall > 1. and fired.joints[0] >= 10.-.7
    else:
        with pytest.raises(RuntimeError, match='位置未到达'):
            run(piper_backend, plan, recorder=recorder)
        assert rt.last_gripper_command['left'] is None and rt.gripper_mm('left') == 0.


@pytest.mark.parametrize('controller_percent,arrives', [(5, True), (1, False)])
def test_a_pause_waits_for_arrival_holds_and_sends_no_gripper_command(piper_backend, controller_percent, arrives):
    rt = piper_backend.runtime
    piper_backend.tracking_limit_deg = 0.
    pause = {'time_s': .3, 'hold_s': .6, 'hold_end_time_s': .9, 'wait_for_arrival': True}
    plan = joint_plan(piper_backend, [0., .3, .9, 1.2], single_joint(piper_backend, [0., 10., 10., 20.]), [pause],
                      stops=(1, 2), controller_percent=controller_percent)
    recorder = Recorder(rt)
    if arrives:
        result, _ = run(piper_backend, plan, recorder=recorder)
        arrived = recorder.first(lambda row: row.joints[0] >= 10.-.7)
        resumed = recorder.first(lambda row: row.joints[0] > 10.5)
        assert result['completed'] and resumed.wall >= arrived.wall+.6   # the full pause after measured arrival
    else:
        with pytest.raises(RuntimeError, match='位置未到达'):
            run(piper_backend, plan, recorder=recorder)
        assert max(row.joints[0] for row in recorder.rows) < 10.5
    assert all(row.commanded is None for row in recorder.rows) and rt.gripper_mm('left') == 0.


def test_measured_progress_never_moves_backwards_and_picks_the_nearest_reference_time():
    curve = make_interp_spline([0., 1.], np.array([np.zeros(6), np.full(6, 10.)]), k=1)   # 10 deg/s on every joint
    assert measured_progress(curve, np.full(6, 3.), .1, .5) == pytest.approx(.3, abs=.011)
    assert measured_progress(curve, np.zeros(6), .2, .5) == .2
    assert measured_progress(curve, np.full(6, 9.), .2, .5) == .5
    assert measured_progress(curve, np.full(6, 3.), .4, .4) == .4


@pytest.mark.parametrize('on_measured', [False, True])
def test_a_measured_progress_release_waits_for_the_lagging_arm(piper_backend, on_measured):
    piper_backend.tracking_limit_deg = 0.
    release = {'time_s': .5, 'opening_mm': 70., 'lead_s': .05, **({'on_measured': True} if on_measured else {})}
    plan = joint_plan(piper_backend, [0., 1., 1.5], single_joint(piper_backend, [0., 20., 20.]), [release], stops=(1,),
                      controller_percent=5)
    due = float(plan['arms']['left']['curve'](.45)[0])    # where the reference is at the release time
    result, recorder = run(piper_backend, plan)
    fired = recorder.first(lambda row: row.commanded == 70.)
    assert result['completed']
    if on_measured:
        assert fired.joints[0] >= due-.6                  # sent once the lagging arm itself got there
        progress = [row['progress_s']['left'] for row in result['trace']]
        assert max(progress) == pytest.approx(.45, abs=.03)
    else:
        assert fired.wall == pytest.approx(.45, abs=.05) and fired.joints[0] < due-2.
        assert 'progress_s' not in result['trace'][0]


def test_a_gripper_ramp_walks_the_jaws_over_ramp_s(piper_backend):
    events = [{'time_s': 0., 'opening_mm': 70.}, {'time_s': 1., 'opening_mm': 0., 'ramp_s': .4}]
    plan = joint_plan(piper_backend, [0., 1.6], single_joint(piper_backend, [0., 0.]), events)
    result, recorder = run(piper_backend, plan)
    assert result['completed']
    ramp = [row for row in recorder.rows if row.t >= 1.]
    commands = [row.commanded for row in ramp]
    assert commands[0] == pytest.approx(70., abs=4.)     # the walk starts from the measured opening
    assert all(a >= b for a, b in pairwise(commands)) and commands[-1] == 0.
    closed = next(row for row in ramp if row.commanded == 0.)
    assert closed.t-ramp[0].t == pytest.approx(.4, abs=.05)
    assert len(set(commands)) >= 15                    # one intermediate opening per control cycle


def test_an_event_effort_reaches_the_gripper_driver_instead_of_the_runtime_default(piper_backend):
    """The simulated gripper validates the torque limit like the real driver: the runtime's default effort is in
    range, so only a plan's own effort can trip the 0-5000 check."""
    plan = joint_plan(piper_backend, [0., .3], single_joint(piper_backend, [0., 0.]), [{'time_s': 0., 'opening_mm': 30.}])
    assert run(piper_backend, plan)[0]['completed']
    plan = joint_plan(piper_backend, [0., .3], single_joint(piper_backend, [0., 0.]),
                      [{'time_s': 0., 'opening_mm': 0., 'effort': 6000}])
    with pytest.raises(ValueError, match='effort 6000'):
        run(piper_backend, plan)
