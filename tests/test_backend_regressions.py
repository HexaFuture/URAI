"""Regression tests for the PiPER-X executor."""
from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from urai.backend import PiperBackend
from urai.backend.piperx import scheduler_stall


def test_a_stall_before_the_first_control_cycle_is_reported_at_t_zero():
    assert scheduler_stall(.3, 0., PiperBackend.scheduler_gap_s) == \
        'Trajectory scheduler stalled for 0.300s at t=0.00s (limit 0.25s)'


@pytest.mark.parametrize('gap,stalled', [(.02, False), (.2, False), (.25, False), (.3, True), (1.7, True)])
def test_the_scheduler_gap_limit_is_a_quarter_second(gap, stalled):
    assert PiperBackend.scheduler_gap_s == .25
    message = scheduler_stall(gap, 1.5, PiperBackend.scheduler_gap_s)
    assert (message is not None) is stalled
    if stalled:
        assert f'stalled for {gap:.3f}s at t=1.50s' in message


def _assigns_t(node):
    return isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and any(
        isinstance(target, ast.Name) and target.id == 't'
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))


def test_the_control_loop_reads_a_reference_clock_that_exists_before_its_first_cycle():
    """The stall check of the first cycle runs before the loop assigns ``t``.

    A thread preempted for more than the gap limit between starting the clock and the first cycle then reported
    the stall with an unassigned ``t`` (an UnboundLocalError instead of the stall). The window cannot be hit on
    purpose, so the source is checked: ``t`` is assigned at function level before the loop starts.
    """
    function = ast.parse(textwrap.dedent(inspect.getsource(PiperBackend._execute))).body[0]
    loop = next(node for node in ast.walk(function)
                if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True)
    stall = next(node for node in ast.walk(loop) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'scheduler_stall')
    assert any(isinstance(arg, ast.Name) and arg.id == 't' for arg in stall.args)
    first_in_loop = min(node.lineno for node in ast.walk(loop) if _assigns_t(node))
    assert stall.lineno < first_in_loop                # the first cycle reads t before setting it
    before = [node for node in function.body if _assigns_t(node) and node.lineno < loop.lineno]
    assert before, '_execute must assign t before its control loop'
