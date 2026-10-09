"""An actively commanded holder during a twist: only a stationary, closed-jaw reference is accepted."""
import copy
import pytest
from test_bottle_cap import ready_to_twist
from urai.skills.bottle_cap import held_session, stage_metadata, validate_stage


def held_plan():
    service, ctx, cap, pixel = ready_to_twist()
    current = service.backend.state()
    held = current['left']
    metadata = {'stage': 'twist', 'holder': 'left', 'worker': 'right', 'active_holder': True, 'id': service.bottle_cap['id']}
    holder = {'path': {'mode': 'waypoints', 'points': [held['xyz'], held['xyz']]},
              'orientation': {'mode': 'keyframes', 'points': [[0, *held['rpy_deg']], [1, *held['rpy_deg']]]},
              'gripper_events': []}
    return service, current, {'input_draft': {'arms': {'right': {'bottle_cap': metadata}, 'left': holder}}}


def test_stationary_active_holder_is_accepted():
    service, current, plan = held_plan()
    assert validate_stage(service, plan, current)['active_holder']


def test_moving_holder_is_rejected():
    service, current, plan = held_plan()
    plan = copy.deepcopy(plan)
    plan['input_draft']['arms']['left']['path']['points'][1][0] += .02
    with pytest.raises(ValueError, match='保持当前姿态'):
        validate_stage(service, plan, current)


def test_opening_holder_is_rejected():
    service, current, plan = held_plan()
    plan['input_draft']['arms']['left']['gripper_events'] = [{'s': 0, 'opening_mm': 70}]
    with pytest.raises(ValueError, match='不得松爪'):
        validate_stage(service, plan, current)


def test_unmarked_extra_arm_is_rejected():
    service, current, plan = held_plan()
    plan['input_draft']['arms']['right']['bottle_cap'].pop('active_holder')
    with pytest.raises(ValueError, match='不能与其他轨迹'):
        stage_metadata(plan)


def test_contact_compliance_keeps_bounded_motion_guards():
    service, current, plan = held_plan()
    current = copy.deepcopy(current)
    current['left']['joints_deg'][0] += .7
    current['left']['xyz'][0] += .004
    with pytest.raises(ValueError):
        held_session(service, current)
    assert held_session(service, current, active_hold=True)
    current['left']['xyz'][0] += .01
    with pytest.raises(ValueError):
        held_session(service, current, active_hold=True)
