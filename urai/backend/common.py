"""Names shared by every backend: arm names, homing defaults and the planning error types."""
from __future__ import annotations

ARMS = ('left', 'right')
#: Requested TCP speed of a homing move; the motion profile's joint limits and every geometry check still apply.
HOME_SPEED_M_S = .8
HOME_OPENING_MM = 70.
HOME_HOLD_S = .6


def home_gripper_event():
    """Open at the start pose and wait for measured arrival before moving away."""
    return {'s': 0., 'opening_mm': HOME_OPENING_MM, 'hold_s': HOME_HOLD_S, 'wait_for_arrival': True}


def home_note(arms, target):
    names = '、'.join('左臂' if arm == 'left' else '右臂' for arm in arms)
    return (f'回原位（{names}）：先在当前位置张开夹爪 {HOME_OPENING_MM:g} mm，再{target}；'
            '未选中的臂保持不动并参与碰撞检查。')


class Cancelled(Exception):
    pass


class RobotStateError(ValueError):
    """Robot readiness failures cannot be repaired by another geometric route."""

    def __init__(self, message, reason=None):
        super().__init__(message)
        self.reason = reason


class FloorError(ValueError):
    """The requested path itself dips below the table floor; no approach route can repair it."""


class ApproachError(ValueError):
    """One arm's own approach segment failed (IK or pose tolerance); other routes for that arm may still work."""

    def __init__(self, message, arm):
        super().__init__(message)
        self.arm = arm

    def __reduce__(self):
        # Worker processes raise this too; keep the arm across pickling.
        return ApproachError, (str(self), self.arm)


class TaskError(ValueError):
    """The user's task segment itself failed; no approach route can repair it."""
