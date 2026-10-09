"""Fixtures for tests on the real rig (see docs/hardware.md).

Environment:
    URAI_HARDWARE_TESTS=1       run tests marked ``hardware`` (read-only: CAN feedback and cameras)
    URAI_HARDWARE_MOTION=1      also run tests marked ``motion`` (they take control and move a joint or a gripper)
    URAI_CALIBRATION=<path>     calibration JSON of the rig (required)
    URAI_RIG=<path>             optional JSON overriding RigConfig fields (CAN interfaces, resolutions, ...)

Each test module opens its own runtime (module scope), so only one module holds the CAN channels and
the cameras at a time.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


@pytest.fixture(scope="session")
def hardware_config():
    """``(calibration, rig)`` of the rig under test."""
    from urai.robot.calibration import load_calibration
    from urai.robot.launch import load_rig_config
    from urai.robot.rig import RigConfig

    path = os.environ.get("URAI_CALIBRATION")
    if not path:
        pytest.fail("set URAI_CALIBRATION to the calibration JSON of the rig under test")
    rig = os.environ.get("URAI_RIG")
    return load_calibration(path), load_rig_config(rig) if rig else RigConfig()


@pytest.fixture(scope="session")
def open_robot_runtime(hardware_config):
    """Context manager factory: ``with open_robot_runtime(allow_motion=...) as runtime`` on the real rig."""
    from urai.robot.runtime import build_robot_runtime

    @contextmanager
    def opened(*, allow_motion: bool) -> Iterator:
        calibration, rig = hardware_config
        runtime = build_robot_runtime(calibration, rig, allow_motion=allow_motion)
        try:
            yield runtime
        finally:
            runtime.close()

    return opened
