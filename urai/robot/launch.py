"""``urai-robot``: serve URAI on the dual PiPER-X rig, or on its kinematic simulator with ``--sim``.

Examples::

    urai-robot --calibration my_calibration.json                 # observe only, motion gate closed
    urai-robot --calibration my_calibration.json --allow-motion  # the console may move the arms
    urai-robot --sim                                            # simulated arms, synthetic cameras

Without ``--allow-motion`` the arm drivers are read-only: the console can observe and plan, and every
request that would move an arm or a gripper is refused. Start-up never moves an arm.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import threading
from collections.abc import Sequence

from .calibration import example_calibration_path, load_calibration
from .rig import RigConfig
from .runtime import RobotRuntime

__all__ = ["build_parser", "build_runtime", "load_rig_config", "main"]


def load_rig_config(path: pathlib.Path | str) -> RigConfig:
    """A :class:`RigConfig` with the fields of the JSON object in ``path`` overriding the defaults.

    Raises:
        TypeError: The file is not a JSON object or a value has the wrong type.
        ValueError: The object names a field :class:`RigConfig` does not have.
    """
    source = pathlib.Path(path)
    data = json.loads(source.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{source}: expected a JSON object of RigConfig fields")
    defaults = dataclasses.asdict(RigConfig())
    unknown = sorted(set(data) - set(defaults))
    if unknown:
        raise ValueError(f"{source}: unknown field(s) {unknown}; allowed: {sorted(defaults)}")
    values = {}
    for name, value in data.items():
        kind = type(defaults[name])
        numeric = kind in (int, float) and isinstance(value, (int, float)) and not isinstance(value, bool)
        if not (numeric and (kind is float or isinstance(value, int)) or kind in (bool, str) and isinstance(value, kind)):
            raise TypeError(f"{source}: {name} must be {kind.__name__}, got {value!r}")
        values[name] = kind(value)
    return RigConfig(**values)


def build_parser() -> argparse.ArgumentParser:
    from ..app import add_server_arguments

    parser = argparse.ArgumentParser(prog="urai-robot", description=(
        "Serve URAI on a dual AgileX PiPER-X rig with Intel RealSense cameras, or on its kinematic simulator."))
    parser.add_argument("--calibration", help="calibration JSON of the rig (required unless --sim; "
                                              "the simulator defaults to the bundled nominal example)")
    parser.add_argument("--rig", help="JSON object overriding RigConfig fields (CAN interfaces, speeds, "
                                      "gripper efforts, camera resolutions, ...)")
    parser.add_argument("--allow-motion", action="store_true",
                        help="open the motion gate: take control of both arms at start-up (holding their pose) "
                             "and let the console move them; without it nothing is commanded")
    parser.add_argument("--sim", action="store_true",
                        help="run on simulated arms and synthetic cameras instead of hardware (motion allowed)")
    add_server_arguments(parser, default_port=7860)
    return parser


def build_runtime(args: argparse.Namespace) -> RobotRuntime:
    """The runtime selected by the parsed command line (hardware or simulator)."""
    rig = load_rig_config(args.rig) if args.rig else RigConfig()
    if args.sim:
        from .simulated import build_simulated_runtime

        calibration = load_calibration(args.calibration or example_calibration_path())
        return build_simulated_runtime(calibration, rig, allow_motion=True)
    from .runtime import build_robot_runtime

    return build_robot_runtime(load_calibration(args.calibration), rig, allow_motion=args.allow_motion)


def main(argv: Sequence[str] | None = None) -> None:
    from ..app import apply_settings_file, is_loopback, serve
    from ..backend import PiperBackend
    from ..service import Service

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.sim and not args.calibration:
        parser.error("--calibration is required on the real robot (or use --sim)")
    if not is_loopback(args.host) and not args.token:
        parser.error(f"refusing to bind {args.host} without a token: set --token or URAI_TOKEN")
    runtime = build_runtime(args)
    backend = None
    try:
        backend = PiperBackend(runtime, workers=os.cpu_count() or 1)
        service = Service(backend, args.log_dir)
        if args.settings:
            apply_settings_file(service, args.settings)
        threading.Thread(target=backend.warm_up, name="planner-warm-up", daemon=True).start()
        serve(service, args.host, args.port, args.token)
    finally:
        if backend is not None:
            backend.close()
        runtime.close()


if __name__ == "__main__":
    main()
