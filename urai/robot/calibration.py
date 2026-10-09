# SPDX-License-Identifier: Apache-2.0
"""Calibration of a dual PiPER-X rig: file format, loading and validation.

The file is JSON (format ``urai-calibration``, version 1); docs/calibration.md describes every
field and a calibration procedure. Frames:

* ``world``: the base frame (URDF ``base_link``) of the left arm.
* ``base``: the base frame of an arm; the right arm's base pose in the world is calibrated.
* ``link6``: the URDF link6 frame of an arm, i.e. the frame of the firmware end pose.
* ``camera``: the optical frame of a camera's colour stream (OpenCV: x right, y down, z forward).

All translations in the file are in metres. A transform ``T_a_b`` maps points from frame ``b``
into frame ``a``.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from .kinematics import mm_to_m_matrix

__all__ = [
    "ARMS",
    "EXAMPLE_CALIBRATION_PATH",
    "FORMAT",
    "ROTATION_TOLERANCE",
    "VERSION",
    "Calibration",
    "CalibrationError",
    "WristCamera",
    "example_calibration_path",
    "load_calibration",
]

#: Value of the ``format`` field.
FORMAT = "urai-calibration"
#: Supported value of the ``version`` field.
VERSION = 1
#: Arm names, in the order the models are built.
ARMS = ("left", "right")
#: Largest accepted entry of ``|R^T R - I|`` for a rotation block.
ROTATION_TOLERANCE = 1e-5
#: Nominal example shipped with the package (not a measured calibration).
EXAMPLE_CALIBRATION_PATH = pathlib.Path(__file__).parent / "assets" / "example_calibration.json"


class CalibrationError(ValueError):
    """Calibration data is malformed: a field is missing, unknown or has an invalid value."""


@dataclass(frozen=True)
class WristCamera:
    """A camera mounted on an arm's wrist (eye-in-hand).

    Attributes:
        serial: camera serial number.
        t_link6_camera: camera pose in the arm's link6 frame, 4x4 in metres.
    """

    serial: str
    t_link6_camera: np.ndarray

    def __post_init__(self) -> None:
        _serial(self.serial, "WristCamera.serial")
        object.__setattr__(self, "t_link6_camera", _rigid(self.t_link6_camera, "WristCamera.t_link6_camera"))


@dataclass(frozen=True)
class Calibration:
    """Validated calibration of the rig.

    Attributes:
        t_world_right_base: right arm base pose in the world (left arm base) frame, 4x4, metres.
        head_serial: serial number of the fixed head camera.
        t_world_head_camera: head camera pose in the world frame, 4x4, metres (eye-to-hand).
        wrist_cameras: ``{"left": WristCamera, "right": WristCamera}``.
        notes: free-form text ("" when absent).
    """

    t_world_right_base: np.ndarray
    head_serial: str
    t_world_head_camera: np.ndarray
    wrist_cameras: dict[str, WristCamera]
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "t_world_right_base", _rigid(self.t_world_right_base, "Calibration.t_world_right_base"))
        _serial(self.head_serial, "Calibration.head_serial")
        object.__setattr__(self, "t_world_head_camera", _rigid(self.t_world_head_camera, "Calibration.t_world_head_camera"))
        if not isinstance(self.wrist_cameras, dict) or sorted(self.wrist_cameras) != sorted(ARMS) or not all(
            isinstance(cam, WristCamera) for cam in self.wrist_cameras.values()
        ):
            raise CalibrationError(f"Calibration.wrist_cameras: expected a WristCamera for each of {ARMS}")
        if not isinstance(self.notes, str):
            raise CalibrationError("Calibration.notes: must be a string")

    def t_world_base(self, arm: str) -> np.ndarray:
        """Base pose of ``arm`` in the world frame (identity for the left arm), 4x4, metres."""
        _check_arm(arm)
        return np.eye(4) if arm == "left" else self.t_world_right_base.copy()

    def t_world_wrist_camera(self, arm: str, t_base_link6_mm: np.ndarray) -> np.ndarray:
        """World pose (4x4, m) of ``arm``'s wrist camera when its link6 pose in the base frame is
        ``t_base_link6_mm`` (4x4, translation in mm, e.g. the firmware end pose or ``kin.fk(q)``)."""
        _check_arm(arm)
        t_base_link6 = np.asarray(t_base_link6_mm, dtype=np.float64)
        if t_base_link6.shape != (4, 4):
            raise ValueError(f"t_base_link6_mm must be 4x4, got shape {t_base_link6.shape}")
        return self.t_world_base(arm) @ mm_to_m_matrix(t_base_link6) @ self.wrist_cameras[arm].t_link6_camera

    @classmethod
    def from_dict(cls, data: Any, *, source: str = "calibration") -> Calibration:
        """Validate a decoded calibration document; ``source`` prefixes error messages."""
        _keys(data, source, required=("format", "version", "right_arm_base", "head_camera", "wrist_cameras"),
              optional=("notes",))
        if data["format"] != FORMAT:
            raise CalibrationError(f"{source}: format must be {FORMAT!r}, got {data['format']!r}")
        version = data["version"]
        if isinstance(version, bool) or version != VERSION:
            raise CalibrationError(f"{source}: unsupported version {version!r}; expected {VERSION}")
        notes = data.get("notes", "")
        if not isinstance(notes, str):
            raise CalibrationError(f"{source}: notes must be a string")
        right = data["right_arm_base"]
        _keys(right, f"{source}: right_arm_base", required=("T_world_base_m",))
        head = data["head_camera"]
        _keys(head, f"{source}: head_camera", required=("serial", "T_world_camera_m"))
        wrists = data["wrist_cameras"]
        _keys(wrists, f"{source}: wrist_cameras", required=ARMS)
        wrist_cameras = {}
        for arm in ARMS:
            where = f"{source}: wrist_cameras.{arm}"
            _keys(wrists[arm], where, required=("serial", "T_link6_camera_m"))
            wrist_cameras[arm] = WristCamera(
                serial=_serial(wrists[arm]["serial"], f"{where}.serial"),
                t_link6_camera=_rigid(wrists[arm]["T_link6_camera_m"], f"{where}.T_link6_camera_m"),
            )
        return cls(
            t_world_right_base=_rigid(right["T_world_base_m"], f"{source}: right_arm_base.T_world_base_m"),
            head_serial=_serial(head["serial"], f"{source}: head_camera.serial"),
            t_world_head_camera=_rigid(head["T_world_camera_m"], f"{source}: head_camera.T_world_camera_m"),
            wrist_cameras=wrist_cameras,
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable document in the file format."""
        out: dict[str, Any] = {"format": FORMAT, "version": VERSION}
        if self.notes:
            out["notes"] = self.notes
        out["right_arm_base"] = {"T_world_base_m": self.t_world_right_base.tolist()}
        out["head_camera"] = {"serial": self.head_serial, "T_world_camera_m": self.t_world_head_camera.tolist()}
        out["wrist_cameras"] = {
            arm: {"serial": self.wrist_cameras[arm].serial, "T_link6_camera_m": self.wrist_cameras[arm].t_link6_camera.tolist()}
            for arm in ARMS
        }
        return out

    def save(self, path: pathlib.Path | str) -> pathlib.Path:
        """Write the calibration as JSON (one matrix row per line) and return the path."""
        target = pathlib.Path(path)
        target.write_text(_to_json(self.to_dict()) + "\n")
        return target


def load_calibration(path: pathlib.Path | str) -> Calibration:
    """Load and validate a calibration file.

    Raises:
        FileNotFoundError: the file does not exist.
        CalibrationError: the file is not valid JSON or does not match the format.
    """
    source = pathlib.Path(path)
    try:
        data = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise CalibrationError(f"{source}: invalid JSON: {exc}") from exc
    return Calibration.from_dict(data, source=str(source))


def example_calibration_path() -> pathlib.Path:
    """Path of the bundled nominal example calibration."""
    return EXAMPLE_CALIBRATION_PATH


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def _keys(obj: Any, where: str, *, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    if not isinstance(obj, dict):
        raise CalibrationError(f"{where}: expected a JSON object, got {type(obj).__name__}")
    missing = [k for k in required if k not in obj]
    if missing:
        raise CalibrationError(f"{where}: missing required field(s) {', '.join(map(repr, missing))}")
    unknown = sorted(set(obj) - set(required) - set(optional))
    if unknown:
        allowed = ", ".join(map(repr, required + optional))
        raise CalibrationError(f"{where}: unknown field(s) {', '.join(map(repr, unknown))}; allowed: {allowed}")


def _serial(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationError(f"{where}: must be a non-empty string (quote serial numbers), got {value!r}")
    return value


def _rigid(value: Any, where: str) -> np.ndarray:
    """Validated 4x4 rigid transform from nested lists of numbers or an array."""
    rows = value.tolist() if isinstance(value, np.ndarray) else value
    if not isinstance(rows, list) or not all(isinstance(r, list) for r in rows) or not all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for r in rows for v in r
    ):
        raise CalibrationError(f"{where}: expected a 4x4 array of numbers (a list of 4 rows)")
    if len(rows) != 4 or any(len(r) != 4 for r in rows):
        raise CalibrationError(f"{where}: expected 4 rows of 4 numbers, got row lengths {[len(r) for r in rows]}")
    t = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(t).all():
        raise CalibrationError(f"{where}: contains non-finite values")
    if not np.array_equal(t[3], [0.0, 0.0, 0.0, 1.0]):
        raise CalibrationError(f"{where}: bottom row must be [0, 0, 0, 1], got {t[3].tolist()}")
    rotation = t[:3, :3]
    error = float(np.abs(rotation.T @ rotation - np.eye(3)).max())
    if error > ROTATION_TOLERANCE:
        raise CalibrationError(
            f"{where}: rotation block is not orthonormal (max |R^T R - I| = {error:.2e} > {ROTATION_TOLERANCE:.0e})"
        )
    det = float(np.linalg.det(rotation))
    if det < 0.0:
        raise CalibrationError(f"{where}: rotation block is a reflection (det = {det:.3f})")
    return t


def _to_json(value: Any, indent: int = 0) -> str:
    """JSON text with two-space indentation and one matrix row per line."""
    pad = "  " * (indent + 1)
    if isinstance(value, dict):
        items = [f"{pad}{json.dumps(key)}: {_to_json(item, indent + 1)}" for key, item in value.items()]
        return "{\n" + ",\n".join(items) + "\n" + "  " * indent + "}"
    if isinstance(value, list) and value and all(isinstance(row, list) for row in value):
        return "[\n" + ",\n".join(pad + json.dumps(row) for row in value) + "\n" + "  " * indent + "]"
    return json.dumps(value)
