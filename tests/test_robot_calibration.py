# SPDX-License-Identifier: Apache-2.0
"""Calibration file format, validation, the bundled example and model construction."""

import copy
import json

import numpy as np
import pytest

from urai.robot.calibration import (
    Calibration,
    CalibrationError,
    WristCamera,
    example_calibration_path,
    load_calibration,
)
from urai.robot.kinematics import PiperXKinematics, mm_to_m_matrix
from urai.robot.rig import RigConfig, build_models


@pytest.fixture(scope="module")
def example():
    return load_calibration(example_calibration_path())


@pytest.fixture
def document():
    return json.loads(example_calibration_path().read_text())


def is_rigid(t):
    r = t[:3, :3]
    return np.allclose(r.T @ r, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1.0)


class TestExample:
    def test_is_marked_as_a_nominal_example(self, example):
        assert "NOT a measured calibration" in example.notes
        assert example.head_serial.startswith("EXAMPLE")
        assert all(cam.serial.startswith("EXAMPLE") for cam in example.wrist_cameras.values())

    def test_nominal_geometry(self, example):
        np.testing.assert_array_equal(example.t_world_right_base, np.diag([1.0, 1.0, 1.0, 1.0]) + np.array(
            [[0, 0, 0, 0], [0, 0, 0, -0.59], [0, 0, 0, 0], [0, 0, 0, 0]]))
        head = example.t_world_head_camera
        assert is_rigid(head)
        assert head[2, 3] > 0.5 and head[2, 2] < -0.8
        hit = head[:3, 3] - head[:3, 2] * head[2, 3] / head[2, 2]
        assert 0.2 < hit[0] < 0.5 and hit[1] == pytest.approx(-0.295)
        for cam in example.wrist_cameras.values():
            assert is_rigid(cam.t_link6_camera)

    def test_saving_reproduces_the_file(self, example, tmp_path):
        target = example.save(tmp_path / "copy.json")
        assert target.read_text() == example_calibration_path().read_text()
        assert '[0.0, 1.0, 0.0, -0.59],' in target.read_text()


class TestRigAndModels:
    def test_rig_defaults(self):
        rig = RigConfig()
        assert (rig.table_z_mm, rig.link_clearance_mm, rig.fingertip_below_table_mm, rig.ik_position_tol_mm) == (3.5, 60.0, 5.0, 2.0)
        assert (rig.left_interface, rig.right_interface, rig.speed_percent) == ("canleft", "canright", 20)
        assert (rig.grip_effort, rig.open_effort, rig.max_opening_mm) == (1000, 1000, 70.0)
        assert (rig.head_width, rig.head_height, rig.wrist_width, rig.wrist_height) == (1280, 720, 1280, 720)
        assert (rig.depth_width, rig.depth_height, rig.head_stream) == (848, 480, True)
        assert rig.fingertip_bias_for("left") == rig.fingertip_bias_for("right") == 0.0
        with pytest.raises(ValueError, match="unknown arm"):
            rig.fingertip_bias_for("middle")

    def test_build_models_places_the_arms_from_the_calibration(self, example):
        rig = RigConfig(table_z_mm=4.0, link_clearance_mm=50.0, fingertip_bias_mm_left=5.0, fingertip_bias_mm_right=-2.0,
                        fingertip_below_table_mm=3.0, ik_position_tol_mm=1.5)
        models = build_models(example, rig)
        assert set(models) == {"left", "right"} and models["left"].kin is models["right"].kin
        np.testing.assert_array_equal(models["left"].t_world_base, np.eye(4))
        np.testing.assert_array_equal(models["right"].t_world_base, example.t_world_right_base)
        for arm, bias in (("left", 5.0), ("right", -2.0)):
            m = models[arm]
            assert m.name == arm and m.fingertip_bias_mm == bias
            assert (m.table_z_mm, m.link_clearance_mm, m.fingertip_below_table_mm, m.ik_position_tol_mm) == (4.0, 50.0, 3.0, 1.5)
        q = np.array([10.0, 60.0, -60.0, 0.0, 30.0, 0.0])
        np.testing.assert_allclose(models["right"].fk_tcp_world(q)[:3, 3] - models["left"].fk_tcp_world(q)[:3, 3], [0, -0.59, 0], atol=1e-12)

    def test_wrist_camera_chain(self, example):
        kin = PiperXKinematics()
        q = np.array([-20.0, 70.0, -50.0, 10.0, 20.0, 30.0])
        for arm in ("left", "right"):
            expected = example.t_world_base(arm) @ mm_to_m_matrix(kin.fk(q)) @ example.wrist_cameras[arm].t_link6_camera
            np.testing.assert_allclose(example.t_world_wrist_camera(arm, kin.fk(q)), expected, atol=1e-15)
        with pytest.raises(ValueError, match="unknown arm"):
            example.t_world_base("middle")
        with pytest.raises(ValueError, match="4x4"):
            example.t_world_wrist_camera("left", np.eye(3))

    def test_world_base_is_a_copy(self, example):
        t = example.t_world_base("right")
        t[0, 3] = 99.0
        assert example.t_world_right_base[0, 3] == 0.0


class TestRoundTrip:
    def test_to_dict_and_back(self, example, tmp_path):
        rebuilt = Calibration.from_dict(example.to_dict())
        for name in ("t_world_right_base", "t_world_head_camera"):
            np.testing.assert_array_equal(getattr(rebuilt, name), getattr(example, name))
        assert rebuilt.head_serial == example.head_serial and rebuilt.notes == example.notes
        loaded = load_calibration(example.save(tmp_path / "c.json"))
        np.testing.assert_array_equal(loaded.wrist_cameras["right"].t_link6_camera, example.wrist_cameras["right"].t_link6_camera)

    def test_notes_are_optional(self, document):
        del document["notes"]
        calibration = Calibration.from_dict(document)
        assert calibration.notes == "" and "notes" not in calibration.to_dict()

    def test_rounded_rotation_within_tolerance_is_accepted(self, document):
        head = np.round(np.array(document["head_camera"]["T_world_camera_m"]), 6)
        document["head_camera"]["T_world_camera_m"] = head.tolist()
        Calibration.from_dict(document)


def edit(document, path, value):
    doc = copy.deepcopy(document)
    node = doc
    for key in path[:-1]:
        node = node[key]
    if value is DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return doc


DELETE = object()
SCALED = [[1.01, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
MIRRORED = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


@pytest.mark.parametrize(("path", "value", "message"), [
    (("head_camera", "serial"), DELETE, r"head_camera: missing required field\(s\) 'serial'"),
    (("wrist_cameras", "right"), DELETE, r"wrist_cameras: missing required field\(s\) 'right'"),
    (("right_arm_base",), DELETE, r"missing required field\(s\) 'right_arm_base'"),
    (("head_camera", "T_world_camera_mm"), [[1]], r"head_camera: unknown field\(s\) 'T_world_camera_mm'"),
    (("format",), "other-format", r"format must be 'urai-calibration'"),
    (("version",), 2, r"unsupported version 2"),
    (("version",), True, r"unsupported version True"),
    (("notes",), 3, r"notes must be a string"),
    (("right_arm_base", "T_world_base_m"), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], r"right_arm_base.T_world_base_m: expected 4 rows of 4 numbers"),
    (("right_arm_base", "T_world_base_m"), [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]], r"expected 4 rows of 4 numbers"),
    (("right_arm_base", "T_world_base_m"), "identity", r"expected a 4x4 array of numbers"),
    (("right_arm_base", "T_world_base_m"), [[1, 0, 0, "0"], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], r"expected a 4x4 array of numbers"),
    (("right_arm_base", "T_world_base_m"), [[1, 0, 0, float("nan")], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], r"non-finite"),
    (("right_arm_base", "T_world_base_m"), [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]], r"bottom row must be \[0, 0, 0, 1\]"),
    (("head_camera", "T_world_camera_m"), SCALED, r"head_camera.T_world_camera_m: rotation block is not orthonormal"),
    (("wrist_cameras", "left", "T_link6_camera_m"), MIRRORED, r"wrist_cameras.left.T_link6_camera_m: rotation block is a reflection"),
    (("wrist_cameras", "left", "serial"), 123456789012, r"wrist_cameras.left.serial: must be a non-empty string"),
    (("head_camera", "serial"), " ", r"head_camera.serial: must be a non-empty string"),
])
def test_invalid_documents_are_rejected_with_a_located_message(document, path, value, message):
    with pytest.raises(CalibrationError, match=message):
        Calibration.from_dict(edit(document, path, value), source="rig.json")


def test_load_errors(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(CalibrationError, match="broken.json: invalid JSON"):
        load_calibration(broken)
    not_object = tmp_path / "list.json"
    not_object.write_text("[]")
    with pytest.raises(CalibrationError, match="expected a JSON object, got list"):
        load_calibration(not_object)
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "missing.json")


def test_constructor_validates(example):
    with pytest.raises(CalibrationError, match="Calibration.t_world_right_base: rotation block is not orthonormal"):
        Calibration(np.array(SCALED, dtype=float), "s", np.eye(4), dict(example.wrist_cameras))
    with pytest.raises(CalibrationError, match="Calibration.wrist_cameras"):
        Calibration(np.eye(4), "s", np.eye(4), {"left": example.wrist_cameras["left"]})
    with pytest.raises(CalibrationError, match="WristCamera.serial"):
        WristCamera("", np.eye(4))
