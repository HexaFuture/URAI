"""The planner's fingertip floor agrees with the arm model's own table check."""
import dataclasses

import numpy as np
import pytest

from urai.backend.model_checks import floor_world_mm


@pytest.mark.parametrize('arm,bias_mm', [('left', 0.), ('left', 5.), ('right', -3.)])
def test_planner_fingertip_height_matches_the_table_check(arm_models, arm, bias_mm):
    model = dataclasses.replace(arm_models[arm], fingertip_bias_mm=bias_mm)
    # The bases of the example rig stand on the world plane z = 0.
    assert floor_world_mm(model) == pytest.approx(model.table_z_mm-model.fingertip_below_table_mm)
    lower = np.array([model.kin.limits_deg[i][0] for i in range(1, 7)])
    upper = np.array([model.kin.limits_deg[i][1] for i in range(1, 7)])
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(lower, upper)
        planner_tip = model.fk_tcp_world(q)[2, 3]*1000.+model.fingertip_bias_mm
        table_tip = model.kin.link_origins(q)[-1, 2]+model.t_world_base[2, 3]*1000.+model.fingertip_bias_mm
        assert planner_tip == pytest.approx(table_tip, abs=1e-6)


def test_raised_base_raises_the_world_floor_by_the_same_height(arm_models):
    t = arm_models['left'].t_world_base.copy(); t[2, 3] = .1
    raised = dataclasses.replace(arm_models['left'], t_world_base=t)
    assert floor_world_mm(raised) == pytest.approx(floor_world_mm(arm_models['left'])+100.)


def test_a_tilted_base_has_no_single_floor_height(arm_models):
    angle = np.deg2rad(5.)
    t = np.eye(4); t[:3, :3] = [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
    with pytest.raises(ValueError, match='not upright'):
        floor_world_mm(dataclasses.replace(arm_models['left'], t_world_base=t))
