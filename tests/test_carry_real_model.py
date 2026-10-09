"""The one-stroke carry against the PiPER-X kinematics of the example rig; nothing moves.

The observation is rendered by the simulator's ray caster from the synthetic backend's nadir camera: a block on
its plate and a destination plate placed at chosen distances from an arm base. What runs is the skill's own
candidate search over the real inverse kinematics and, in two tests, the whole planner: IK on every sample,
joint limits, self collision and the motion limits.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from urai.backend import SimulationBackend
from urai.geometry import Frame
from urai.pour import EXACT_DEG, EXACT_MM
from urai.robot.arm_model import joint_limit_reason
from urai.robot.scene import Box, SceneObject, Table, render_rgbd
from urai.settings import motion_limits
from urai.skills import plan
from urai.skills.carry import corridor_top, levelled_heights
from urai.skills.context import SkillContext
from urai.trajectory import compile_arms
from test_carry import ink

TOLERANCE = (30., 60.)
SOURCE_SIZE = (.05, .05)
SOURCE_HEIGHT_M = .05
PLATE_SIZE = (.19, .19)
PLATE_HEIGHT_M = .012
GRASP_HALF_M = .021      # half the drawn grasp line: 42 mm across a 50 mm block, inside the 70 mm jaw
PLATE_RGB = (225, 222, 210)
BLOCK_RGB = (200, 45, 40)


def placed(name, centre_xy, z, size, height, rgb):
    pose = np.eye(4)
    pose[:3, 3] = [centre_xy[0], centre_xy[1], z]
    return SceneObject(name, (Box((size[0], size[1], height), rgb),), pose)


def rendered_table(table_z, objects):
    """The synthetic backend's nadir camera looking at a table carrying ``objects``, rendered by ray casting."""
    camera = SimulationBackend().capture()
    height, width = camera.depth.shape
    table = Table(z=table_z, x_range=(-.5, 1.5), y_range=(-1.5, 1.))
    rgb, depth = render_rgbd(camera.k, camera.t, width, height, table, objects)
    return Frame(rgb, depth.astype(float), camera.k, camera.t)


def carry_scene(planner, start, arm, source_reach, landing_reach, sideways_m=.20):
    """A context on the arm model plus the drawn stroke: grasp line across the block, route to the plate."""
    model = planner.models[arm]
    base = np.asarray(model.base_xy, dtype=float)
    table_z = model.table_z_mm/1000
    source_xy, landing_xy = base+[source_reach, 0.], base+[landing_reach, sideways_m]
    frame = rendered_table(table_z, [
        placed('source plate', source_xy, table_z, PLATE_SIZE, PLATE_HEIGHT_M, PLATE_RGB),
        placed('landing plate', landing_xy, table_z, PLATE_SIZE, PLATE_HEIGHT_M, PLATE_RGB),
        placed('block', source_xy, table_z+PLATE_HEIGHT_M, SOURCE_SIZE, SOURCE_HEIGHT_M-PLATE_HEIGHT_M, BLOCK_RGB)])
    top = table_z+SOURCE_HEIGHT_M
    corners = frame.project([[source_xy[0], source_xy[1]-GRASP_HALF_M, top],
                             [source_xy[0], source_xy[1]+GRASP_HALF_M, top],
                             [(source_xy[0]+landing_xy[0])/2, (source_xy[1]+landing_xy[1])/2, table_z],
                             [landing_xy[0], landing_xy[1], table_z+PLATE_HEIGHT_M]])
    context = SkillContext(frame, arm, planner, None, model, current=start)
    assert context.has_kinematics and context.pose_tolerance == TOLERANCE
    return context, ink(corners), table_z


def carried(planner, start, arm='left', source_reach=.36, landing_reach=.34, **numbers):
    context, stroke, table_z = carry_scene(planner, start, arm, source_reach, landing_reach)
    return context, plan('carry_line', context, {'pixels': stroke.tolist(), **numbers}), table_z


def test_the_left_arm_solves_every_pose_of_a_one_stroke_carry(planner, working_start):
    context, result, table_z = carried(planner, working_start, 'left')
    assert result['reach_error_mm'] <= EXACT_MM and result['reach_error_deg'] <= EXACT_DEG
    assert result['grasp_width_mm'] == pytest.approx(42., abs=4.)
    # The block is 50 mm tall on a 12 mm plate, so the carry has to clear 50 mm plus how far the block hangs.
    assert result['obstacle_top_mm'] == pytest.approx(SOURCE_HEIGHT_M*1000, abs=3.)
    assert result['carry_height_mm'] >= result['obstacle_top_mm']+result['object_hang_mm']
    assert result['release_height_mm'] == pytest.approx(PLATE_HEIGHT_M*1000+result['object_hang_mm']+10, abs=3.)


def test_the_right_arm_carries_across_its_own_side_of_the_table(planner, working_start):
    context, result, table_z = carried(planner, working_start, 'right', source_reach=.34, landing_reach=.32)
    assert result['reach_error_mm'] <= EXACT_MM and result['reach_error_deg'] <= EXACT_DEG
    assert result['route_points'] >= 2


def test_every_waypoint_of_the_carry_solves_as_a_chain_from_the_start(planner, working_start):
    context, result, table_z = carried(planner, working_start, 'left')
    points = np.asarray(result['arm']['path']['points'], dtype=float)
    keyframes = np.asarray(result['arm']['orientation']['points'], dtype=float)
    tolerance = {'position_mm': TOLERANCE[0], 'orientation_deg': TOLERANCE[1]}
    seed = np.asarray(working_start['left']['joints_deg'], dtype=float)
    for index, point in enumerate(points):
        progress = index/(len(points)-1)
        pose = keyframes[np.searchsorted(keyframes[:, 0], progress, side='right')-1, 1:]
        rotation = Rotation.from_euler('xyz', pose, degrees=True).as_matrix()
        position_error, rotation_error, seed = planner.pose_error('left', point, rotation, seed, tolerance)
        assert position_error <= EXACT_MM and rotation_error <= EXACT_DEG, f'waypoint {index}'
        assert joint_limit_reason(seed) is None


@pytest.mark.parametrize('approach_speed_m_s', [.02, .08, .15])
def test_the_planner_accepts_the_whole_carry_at_every_approach_speed(approach_speed_m_s, planner, working_start):
    """The full planning pipeline, at approach speeds from 0.02 to 0.15 m/s."""
    context, result, table_z = carried(planner, working_start, 'left', approach_speed_m_s=approach_speed_m_s)
    planned = planner.plan(compile_arms({'left': result['arm']}, working_start), working_start,
                           pose_tolerance={'position_mm': TOLERANCE[0], 'orientation_deg': TOLERANCE[1]})
    limits = motion_limits('normal')
    arm = planned['arms']['left']
    times = np.linspace(arm['time_s'][0], arm['time_s'][-1], 4000)
    assert np.abs(arm['curve'](times, 1)).max() <= limits['joint_speed_deg_s']*1.001
    assert np.abs(arm['curve'](times, 2)).max() <= limits['joint_acceleration_deg_s2']*1.001
    assert all(joint_limit_reason(q) is None for q in arm['joints_deg'])
    assert 0 < planned['duration_s'] <= 180


def test_a_faster_approach_shortens_the_run_without_breaking_the_limits(planner, working_start):
    durations = []
    for speed in (.02, .15):
        context, result, table_z = carried(planner, working_start, 'left', approach_speed_m_s=speed)
        planned = planner.plan(compile_arms({'left': result['arm']}, working_start), working_start,
                               pose_tolerance={'position_mm': TOLERANCE[0], 'orientation_deg': TOLERANCE[1]})
        durations.append(planned['arms']['left']['task_offset_s'])
    assert durations[1] < durations[0]*.5


def test_the_plate_the_segmentation_cloud_hides_is_still_a_rim_the_carry_has_to_clear(planner, working_start):
    """transfer.table_background is a 20th percentile over a 200 mm window, so a 190 mm plate reads as table in
    the segmentation cloud; the carry reads the plane-levelled cloud, where the plate keeps its height."""
    context, result, table_z = carried(planner, working_start, 'left')
    route = np.vstack([np.asarray(result['source_xyz'], dtype=float)[:2],
                       np.asarray(result['landing_xyz'], dtype=float)[:2]])
    segmentation = corridor_top(context.cloud, table_z, route)
    absolute = corridor_top(levelled_heights(context.frame, table_z), table_z, route)
    assert absolute-table_z == pytest.approx(SOURCE_HEIGHT_M, abs=.003)
    assert absolute-segmentation >= .004
    assert result['obstacle_top_mm'] == pytest.approx((absolute-table_z)*1000, abs=.5)
