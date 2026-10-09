"""Geometry of the one-stroke carry on synthetic scenes: where it grasps, how high it flies, when it lets go.

Every number below comes out of the same segmentation, the same grasp-line planner and the same levelled cloud
the console uses.
"""
import numpy as np
import pytest
from test_transfer import scene
from urai.backend import SimulationBackend
from urai.service import Service
from urai.skills import plan, skill_context
from urai.skills.carry import MAX_CARRY_Z_M, corridor_top, levelled_heights
from urai.trajectory import compile_arm

# The simulation camera is nadir at 1 m: x = 1.667 mm per row, y = 1.667 mm per column from column 520.
PLATE = (196, 256, 364, 428, .995)      # 5 mm, the source plate the bread sits on
SOURCE = (210, 242, 380, 412, .94)      # 60 mm tall, 53 mm across, centred near (0.377, -0.207)
LANDING = (280, 340, 530, 594, .985)    # 15 mm, the destination plate around (0.517, 0.067)
ON_ROUTE = (246, 278, 462, 494, .88)    # 120 mm, straddling the straight route
OFF_ROUTE = (380, 412, 300, 332, .88)   # the same block parked at (0.66, -0.34), far from any route
TOO_TALL = (246, 278, 462, 494, .80)    # 200 mm: the arms and the camera bracket, not something to fly over
TALL_SOURCE = (210, 242, 380, 412, .88)  # the same footprint 120 mm tall: the object then hangs 110 mm down
TABLE_SCENE = (PLATE, SOURCE)
PLATE_SCENE = (PLATE, SOURCE, LANDING)


def ink(corners, step=6.):
    """A polyline sampled every ``step`` pixels, the way a pointer stroke actually arrives."""
    points = [np.asarray(corners[0], dtype=float)]
    for start, end in zip(corners[:-1], corners[1:]):
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        count = max(1, int(np.ceil(np.linalg.norm(end-start)/step)))
        points.extend(start+(end-start)*np.arange(1, count+1)[:, None]/count)
    return np.asarray(points)


GRASP_LINE = [[382, 226], [410, 226]]           # across the block, about 42 mm at its top
STRAIGHT = ink([*GRASP_LINE, [560, 310]])
CURVED = ink([*GRASP_LINE, [452, 150], [570, 220], [560, 310]])


def context(blocks=TABLE_SCENE, arm='left', **numbers):
    service = Service(SimulationBackend())
    service.observe()
    service.frame = scene(list(blocks))
    return skill_context(service, arm, **numbers)


def run(pixels=STRAIGHT, blocks=TABLE_SCENE, ctx=None, **numbers):
    ctx = ctx or context(blocks)
    return ctx, plan('carry_line', ctx, {'pixels': np.asarray(pixels).tolist(), **numbers})


def path(result):
    return np.asarray(result['arm']['path']['points'], dtype=float)


def test_one_stroke_grasps_lifts_flies_and_releases_in_a_single_draft():
    ctx, result = run()
    points = path(result)
    carry = result['carry_height_mm']/1000+ctx.table_z
    assert points[0, 2] > points[1, 2]                                  # hover down onto the grasp
    np.testing.assert_allclose(points[1, :2], points[2, :2])            # lift straight up from the grasp
    assert points[2, 2] == pytest.approx(carry)
    np.testing.assert_allclose(points[3:-2, 2], carry)                  # one flight height all the way across
    assert points[-2, 2] == pytest.approx(result['release_height_mm']/1000+ctx.table_z)
    assert points[-2, 2] < carry
    np.testing.assert_allclose(points[-1], points[-3])                  # retract back up over the drop point
    events = result['arm']['gripper_events']
    assert [event['opening_mm'] for event in events] == [70, 0, 70]
    # Only the grip is verified: the jaws report 62-69 mm when open, so a gate on the opening aborts runs.
    assert [event.get('verify') for event in events] == [None, 'holding', None]
    assert all(event['wait_for_arrival'] and event['hold_s'] >= .6 for event in events)
    steps = len(points)-1
    assert [event['s'] for event in events] == [0., 1/steps, (steps-1)/steps]
    # The jaws close on the grasp waypoint and open on the release waypoint, both at a full stop.
    plan_arm = compile_arm(result['arm'], SimulationBackend().state()['left'])
    for event, point in zip(plan_arm['gripper_events'], [points[1], points[-2]][:2]):
        index = np.where(plan_arm['time_s'] == event['time_s'])[0][0]
        assert np.linalg.norm(plan_arm['xyz'][index+1]-plan_arm['xyz'][index]) < 1e-12
    np.testing.assert_allclose(plan_arm['xyz'][np.where(
        plan_arm['time_s'] == plan_arm['gripper_events'][1]['time_s'])[0][0]], points[1])


def test_the_leading_ink_is_the_grasp_line_the_grasp_line_skill_would_plan():
    ctx, result = run()
    alone = plan('grasp_line', context(), {'pixels': GRASP_LINE, 'inset_mm': 10, 'endpoint_action': 'grasp'})
    np.testing.assert_allclose(result['source_xyz'], np.asarray(alone['arm']['path']['points'])[1])
    assert result['grasp_width_mm'] == pytest.approx(alone['width_mm'])
    assert result['actual_inset_mm'] == pytest.approx(alone['actual_inset_mm'])
    # The closing axis, and therefore the whole grasp pose, is the grasp-line planner's, not a second guess.
    np.testing.assert_allclose(result['arm']['orientation']['points'][0],
                               alone['arm']['orientation']['points'][0])
    assert result['grasp_line_mm'] == pytest.approx(result['grasp_width_mm'], abs=1.)


def test_a_longer_grasp_line_moves_the_grasp_point_along_the_stroke():
    _, short = run(grasp_span_mm=20)
    _, long_line = run(grasp_span_mm=40)
    assert short['grasp_width_mm'] < long_line['grasp_width_mm']
    assert np.linalg.norm(np.asarray(short['source_xyz'])[:2]-np.asarray(long_line['source_xyz'])[:2]) > .005


def test_the_carry_height_clears_the_tallest_thing_under_the_route_plus_the_object_hang():
    """What sets the height here is the obstacle, so the lift is held at 8 cm.

    At the 12 cm default the hover after the grasp is already higher than an empty route needs, and the
    clear carry rides at that instead - true, and not what this test is about.
    """
    low = dict(clearance_m=.08)
    _, clear = run(ctx=context(TABLE_SCENE, **low))
    _, blocked = run(ctx=context((*TABLE_SCENE, ON_ROUTE), **low))
    _, beside = run(ctx=context((*TABLE_SCENE, OFF_ROUTE), **low))
    hang = clear['object_hang_mm']
    assert clear['obstacle_top_mm'] == pytest.approx(60, abs=2)          # the source block is its own obstacle
    assert blocked['obstacle_top_mm'] == pytest.approx(120, abs=2)
    assert beside['obstacle_top_mm'] == pytest.approx(clear['obstacle_top_mm'], abs=2)
    assert blocked['carry_height_mm']-clear['carry_height_mm'] == pytest.approx(60, abs=2)
    assert beside['carry_height_mm'] == pytest.approx(clear['carry_height_mm'], abs=2)
    # What must clear the obstacle is the bottom of the carried object, not the fingertips.
    assert blocked['carry_height_mm']-hang >= blocked['obstacle_top_mm']
    assert clear['carry_height_mm'] == pytest.approx(clear['obstacle_top_mm']+hang+50, abs=2)


def test_a_drawn_detour_lowers_the_carry_because_the_corridor_follows_the_ink():
    """Also held at 8 cm: a detour can only lower the carry while the obstacle is what raised it."""
    _, straight = run(STRAIGHT, ctx=context((*TABLE_SCENE, ON_ROUTE), clearance_m=.08))
    _, around = run(CURVED, ctx=context((*TABLE_SCENE, ON_ROUTE), clearance_m=.08))
    assert straight['obstacle_top_mm'] == pytest.approx(120, abs=2)
    assert around['obstacle_top_mm'] < 70
    assert around['carry_height_mm'] < straight['carry_height_mm']-40


def test_the_flight_follows_the_drawn_curve_instead_of_the_straight_line():
    ctx, result = run(CURVED)
    points = path(result)[3:-3, :2]
    start, end = path(result)[2, :2], path(result)[-3, :2]
    direction = (end-start)/np.linalg.norm(end-start)
    offsets = np.abs(direction[0]*(points-start)[:, 1]-direction[1]*(points-start)[:, 0])
    assert offsets.max() > .05          # the bow the operator drew survives into the waypoints
    drawn = np.array([ctx.frame.pick(u, v, mode='plane', z=ctx.table_z)[:2] for u, v in CURVED])
    for point in points:                # and every waypoint sits on the drawn ink
        assert np.linalg.norm(drawn-point, axis=1).min() < .02


def test_something_taller_than_a_tabletop_object_is_the_room_and_does_not_lift_the_carry():
    _, clear = run(blocks=TABLE_SCENE)
    _, tall = run(blocks=(*TABLE_SCENE, TOO_TALL))
    assert tall['carry_height_mm'] == pytest.approx(clear['carry_height_mm'], abs=2)


def test_the_release_height_follows_the_landing_surface_and_the_object_thickness():
    _, onto_table = run(blocks=TABLE_SCENE)
    _, onto_plate = run(blocks=PLATE_SCENE)
    hang = onto_table['object_hang_mm']
    assert onto_table['release_height_mm'] == pytest.approx(hang+10, abs=2)
    assert onto_plate['release_height_mm']-onto_table['release_height_mm'] == pytest.approx(15, abs=2)
    _, wide_gap = run(blocks=PLATE_SCENE, release_gap_mm=30)
    assert wide_gap['release_height_mm']-onto_plate['release_height_mm'] == pytest.approx(20, abs=1)
    # The object's own bottom, not the fingertips, is what ends up release_gap above the plate.
    assert onto_plate['release_height_mm']-hang == pytest.approx(15+10, abs=2)


def test_the_carry_uses_the_operators_approach_speed_and_the_contexts_clearance():
    ctx, result = run(approach_speed_m_s=.12)
    assert result['arm']['approach'] == {'speed_m_s': .12, 'clearance_m': ctx.clearance_m}
    assert result['arm']['speed'] == ctx.speed_m_s


def test_it_refuses_a_two_point_stroke_a_short_stroke_and_a_landing_on_the_object():
    with pytest.raises(ValueError, match='连续画'):
        run([[382, 226], [410, 226], [430, 226]][:2])
    with pytest.raises(ValueError, match='这一笔太短'):
        run(ink([[382, 226], [410, 226], [424, 240]]))
    with pytest.raises(ValueError, match='落点离抓取点'):
        run(ink([[382, 226], [410, 226], [470, 290], [396, 234]]))


def test_it_clamps_a_wide_grasp_line_but_refuses_a_carry_that_would_fly_too_high():
    _,result=run(ink([[382, 226], [460, 226], [560, 310]]), grasp_span_mm=70)
    assert result['grasp_width_mm']==pytest.approx(70.)
    with pytest.raises(ValueError, match=f'超过 {MAX_CARRY_Z_M*1000:.0f} mm'):
        # A 120 mm object hanging 110 mm below the fingertips, over a 120 mm obstacle, with a 200 mm margin.
        run(blocks=(PLATE, TALL_SOURCE, ON_ROUTE), carry_margin_m=.20)


def test_corridor_top_needs_a_real_surface_not_a_single_depth_spike():
    ctx = context((*TABLE_SCENE, ON_ROUTE))
    heights = levelled_heights(ctx.frame, ctx.table_z)
    route = np.array([[.377, -.207], [.517, .067]])
    assert corridor_top(heights, ctx.table_z, route) == pytest.approx(.12, abs=.002)
    # Bare table off to the side of everything reads as table, not as an obstacle.
    empty = np.array([[.10, -.42], [.30, -.46]])
    assert corridor_top(heights, ctx.table_z, empty) == pytest.approx(ctx.table_z)
    # Nine pixels of bad depth are noise, not a 100 mm obstacle; only the 60 mm source block survives.
    spike = context(TABLE_SCENE)
    spike.frame.depth[299:302, 469:472] = .90
    assert corridor_top(levelled_heights(spike.frame, spike.table_z), spike.table_z, route) \
        == pytest.approx(.06, abs=.002)
