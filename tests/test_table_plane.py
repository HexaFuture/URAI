"""Object segmentation and heights follow the depth-fitted table plane, anchored to the calibrated table."""
import numpy as np
from urai.transfer import PLACE_RELEASE_HEIGHT_M
import pytest
from urai.backend import SimulationBackend
from urai.transfer import (leveled_cloud, level_point, locate_object, objects_in_polygon, pick_place_from_stroke,
                           table_plane)

TABLE_Z = .0035


def tilted_frame(low_block_height=.012, tilt=.02):
    """Synthetic scene whose depth table rises ``tilt`` metres from the left to the right edge.

    A low block sits on the depth-low side; on the depth-high side the bare table reads well
    above the calibrated height, which used to turn it into phantom objects. The simulation's
    default 60 mm object stays as a wide (150 mm) reference block.
    """
    frame = SimulationBackend().capture()
    height, width = frame.depth.shape
    columns = np.arange(width)/(width-1)
    frame.depth[:] = 1.0-np.tile(tilt*columns-.006, (height, 1))   # world z = -6 mm .. +14 mm
    frame.depth[200:236, 150:186] -= low_block_height                # block on the low side
    frame.rgb[200:236, 150:186] = [200, 60, 60]                      # coloured, like real blocks
    frame.depth[180:260, 350:440] -= .06                             # the default 60 mm object
    return frame


def test_plane_fit_recovers_the_tilt_and_levels_the_table():
    frame = tilted_frame()
    cloud = frame.world_points()
    plane = table_plane(cloud, TABLE_Z)
    leveled, _ = leveled_cloud(frame, TABLE_Z, plane)
    bare = leveled[300:400, 600:700, 2]
    assert np.abs(bare-TABLE_Z).max() < .0015
    low_side = leveled[400:440, 60:120, 2]
    assert np.abs(low_side-TABLE_Z).max() < .0015


def test_low_block_on_the_depth_low_side_is_found_and_bare_high_side_is_not():
    frame = tilted_frame(low_block_height=.012)
    found = objects_in_polygon(frame, [[100, 150], [700, 150], [700, 300], [100, 300]], table_z=TABLE_Z)
    heights = sorted(round(item['height_mm']) for item in found)
    assert heights == [12, 60], found
    low, wide = sorted(found, key=lambda item: item['height_mm'])
    assert low['graspable'] and low['reason'] is None
    assert not wide['graspable'] and '70 mm' in wide['reason']   # 90 px at 1 m is 150 mm wide
    assert abs(low['top_z']-(TABLE_Z+.012)) < .0015


def test_locate_object_reports_height_above_the_calibrated_table():
    frame = tilted_frame(low_block_height=.010)
    found = locate_object(frame, (168, 218), TABLE_Z)
    assert found['top'] == pytest.approx(TABLE_Z+.010, abs=.0015)
    assert found['surface'][2] == pytest.approx(TABLE_Z+.010, abs=.0015)
    with pytest.raises(ValueError, match='没有可分离的物体'):
        locate_object(frame, (650, 220), TABLE_Z)


def test_transfer_grasp_and_landing_heights_are_anchored_to_the_calibrated_table():
    frame = tilted_frame(low_block_height=.012)
    result = pick_place_from_stroke(frame, [[168, 218], [650, 420]], table_z=TABLE_Z)
    points = np.array(result['arm']['path']['points'])
    grasp_z, place_z = points[1, 2], points[-2, 2]
    assert grasp_z == pytest.approx(TABLE_Z+.008, abs=.0015)
    assert place_z == pytest.approx(grasp_z+PLACE_RELEASE_HEIGHT_M+.005, abs=.0015)   # placements let go 3 cm above the support
    assert result['object_height_mm'] == pytest.approx(12, abs=1.5)


def test_leveling_a_picked_point_uses_the_fitted_plane():
    frame = tilted_frame()
    plane = table_plane(frame.world_points(), TABLE_Z)
    picked = frame.pick(650, 420, mode='surface')
    assert picked[2] > TABLE_Z+.006
    assert level_point(plane, TABLE_Z, picked)[2] == pytest.approx(TABLE_Z, abs=.0015)


def test_plane_fit_needs_visible_table():
    frame = SimulationBackend().capture()
    frame.depth[:] = .5
    with pytest.raises(ValueError, match='桌面'):
        table_plane(frame.world_points(), TABLE_Z)


def curved_frame(low_block_height=.012):
    """Tilted table plus a 6 mm bowl-shaped curvature that a plane fit cannot remove."""
    frame = tilted_frame(low_block_height=low_block_height)
    height, width = frame.depth.shape
    v, u = np.mgrid[0:height, 0:width]
    curvature = .006*(((u-width/2)/(width/2))**2+((v-height/2)/(height/2))**2)
    frame.depth -= curvature
    return frame


def test_background_filter_removes_residual_curvature():
    frame = curved_frame()
    leveled, _ = leveled_cloud(frame, TABLE_Z)
    corners = np.r_[leveled[20:60, 20:60, 2].ravel(), leveled[20:60, -60:-20, 2].ravel(), leveled[-60:-20, 20:60, 2].ravel()]
    assert np.abs(corners-TABLE_Z).max() < .0025
    found = objects_in_polygon(frame, [[20, 20], [780, 20], [780, 460], [20, 460]], table_z=TABLE_Z)
    heights = sorted(round(item['height_mm']) for item in found)
    assert heights == [12, 60], found


def test_small_noise_blobs_are_not_objects():
    frame = tilted_frame(low_block_height=.012)
    rng = np.random.default_rng(3)
    for _ in range(12):
        v, u = rng.integers(300, 440), rng.integers(500, 760)
        frame.depth[v:v+6, u:u+6] -= .009   # 36-pixel spikes, 9 mm high
    found = objects_in_polygon(frame, [[100, 150], [780, 150], [780, 470], [100, 470]], table_z=TABLE_Z)
    assert sorted(round(item['height_mm']) for item in found) == [12, 60]


def test_colour_finds_low_coloured_blocks_but_grey_and_black_need_clear_depth():
    frame = tilted_frame(low_block_height=.004)          # coloured block only 4 mm tall
    frame.depth[300:336, 600:636] -= .008                 # grey table-coloured lump, 8 mm: too low
    frame.depth[380:416, 600:636] -= .030                 # black 30 mm region: an arm, never a candidate
    frame.rgb[380:416, 600:636] = [20, 20, 20]
    frame.depth[300:336, 700:736] -= .020                 # grey 20 mm object: within bright-light depth noise, not found
    frame.rgb[300:336, 700:736] = [200, 200, 200]
    frame.depth[420:456, 700:736] -= .040                 # grey 40 mm object: found by depth alone
    frame.rgb[420:456, 700:736] = [200, 200, 200]
    found = objects_in_polygon(frame, [[100, 150], [780, 150], [780, 470], [100, 470]], table_z=TABLE_Z)
    heights = sorted(round(item['height_mm']) for item in found)
    assert heights == [4, 40, 60], found


def test_locate_object_snaps_to_the_nearest_candidate_pixel():
    frame = tilted_frame(low_block_height=.010)
    frame.rgb[214:222, 164:172] = [235, 235, 235]      # a white letter in the middle of the coloured block
    frame.depth[214:222, 164:172] = np.nan              # and a depth hole on it
    frame.rgb[226:230, 176:180] = [60, 200, 60]       # a coloured sliver next to the letter
    frame.depth[226:230, 176:180] -= .010
    frame.rgb[222:226, 172:176] = [235, 235, 235]      # separated from the block body
    found = locate_object(frame, (176, 224), TABLE_Z)  # closest region is the sliver, largest is the block
    assert len(found['points']) > 900
    assert found['top'] == pytest.approx(TABLE_Z+.010, abs=.0015)
    with pytest.raises(ValueError, match='没有可分离的物体'):
        locate_object(frame, (168, 218+36), TABLE_Z)   # 18 px below the block: outside the snap radius


def test_coloured_block_without_depth_is_placed_on_the_table():
    frame = tilted_frame(low_block_height=.012)
    frame.depth[200:236, 150:186] = np.nan                    # the projector shadow swallows the block's depth
    found = objects_in_polygon(frame, [[100, 150], [700, 150], [700, 300], [100, 300]], table_z=TABLE_Z)
    heights = sorted(round(item['height_mm']) for item in found)
    assert heights == [20, 60], found                         # filled at the default 20 mm
    filled = min(found, key=lambda item: item['height_mm'])
    assert filled['depth_filled'] == pytest.approx(1.)
    with_depth = objects_in_polygon(tilted_frame(low_block_height=.012), [[100, 150], [700, 150], [700, 300], [100, 300]], table_z=TABLE_Z)
    reference = min(with_depth, key=lambda item: item['height_mm'])
    assert np.allclose(filled['center_xy'], reference['center_xy'], atol=.008)   # the 20 mm default vs the 12 mm block shifts the ray hit
    assert abs(filled['width_mm']-reference['width_mm']) < 3
    located = locate_object(frame, (168, 218), TABLE_Z)
    assert located['depth_filled'] == pytest.approx(1.) and located['top'] == pytest.approx(TABLE_Z+.02, abs=1e-6)


def test_partial_depth_holes_are_filled_at_the_objects_own_height():
    """Near the bases the projector shadow can cover about half a block; filling that half at the
    default 20 mm pulled a 12 mm block's centre toward the camera and a 28 mm block's away from it by 2-3 cm, and
    the jaws missed both. With enough measured pixels the region's own top height fills the rest."""
    polygon = [[100, 150], [700, 150], [700, 300], [100, 300]]
    reference = min(objects_in_polygon(tilted_frame(low_block_height=.012), polygon, table_z=TABLE_Z), key=lambda item: item['height_mm'])
    frame = tilted_frame(low_block_height=.012)
    frame.depth[200:236, 150:168] = np.nan                    # half of the block has no depth
    found = min(objects_in_polygon(frame, polygon, table_z=TABLE_Z), key=lambda item: item['height_mm'])
    assert .4 <= found['depth_filled'] <= .6
    assert round(found['height_mm']) == 12
    assert np.allclose(found['center_xy'], reference['center_xy'], atol=.002)
    assert abs(found['width_mm']-reference['width_mm']) < 2
    located = locate_object(frame, (168, 218), TABLE_Z)
    assert located['top'] == pytest.approx(TABLE_Z+.012, abs=.0015)
