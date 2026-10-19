"""Geometry invariants for observed-rim pouring; no hardware access."""
import copy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from urai.pour_alignment import fit_rim, lowest_lip, plan_step


def rim(center, normal=(0., 0., 1.), radius=.035, residual=0.):
    return {'center': np.asarray(center, float), 'normal': np.asarray(normal, float),
            'radius': radius, 'residual_m': residual}


def circle(center, normal, radius, angles=None):
    normal = np.asarray(normal, float)
    normal /= np.linalg.norm(normal)
    ref = np.eye(3)[np.argmin(abs(normal))]
    x = np.cross(normal, ref)
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False) if angles is None else np.asarray(angles)
    return np.asarray(center)+radius*(np.cos(angles)[:, None]*x+np.sin(angles)[:, None]*y)


@pytest.mark.parametrize('span', [120., 180., 300.])
def test_fit_recovers_tilted_rim_from_wide_arc_without_mutation(span):
    center = np.array([.42, -.21, .37])
    normal = Rotation.from_euler('xyz', [28, -37, 52], degrees=True).apply([0, 0, 1])
    points = circle(center, normal, .028, np.deg2rad(np.linspace(15, 15+span, 12)))
    before = points.copy()
    fitted = fit_rim(points, normal)
    np.testing.assert_allclose(fitted['center'], center, atol=1e-9)
    np.testing.assert_allclose(fitted['normal'], normal, atol=1e-9)
    assert fitted['radius'] == pytest.approx(.028, abs=1e-9)
    assert fitted['residual_m'] < 1e-9
    np.testing.assert_array_equal(points, before)


def test_fit_tracks_normal_sign_past_ninety_degrees():
    normal = np.array([np.sin(np.deg2rad(115)), 0., np.cos(np.deg2rad(115))])
    points = circle([.4, -.2, .3], normal, .018)
    fitted = fit_rim(points, normal_hint=normal)
    assert fitted['normal'][2] < 0
    np.testing.assert_allclose(fitted['normal'], normal, atol=1e-9)


@pytest.mark.parametrize('points', [
    np.zeros((3, 3)), np.zeros((17, 3)), np.zeros((4, 2)),
    np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])*.01,
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0]])*.01,
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])*.01,
])
def test_fit_rejects_invalid_or_degenerate_landmarks(points):
    with pytest.raises(ValueError):
        fit_rim(points)


def test_fit_rejects_narrow_arc_even_when_circle_residual_is_zero():
    points = circle([.4, .2, .2], [0, 0, 1], .04, np.deg2rad(np.linspace(0, 110, 16)))
    with pytest.raises(ValueError, match='120'):
        fit_rim(points)


@pytest.mark.parametrize('radius', [.007, .101])
def test_fit_rejects_out_of_range_radius(radius):
    with pytest.raises(ValueError, match='radius'):
        fit_rim(circle([.4, .2, .2], [0, 0, 1], radius))


def test_fit_rejects_plane_and_radial_outliers():
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    points = circle([.4, .2, .2], [0, 0, 1], .04, angles)
    points[:, 2] += np.tile([-.005, .005], 8)
    with pytest.raises(ValueError, match='RMS'):
        fit_rim(points)
    radii = .04+np.tile([-.006, .006], 8)
    points = np.c_[.4+radii*np.cos(angles), .2+radii*np.sin(angles), np.full(16, .2)]
    with pytest.raises(ValueError, match='RMS'):
        fit_rim(points)


@pytest.mark.parametrize('tilt', [10., 80., 90., 110., 150.])
def test_lowest_lip_has_correct_sign_before_and_after_ninety(tilt):
    angle = np.deg2rad(tilt)
    center = np.array([.3, -.2, .4])
    normal = np.array([np.sin(angle), 0., np.cos(angle)])
    expected = center+.023*np.array([np.cos(angle), 0., -np.sin(angle)])
    np.testing.assert_allclose(lowest_lip(center, normal, .023), expected, atol=1e-12)


@pytest.mark.parametrize('normal', [[0, 0, 1], [0, 0, -1]])
def test_horizontal_rim_lowest_lip_convention_is_center(normal):
    np.testing.assert_array_equal(lowest_lip([.3, -.2, .4], normal, .02), [.3, -.2, .4])


TARGET = rim([.6, -.25, .12], radius=.04)
DIRECTION = np.array([.6, .8])


def aligned_source(tilt, radius=.014, length=.35, body_radius=.06, direction=DIRECTION):
    angle = np.deg2rad(tilt)
    normal = np.r_[direction*np.sin(angle), np.cos(angle)]
    xy = TARGET['center'][:2]-(radius*np.cos(angle)*direction if tilt else 0.)
    z = TARGET['center'][2]+max(.04+radius*np.sin(angle),
                                    .02+length*max(np.cos(angle), 0.)+body_radius*np.sin(angle))
    return rim(np.r_[xy, z], normal, radius)


def assert_same_rim(actual, expected):
    np.testing.assert_array_equal(actual['center'], expected['center'])
    np.testing.assert_array_equal(actual['normal'], expected['normal'])
    assert actual['radius'] == expected['radius']


def test_align_clamps_translation_and_preserves_rotation_and_off_axis_grasp():
    source = rim([.45, -.31, .3], [0, 0, 1], .014)
    source_before, target_before = copy.deepcopy(source), copy.deepcopy(TARGET)
    p0 = np.array([.32, -.2, .22])
    rpy = [43., -29., 117.]
    step = plan_step(p0, rpy, source, TARGET, max_translation_m=.006)
    delta = np.asarray(step['xyz'])-p0
    assert np.linalg.norm(delta) == pytest.approx(.006)
    np.testing.assert_allclose(step['source_center'], source['center']+delta)
    np.testing.assert_allclose(Rotation.from_euler('xyz', step['rpy_deg'], degrees=True).as_matrix(),
                               Rotation.from_euler('xyz', rpy, degrees=True).as_matrix(), atol=1e-12)
    assert not step['aligned'] and step['delta_tilt_deg'] == pytest.approx(0)
    assert_same_rim(source, source_before)
    assert_same_rim(TARGET, target_before)


def test_align_body_clearance_uses_body_radius_not_mouth_radius():
    source = aligned_source(60)
    source['center'][2] -= .08
    p0 = source['center']-[.1, .02, .2]
    step = plan_step(p0, [35, -25, 72], source, TARGET, max_translation_m=.2,
                     bottle_radius_m=.09)
    n = np.asarray(step['source_normal'])
    c = np.asarray(step['source_center'])
    body_low = c[2]-.35*max(n[2], 0.)-.09*np.linalg.norm(n[:2])
    assert body_low == pytest.approx(TARGET['center'][2]+.02)
    assert step['lip_xyz'][2] >= TARGET['center'][2]+.04
    assert not step['aligned']
    assert step['planned_aligned']


@pytest.mark.parametrize('tilt', [25., 60., 89., 90., 110., 145., 149.])
@pytest.mark.parametrize('rpy', [[43., -29., 117.], [-121., 58., -36.]])
def test_tilt_preserves_full_three_dimensional_mouth_offset_and_rim_alignment(tilt, rpy):
    source = aligned_source(tilt)
    R0 = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
    mouth_tcp = np.array([.081, -.217, .047])
    p0 = source['center']-R0 @ mouth_tcp
    source_before = copy.deepcopy(source)
    step = plan_step(p0, rpy, source, TARGET, mode='tilt', pour_direction_xy=DIRECTION)
    points = np.asarray(step['path']['points'])
    rotations = Rotation.from_euler('xyz', np.asarray(step['orientation']['points'])[:, 1:], degrees=True)
    increments = np.rad2deg((rotations[:-1].inv()*rotations[1:]).magnitude())
    assert increments.max() <= 1.+1e-9
    np.testing.assert_allclose(points[0], p0)
    initial_axis_tcp = R0.T @ source['normal']
    for point, R in zip(points, rotations.as_matrix()):
        center = point+R @ mouth_tcp
        normal = R @ initial_axis_tcp
        lip = lowest_lip(center, normal, source['radius'])
        np.testing.assert_allclose(lip[:2], TARGET['center'][:2], atol=1e-10)
        assert lip[2] >= TARGET['center'][2]+.04-1e-10
        body_low = center[2]-.35*max(normal[2], 0.)-.06*np.linalg.norm(normal[:2])
        assert body_low >= TARGET['center'][2]+.02-1e-10
    np.testing.assert_allclose(step['source_center'], points[-1]+rotations.as_matrix()[-1] @ mouth_tcp, atol=1e-10)
    assert step['tilt_deg'] > tilt
    assert step['tilt_deg'] <= 150.+1e-9
    assert step['delta_tilt_deg'] == pytest.approx(step['tilt_deg']-tilt)
    assert step['aligned']
    assert step['path_length_m'] <= .015+1e-10
    assert_same_rim(source, source_before)


def test_tilt_reduces_angle_instead_of_clipping_compensation():
    source = aligned_source(60)
    R0 = Rotation.from_euler('xyz', [23, -52, 132], degrees=True).as_matrix()
    p0 = source['center']-R0 @ [.09, -.32, .06]
    step = plan_step(p0, [23, -52, 132], source, TARGET, mode='tilt',
                     max_translation_m=.002, pour_direction_xy=DIRECTION)
    assert 0 < step['delta_tilt_deg'] < 5
    assert step['path_length_m'] == pytest.approx(.002, abs=1e-9)
    np.testing.assert_allclose(step['lip_xyz'][:2], TARGET['center'][:2], atol=1e-10)


def test_tilt_is_refused_until_aligned():
    source = aligned_source(45)
    source['center'][0] += .01
    with pytest.raises(ValueError, match='Align'):
        plan_step(source['center']-[.1, .1, .2], [0, 0, 0], source, TARGET, mode='tilt',
                  pour_direction_xy=DIRECTION)


def test_target_uses_upper_rim_height_and_rejects_excessive_slope():
    source = aligned_source(90)
    target = copy.deepcopy(TARGET)
    target['normal'] = [np.sin(np.deg2rad(10)), 0., np.cos(np.deg2rad(10))]
    step = plan_step(source['center']-[.1, .1, .2], [0, 0, 0], source, target,
                     max_translation_m=.1)
    upper = target['center'][2]+target['radius']*np.sin(np.deg2rad(10))
    assert step['lip_xyz'][2] >= upper+.04-1e-10
    target['normal'] = [np.sin(np.deg2rad(15)), 0., np.cos(np.deg2rad(15))]
    with pytest.raises(ValueError, match='15 degrees'):
        plan_step(source['center'], [0, 0, 0], source, target)


def test_target_checks_uncertainty_but_does_not_require_entire_source_mouth_to_fit():
    source = aligned_source(90, radius=.05)
    small_target = rim(TARGET['center'], radius=.02)
    result = plan_step(source['center']-[.1, .1, .2], [0, 0, 0], source, small_target,
                       max_translation_m=.1)
    assert result['aligned']
    small_target['radius'] = .012
    source['residual_m'] = .003
    with pytest.raises(ValueError, match='10 mm'):
        plan_step(source['center'], [0, 0, 0], source, small_target)


def test_tilt_direction_and_current_tilt_are_validated():
    source = aligned_source(60)
    with pytest.raises(ValueError, match='disagrees'):
        plan_step(source['center']-[.1, .1, .2], [0, 0, 0], source, TARGET, mode='tilt',
                  pour_direction_xy=-DIRECTION)
    source = aligned_source(151)
    with pytest.raises(ValueError, match='150'):
        plan_step(source['center'], [0, 0, 0], source, TARGET)
    source = aligned_source(0)
    with pytest.raises(ValueError, match='requires pour_direction'):
        plan_step(source['center']-[0, 0, .2], [0, 0, 0], source, TARGET, mode='tilt')


@pytest.mark.parametrize('parameter', [
    {'max_translation_m': 0}, {'max_translation_m': float('nan')},
    {'tilt_step_deg': -1}, {'bottle_radius_m': 0}, {'clearance_m': -1},
    {'tolerance_m': float('inf')}, {'pour_direction_xy': [0, 0]}, {'mode': 'bad'},
])
def test_plan_rejects_invalid_parameters(parameter):
    source = aligned_source(60)
    with pytest.raises(ValueError):
        plan_step(source['center'], [0, 0, 0], source, TARGET, **parameter)



def test_upright_edge_alignment_precedes_tilt_and_avoids_radius_jump():
    source = aligned_source(0, radius=.035)
    rpy = [47., -32., 113.]
    R0 = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
    mouth_tcp = np.array([.074, -.19, .056])
    p0 = source['center']-R0 @ mouth_tcp
    align = plan_step(p0, rpy, source, TARGET, pour_direction_xy=DIRECTION,
                      max_translation_m=.04)
    assert not align['aligned']
    assert align['error_m'] == pytest.approx(.035)
    assert align['planned_aligned']
    assert align['planned_error_m'] < 1e-12
    np.testing.assert_allclose(align['lip_xyz'][:2], TARGET['center'][:2], atol=1e-12)
    source_after = rim(align['source_center'], align['source_normal'], source['radius'])
    tilt = plan_step(align['xyz'], align['rpy_deg'], source_after, TARGET, mode='tilt',
                     pour_direction_xy=DIRECTION, max_translation_m=.015)
    assert tilt['aligned'] and tilt['delta_tilt_deg'] > .1
    assert tilt['path_length_m'] <= .015+1e-10
    points = np.asarray(tilt['path']['points'])
    rotations = Rotation.from_euler('xyz', np.asarray(tilt['orientation']['points'])[:, 1:], degrees=True).as_matrix()
    for i, (point, R) in enumerate(zip(points, rotations)):
        center = point+R @ mouth_tcp
        normal = R @ R0.T @ np.array([0., 0., 1.])
        lip = center+np.r_[source['radius']*DIRECTION, 0.] if i == 0 else lowest_lip(center, normal, source['radius'])
        np.testing.assert_allclose(lip[:2], TARGET['center'][:2], atol=1e-9)


def test_aligned_reports_observation_not_unexecuted_planned_endpoint():
    source = aligned_source(60)
    source['center'][0] -= .008
    step = plan_step(source['center']-[.1, .1, .2], [32, 19, -54], source, TARGET,
                     max_translation_m=.015, pour_direction_xy=DIRECTION)
    assert not step['aligned']
    assert step['error_m'] == pytest.approx(.008)
    assert step['planned_aligned'] and step['planned_error_m'] < 1e-12


@pytest.mark.parametrize('tilt', [0., 25., 89., 110.])
def test_compiled_interpolation_preserves_lip_and_body_clearance_within_one_mm(tilt):
    from urai.trajectory import compile_arm
    source = aligned_source(tilt, radius=.03)
    if tilt == 0:
        source['center'][:2] -= source['radius']*DIRECTION
    rpy = [43., -29., 117.]
    R0 = Rotation.from_euler('xyz', rpy, degrees=True).as_matrix()
    mouth_tcp = np.array([.081, -.217, .047])
    axis_tcp = R0.T @ source['normal']
    p0 = source['center']-R0 @ mouth_tcp
    step = plan_step(p0, rpy, source, TARGET, mode='tilt', pour_direction_xy=DIRECTION)
    spec = {'path': step['path'], 'orientation': step['orientation'],
            'speed': .01, 'gripper_events': [], 'angular_speed_deg_s': 5.,
            'approach': {'speed_m_s': .01, 'clearance_m': .12}}
    compiled = compile_arm(spec, {'xyz': p0, 'rpy_deg': rpy})
    for p, R in zip(compiled['xyz'], compiled['rotation_matrices']):
        c, n = p+R @ mouth_tcp, R @ axis_tcp
        lip = c+np.r_[source['radius']*DIRECTION, 0.] if np.linalg.norm(n[:2]) < 1e-8 else lowest_lip(c, n, source['radius'])
        assert np.linalg.norm(lip[:2]-TARGET['center'][:2]) < .001
        assert lip[2] >= TARGET['center'][2]+.04-.001
        body_low = c[2]-.35*max(n[2], 0.)-.06*np.linalg.norm(n[:2])
        assert body_low >= TARGET['center'][2]+.02-.001


def test_tilt_cap_and_requested_step_cap():
    source = aligned_source(149)
    p0 = source['center']-[.05, .08, .2]
    step = plan_step(p0, [18, -42, 56], source, TARGET, mode='tilt',
                     max_translation_m=.1, pour_direction_xy=DIRECTION)
    assert step['tilt_deg'] == pytest.approx(150)
    assert step['delta_tilt_deg'] == pytest.approx(1)
    capped_source = rim(step['source_center'], step['source_normal'], source['radius'])
    stopped = plan_step(step['xyz'], step['rpy_deg'], capped_source, TARGET, mode='tilt',
                        pour_direction_xy=DIRECTION)
    assert stopped['delta_tilt_deg'] == pytest.approx(0)
    np.testing.assert_allclose(stopped['xyz'], step['xyz'], atol=1e-12)
    source = aligned_source(45)
    step = plan_step(source['center']-[.05, .08, .2], [18, -42, 56], source, TARGET,
                     mode='tilt', tilt_step_deg=20, max_tilt_step_deg=2,
                     max_translation_m=.1, pour_direction_xy=DIRECTION)
    assert step['delta_tilt_deg'] == pytest.approx(2)


def test_tilt_bounds_include_the_remaining_observed_alignment_error():
    source = aligned_source(60)
    source['center'][0] += .002
    p0 = source['center']-[.05, .08, .2]
    step = plan_step(p0, [18, -42, 56], source, TARGET, mode='tilt',
                     max_translation_m=.004, pour_direction_xy=DIRECTION)
    assert step['aligned'] and step['error_m'] == pytest.approx(.002)
    assert step['path_length_m'] <= .004+1e-10
    assert step['delta_tilt_deg'] > 0
    points = np.asarray(step['path']['points'])
    assert np.linalg.norm(points[1]-points[0]) == pytest.approx(.002)
    with pytest.raises(ValueError, match='Finish alignment'):
        plan_step(p0, [18, -42, 56], source, TARGET, mode='tilt',
                  max_translation_m=.001, pour_direction_xy=DIRECTION)
