"""Every released tool end to end on the PiPER-X backend driving the kinematic simulator.

Each test observes the simulated scene through the head camera, drafts the tool on pixels projected from the
simulated objects' true positions, previews it (inverse kinematics, joint limits, table, self and inter-arm
collision checks, retiming), executes it on the 50 Hz control loop and checks the outcome in the scene. Motion
runs on the wall clock, so this file takes several minutes.
"""
import dataclasses
import math
import time
import numpy as np
import pytest
from conftest import make_client
from urai.backend import PiperBackend
from urai.robot.scene import Cup, Cylinder, Scene
from urai.robot.simulated import build_simulated_runtime, default_scene
from urai.service import Service

#: The pose tolerance the paper's real-robot trials ran with (configs/paper-settings.json).
PAPER_POSE_TOLERANCE = {'position_mm': 50., 'orientation_deg': 40.}


@pytest.fixture
def simulator(arm_models):
    """Builds a service on the PiPER-X backend over a simulator whose default scene is rearranged as asked."""
    built = []

    def build(moves=None, parts=None):
        scene = default_scene(arm_models)
        objects = []
        for item in scene.objects():
            if item.name in (moves or {}):
                x, y, yaw = moves[item.name]
                pose = np.eye(4)
                pose[:2, :2] = [[math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))],
                                [math.sin(math.radians(yaw)), math.cos(math.radians(yaw))]]
                pose[:3, 3] = [x, y, item.t_world_obj[2, 3]]
                item = dataclasses.replace(item, t_world_obj=pose)
            if item.name in (parts or {}):
                item = dataclasses.replace(item, parts=parts[item.name])
            objects.append(item)
        runtime = build_simulated_runtime(scene=Scene(scene.table, objects))
        backend = PiperBackend(runtime)
        built.append((runtime, backend))
        return Service(backend), runtime
    yield build
    for runtime, backend in built:
        backend.close()
        runtime.close()


def scene_of(runtime):
    return runtime.cameras.scene


def point_on(runtime, name, height_fraction=1.):
    """World point on the vertical axis of object ``name``, ``height_fraction`` of its height above its base."""
    scene = scene_of(runtime)
    item = next(o for o in scene.objects() if o.name == name)
    _, high = item.local_bounds()
    return (scene.pose(name) @ [0., 0., high[2]*height_fraction, 1.])[:3]


def pixel(service, xyz):
    return service.frame.project(np.asarray(xyz, dtype=float))[0].tolist()


def table_point(service, xy):
    return pixel(service, [xy[0], xy[1], service.table_z()])


def held(runtime):
    return {arm: runtime.arms[arm].held_object() for arm in ('left', 'right')}


def run_skill(service, client, name, arm, pixels, **numbers):
    """Draft ``name`` on the current observation, then preview and execute it; returns (draft, preview, execution)."""
    response = client.post(f'/api/skills/{name}', json={'arm': arm, 'observation_id': service.frame.id,
                                                        'pixels': pixels, 'commit': True, **numbers})
    assert response.status_code == 200, response.json()['detail']
    draft = response.json()
    preview = client.post('/api/preview', json={'expected_revision': draft['revision']})
    assert preview.status_code == 200, preview.json()['detail']
    started = client.post('/api/execute', json={'preview_id': preview.json()['id']})
    assert started.status_code == 200, started.text
    service.worker.join(timeout=300)
    return draft, preview.json(), service.execution


def assert_home(service):
    for arm, state in service.backend.state().items():
        np.testing.assert_allclose(state['joints_deg'], np.zeros(6), atol=.7, err_msg=arm)


def assert_placed(runtime, name, xy, tolerance_m=.015):
    scene = scene_of(runtime)
    pose = scene.pose(name)
    assert np.linalg.norm(pose[:2, 3]-np.asarray(xy)) < tolerance_m, pose[:3, 3]
    assert pose[2, 2] == pytest.approx(1.)                       # standing upright on the table
    assert pose[2, 3] == pytest.approx(scene.table.z, abs=1e-6)
    assert name not in held(runtime).values()


def test_pick_place_moves_the_block_to_the_drawn_spot(simulator):
    service, runtime = simulator()
    client = make_client(service)
    client.post('/api/observe', json={})
    landing = [.30, -.02]
    draft, preview, execution = run_skill(service, client, 'pick_place', 'left',
                                          [pixel(service, point_on(runtime, 'red_block')), table_point(service, landing)])
    assert execution['state'] == 'completed', execution.get('error')
    assert preview['pose_tolerance'] == {'position_mm': 10., 'orientation_deg': 3.}
    assert preview['motion_limits']['profile'] == 'normal' and preview['table_checks'] is True
    assert execution['draft_cleared'] and service.draft == {}
    assert_placed(runtime, 'red_block', landing)
    assert_home(service)


def test_pinch_picks_at_the_drawn_table_point(simulator):
    """A pinch closes the jaws at the table where the stroke starts; drawn at the block's foot it takes the block."""
    service, runtime = simulator()
    client = make_client(service)
    client.post('/api/observe', json={})
    landing = [.30, -.02]
    draft, preview, execution = run_skill(service, client, 'pick_place', 'left',
                                          [pixel(service, point_on(runtime, 'red_block', 0.)), table_point(service, landing)],
                                          pinch=True)
    assert draft['pinch'] and 'verify' not in draft['arm']['gripper_events'][1]
    assert execution['state'] == 'completed', execution.get('error')
    assert_placed(runtime, 'red_block', landing)
    assert_home(service)


def test_top_down_pick_place_keeps_the_wrist_vertical(simulator):
    """An upright wrist reaches about 0.33 m from a PiPER base, so the block starts 0.30 m from the left base."""
    service, runtime = simulator(moves={'red_block': (.27, -.12, 15.)})
    client = make_client(service)
    client.post('/api/observe', json={})
    landing = [.20, -.02]
    draft, preview, execution = run_skill(service, client, 'pick_place', 'left',
                                          [pixel(service, point_on(runtime, 'red_block')), table_point(service, landing)],
                                          top_down=True)
    assert all(key[1:3] == [180., 0.] for key in draft['arm']['orientation']['points'])
    assert execution['state'] == 'completed', execution.get('error')
    assert_placed(runtime, 'red_block', landing)
    assert_home(service)


def grasp_line_pixels(service, runtime, name, axis_deg, half_m):
    """A line across the top of ``name`` along ``axis_deg``, ``2*half_m`` long."""
    top = point_on(runtime, name)
    along = np.array([math.cos(math.radians(axis_deg)), math.sin(math.radians(axis_deg)), 0.])*half_m
    return [pixel(service, top-along), pixel(service, top+along)]


def test_grasp_line_grasps_the_block_and_comes_home_holding_it(simulator):
    service, runtime = simulator()
    client = make_client(service)
    client.post('/api/observe', json={})
    draft, preview, execution = run_skill(service, client, 'grasp_line', 'left',
                                          grasp_line_pixels(service, runtime, 'red_block', 15., .035))
    assert draft['grasp_mode'] in ('top_down', 'tilted_side') and draft['width_mm'] == pytest.approx(70, abs=.01)
    assert execution['state'] == 'completed', execution.get('error')
    assert held(runtime) == {'left': 'red_block', 'right': None}
    assert_home(service)


def test_carry_line_grasps_flies_the_drawn_route_and_releases_at_its_end(simulator):
    """The carry holds the landing's wrist orientation along the whole flight, so the drawn route stays about as near
    the base as the landing: a flight point much farther out would need a lean the wrist cannot give at the
    default 3 degree tolerance (j4 reaches its limit)."""
    service, runtime = simulator()
    client = make_client(service)
    client.post('/api/observe', json={})
    top = point_on(runtime, 'red_block')
    # The leading 40 mm of the stroke is the grasp line (grasp_span_mm): drawn centred across the block's top. The
    # simulated jaws only take an object centred between them, where real ones would push it to the middle.
    along = np.array([math.cos(math.radians(105.)), math.sin(math.radians(105.)), 0.])*.02
    landing = np.r_[.28, -.02, service.table_z()]
    route = [top-along, top+along, *[np.r_[xy, service.table_z()] for xy in np.linspace([.30, -.17], landing[:2], 10)]]
    pixels = [pixel(service, start+(end-start)*t) for start, end in zip(route[:-1], route[1:])
              for t in np.linspace(0., 1., 8, endpoint=False)]+[pixel(service, route[-1])]
    draft, preview, execution = run_skill(service, client, 'carry_line', 'left', pixels)
    assert draft['reach_error_mm'] <= 10. and draft['reach_error_deg'] <= 3.
    assert draft['carry_height_mm'] >= draft['obstacle_top_mm']+draft['object_hang_mm']
    assert execution['state'] == 'completed', execution.get('error')
    assert_placed(runtime, 'red_block', landing[:2], tolerance_m=.02)
    assert_home(service)


def test_pour_lifts_the_cup_tips_it_over_the_bowl_and_puts_it_back(simulator):
    """The simulator's default cup is 80 mm wide, more than the 70 mm jaws open, so this scene holds a 60 mm cup.

    The pour ends by returning in a straight line to the arm's start pose, the folded rest pose here, whose
    neighbourhood the default 3 degree tolerance does not admit; the paper's pose tolerance does.
    """
    service, runtime = simulator(parts={'cup': (Cup(.03, .095, .003, .006, (60, 165, 95)),)})
    client = make_client(service)
    assert client.put('/api/settings', json={'pose_tolerance': PAPER_POSE_TOLERANCE}).status_code == 200
    client.post('/api/observe', json={})
    cup = scene_of(runtime).pose('cup')[:3, 3].copy()
    draft, preview, execution = run_skill(service, client, 'pour', 'left',
                                          [pixel(service, point_on(runtime, 'cup', .6)),
                                           pixel(service, point_on(runtime, 'bowl', 0.)+[0., 0., .01])])
    assert draft['reach_error_mm'] <= 3. and draft['reach_error_deg'] <= 3.
    assert draft['cup_diameter_mm'] == pytest.approx(60, abs=6)
    assert execution['state'] == 'completed', execution.get('error')
    # The cup was lifted, tipped over the bowl, put back where it stood and let go.
    assert_placed(runtime, 'cup', cup[:2], tolerance_m=.01)
    assert_home(service)


def test_bottle_cap_stages_hold_the_bottle_try_the_twist_and_place_it(simulator):
    """The default scene's bottle is 182 mm tall: with the table checks on, a body grip low enough to leave the cap
    120 mm from the holder puts the wrist below the 60 mm link clearance, so this scene holds a 210 mm bottle.

    The simulated gripper only closes on an object no other gripper holds, so the twisting arm finds nothing
    between its jaws at the cap of the held bottle and the twist stops at its first grip check, the way a missed
    cap stops it on the robot; the stages around it run as they would there.
    """
    tall = (Cylinder(.032, .185, (95, 145, 205)), Cylinder(.016, .025, (235, 200, 45), z0=.185))
    service, runtime = simulator(parts={'bottle': tall})
    client = make_client(service)
    scene = scene_of(runtime)
    client.post('/api/observe', json={})
    draft, preview, execution = run_skill(service, client, 'bottle_cap_prepare', 'right',
                                          [pixel(service, point_on(runtime, 'bottle'))])
    assert execution['state'] == 'completed', execution.get('error')
    assert held(runtime) == {'left': None, 'right': 'bottle'}
    assert abs(scene.pose('bottle')[2, 2]) < 1e-3                     # the bottle axis is horizontal
    assert service.bottle_cap['state'] == 'waiting_cap' and service.bottle_cap['holder'] == 'right'
    holding = service.backend.state()['right']['joints_deg']
    assert np.abs(holding).max() > 10.                                # the holder stays, it does not go home

    client.post('/api/observe', json={})
    cap = point_on(runtime, 'bottle')
    draft, preview, execution = run_skill(service, client, 'bottle_cap_twist', 'left', [pixel(service, cap)])
    assert set(draft['arms']) == {'left'} and len(draft['phases']) == 6
    assert preview['pose_tolerance'] == {'position_mm': 3., 'orientation_deg': 3.}
    assert execution['state'] == 'error' and '空抓' in execution['error'], execution
    assert service.bottle_cap['state'] == 'needs_relocalization'
    assert held(runtime)['right'] == 'bottle'
    np.testing.assert_allclose(service.backend.state()['right']['joints_deg'], holding, atol=.5)

    client.post('/api/recover', json={})
    client.post('/api/observe', json={})
    spot = [.30, -.30]
    draft, preview, execution = run_skill(service, client, 'bottle_cap_place', 'right', [table_point(service, spot)])
    assert preview['phase_order'] == ['worker_retract', 'worker_home', 'holder_place']
    assert execution['state'] == 'completed', execution.get('error')
    assert service.bottle_cap is None
    assert_placed(runtime, 'bottle', spot, tolerance_m=.01)
    assert_home(service)


def test_the_toss_queue_throws_the_block_overhand(simulator):
    """The queue observes, relocates the block, plans the overhand throw and executes it under the throw profile.

    The simulator has no flight: a released object settles straight below where the jaws let it go.
    """
    service, runtime = simulator()
    client = make_client(service)
    scene = scene_of(runtime)
    client.post('/api/observe', json={})
    item = {'object_xy': scene.pose('red_block')[:2, 3].tolist(), 'landing_xyz': [.62, -.10, service.table_z()],
            'placement': 'toss', 'arm': 'left'}
    refused = client.post('/api/tasks', json={'observation_id': service.frame.id, 'items': [item]})
    assert refused.status_code == 409 and 'throw' in refused.json()['detail']
    assert client.put('/api/settings', json={'motion_profile': 'throw'}).status_code == 200
    response = client.post('/api/tasks', json={'observation_id': service.frame.id, 'items': [item]})
    assert response.status_code == 200, response.text
    deadline = time.monotonic()+300
    while time.monotonic() < deadline and client.get('/api/tasks').json()['active']:
        time.sleep(.2)
    done = client.get('/api/tasks').json()['items'][0]
    assert done['status'] == 'completed', done
    assert done['toss_style'] == 'overhand' and done['throw_line'] in ('fast', 'near')
    assert done['landing_shortfall_m'] == 0. and done['release_speed_m_s'] > .5
    assert held(runtime) == {'left': None, 'right': None}
    released = np.asarray(done['release_xyz'])
    np.testing.assert_allclose(scene.pose('red_block')[:2, 3], released[:2], atol=.03)
    assert_home(service)
