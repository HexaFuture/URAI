"""A toss whose grasp is a drawn line: the two fingertip positions are used directly, nothing is segmented."""
import time
import numpy as np
import pytest
from conftest import make_client
from test_transfer import scene
from urai.grasp import grasp_from_line, grasp_terms, line_object
from urai.tasks import arm_bases
from urai.toss import toss_between
from urai.transfer import Unsegmented, locate_object

LINE = [[300, 226], [490, 226]]
DROP_POINT = [.60, -.45, 0.]


def test_a_long_line_on_flat_cloth_keeps_its_midpoint_and_axis_without_segmentation():
    frame = scene([])
    with pytest.raises(Unsegmented):       # nothing rises above this table: segmentation has nothing to find
        locate_object(frame, [395, 226], 0.)
    found = line_object(frame, LINE, table_z_m=0., min_tcp_z_m=-.003)
    assert found['width'] == .07
    expected = grasp_from_line(frame, LINE, table_z_m=0., min_tcp_z_m=-.003)
    result = toss_between(frame, [395, 226], [.60, -.45, 0.], style='sidearm', base_xy=[0., 0.],
                          grasp_pixels=LINE, grasp_options={'table_z_m': 0., 'min_tcp_z_m': -.003})
    np.testing.assert_allclose(result['source_xyz'], expected['arm']['path']['points'][-1])
    reverse = toss_between(frame, [395, 226], [.60, -.45, 0.], style='sidearm', base_xy=[0., 0.],
                           grasp_pixels=LINE[::-1], grasp_options={'table_z_m': 0., 'min_tcp_z_m': -.003})
    np.testing.assert_allclose(result['arm']['orientation']['points'], reverse['arm']['orientation']['points'], atol=1e-8)


def test_the_queue_stores_the_world_line_of_a_flat_surface_without_segmentation(sim_service):
    service = sim_service
    service.observe()
    service.frame.depth[:] = 1.          # bare table: a segmented item would be refused
    service.set_drop_point(DROP_POINT)
    item = service.tasks._build_item(0, {'object_pixel': [395, 226], 'grasp_pixels': LINE, 'arm': 'left',
                                         'placement': 'toss'},
                                     service.frame, None, service.backend.state(), arm_bases(service.backend))
    assert item['width_mm'] == 70
    assert len(item['grasp_line_world']) == 2
    assert item['placement'] == 'toss'
    assert service.execution['state'] == 'idle'
    assert not service.tasks.active
    with pytest.raises(ValueError, match='画线队列仅支持 toss'):
        service.tasks._build_item(0, {'object_pixel': [395, 226], 'grasp_pixels': LINE, 'arm': 'left'},
                                  service.frame, None, service.backend.state(), arm_bases(service.backend))


def test_a_queued_line_is_replayed_on_the_fresh_observation_and_tossed(sim_service):
    """The queue re-observes, re-projects the stored world line, grasps it with the arm's grasp terms and throws."""
    service = sim_service
    service.observe()
    client = make_client(service)
    assert client.put('/api/settings', json={'motion_profile': 'throw'}).status_code == 200
    submitted = service.frame.id
    response = client.post('/api/tasks', json={'observation_id': submitted, 'toss_style': 'sidearm',
                                               'items': [{'object_pixel': [395, 226], 'grasp_pixels': LINE,
                                                          'arm': 'left', 'placement': 'toss', 'landing_xyz': DROP_POINT}]})
    assert response.status_code == 200, response.text
    queued = response.json()['items'][0]
    deadline = time.monotonic()+120
    while time.monotonic() < deadline and client.get('/api/tasks').json()['active']:
        time.sleep(.2)
    item = client.get('/api/tasks').json()['items'][0]
    assert item['status'] == 'completed', item
    assert item['observation_id'] not in (None, submitted)            # planned on its own fresh observation
    frame = service.frame
    yaw = service.backend.state()['left']['rpy_deg'][2]               # the arm is back at the pose it threw from
    pixels = [frame.project(point)[0].tolist() for point in queued['grasp_line_world']]
    expected = line_object(frame, pixels, clearance_m=.12, speed_m_s=.10, preferred_yaw_deg=yaw,
                           **grasp_terms('left', None, service.backend.table_checks), surface_xyz=queued['object_xyz'])
    np.testing.assert_allclose(item['grasp_xyz'], expected['manual_grasp'], atol=1e-9)


def test_saved_grasp_survives_depth_occlusion():
    frame = scene([])
    saved = line_object(frame, LINE)
    surface = [*saved['center'], saved['top']]
    frame.depth[180:270, 340:450] = np.nan
    found = line_object(frame, LINE, surface_xyz=surface)
    np.testing.assert_allclose(found['center'], saved['center'])
    np.testing.assert_allclose(found['manual_grasp'], saved['manual_grasp'])
    result = toss_between(frame, [395, 226], [.60, -.45, 0.], style='sidearm', base_xy=[0., 0.],
                          grasp_pixels=LINE, grasp_options={'surface_xyz': surface})
    np.testing.assert_allclose(result['source_xyz'], saved['manual_grasp'])
