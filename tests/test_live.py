"""Live MJPEG stream: subscription-driven worker, frame sources, shared encoding and the capture lock.

Frame sources exercised here: ``simulation`` (the synthetic backend) and ``capture`` (the PiPER-X backend on the
kinematic simulator, whose synthetic cameras have no frame pump). The ``stream`` source needs a RealSense head
camera with its frame pump running and is covered by the real-robot checks.
"""
import io
import re
import socket
import statistics
import threading
import time

import cv2
import httpx
import numpy as np
import pytest
import uvicorn
from conftest import make_client
from PIL import Image

from urai.app import create_app
from urai.backend import SimulationBackend
from urai.live import (
    CAPTURE_FPS,
    LiveStream,
    encode_jpeg,
    live_settings_from_env,
    scale_to_width,
)
from urai.service import Service


def eventually(predicate, timeout=3.):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.02)
    return predicate()


def latest_index(live):
    """Index of the newest published frame (0 before the first one)."""
    frame = live.wait_frame(after=-1, timeout=0.)
    return 0 if frame is None else frame.index


# ------------------------------------------------------------ worker lifecycle

def test_stream_runs_only_while_subscribed_and_stops_after_idle():
    live = LiveStream(SimulationBackend(), idle_timeout_s=.3)
    assert not live.running and live.subscribers == 0
    with live.subscription():
        assert live.subscribers == 1
        frame = live.wait_frame(after=0, timeout=3.)
        assert frame is not None and frame.index >= 1
        assert frame.jpeg.startswith(b'\xff\xd8')
        later = live.wait_frame(after=frame.index, timeout=3.)
        assert later.index > frame.index
        assert live.running
    assert live.subscribers == 0
    assert live.running, 'the worker lingers briefly so a reconnecting client does not restart the camera path'
    assert eventually(lambda: not live.running, timeout=3.)


def test_resubscribing_during_the_idle_linger_gets_a_fresh_frame_at_once():
    live = LiveStream(SimulationBackend(), idle_timeout_s=2.)
    with live.subscription():
        last = live.wait_frame(after=0, timeout=3.)
    time.sleep(.2)
    assert live.running, 'the worker lingers'
    with live.subscription():
        began = time.monotonic()
        frame = live.wait_frame(after=latest_index(live), timeout=.5)
        assert frame is not None, 'the lingering worker must wake on subscribe, not at its idle deadline'
        assert time.monotonic() - began < .3
        assert frame.timestamp > last.timestamp
    live.close()


def test_no_subscriber_means_no_capture_and_no_encoding():
    """Every capture is encoded and published, so a frame index that stands still means nothing was captured."""
    live = LiveStream(SimulationBackend(), idle_timeout_s=.2)
    assert live.wait_frame(after=0, timeout=.3) is None and not live.running
    with live.subscription():
        assert live.wait_frame(after=0, timeout=3.) is not None
    assert eventually(lambda: not live.running, 3.)
    settled = latest_index(live)
    assert live.wait_frame(after=settled, timeout=.4) is None
    assert live.status()['fps'] >= 0.


def test_subscribers_share_one_encoded_frame():
    live = LiveStream(SimulationBackend(), idle_timeout_s=.2)
    with live.subscription(), live.subscription():
        first = live.wait_frame(after=0, timeout=3.)
        second = live.wait_frame(after=0, timeout=3.)
        assert first is second, 'both subscribers receive the same LiveFrame object; nothing is re-encoded per client'
        assert live.subscribers == 2
    assert eventually(lambda: not live.running, 3.)


# ------------------------------------------------------------ encoding

def test_frames_are_scaled_to_the_stream_width_and_encoded_with_opencv_at_quality_70():
    backend = SimulationBackend()
    live = LiveStream(backend, idle_timeout_s=.1)
    assert (live.width, live.quality, live.max_fps) == (640, 70, 30.)
    with live.subscription():
        frame = live.wait_frame(after=0, timeout=3.)
    source = backend.capture().rgb
    image = Image.open(io.BytesIO(frame.jpeg))
    assert image.size == (640, round(source.shape[0] * 640 / source.shape[1]))
    assert (frame.width, frame.height) == image.size
    reference = encode_jpeg(cv2.cvtColor(scale_to_width(source, 640), cv2.COLOR_RGB2BGR), 70)
    assert frame.jpeg == reference, 'the stream frame is exactly the OpenCV encoding of the area-scaled picture'
    assert frame.source == 'simulation'
    assert frame.captured <= frame.timestamp


def test_width_and_quality_are_configurable_and_narrow_pictures_are_not_upscaled():
    backend = SimulationBackend()
    source = backend.capture().rgb
    sizes = {}
    for quality in (30, 90):
        live = LiveStream(backend, width=400, quality=quality, idle_timeout_s=.1)
        with live.subscription():
            frame = live.wait_frame(after=0, timeout=3.)
        assert (frame.width, frame.height) == (400, round(source.shape[0] * 400 / source.shape[1]))
        sizes[quality] = len(frame.jpeg)
    assert sizes[30] < sizes[90]
    wide = LiveStream(backend, width=1920, idle_timeout_s=.1)
    with wide.subscription():
        frame = wide.wait_frame(after=0, timeout=3.)
    assert (frame.width, frame.height) == (source.shape[1], source.shape[0])


def test_live_settings_come_from_the_environment():
    assert live_settings_from_env({}) == {}
    assert live_settings_from_env({'URAI_LIVE_WIDTH': '960', 'URAI_LIVE_QUALITY': '80', 'URAI_LIVE_FPS': '15'}) == {
        'width': 960, 'quality': 80, 'max_fps': 15}
    for bad in ({'URAI_LIVE_WIDTH': 'wide'}, {'URAI_LIVE_QUALITY': '0'}, {'URAI_LIVE_FPS': '61'}):
        with pytest.raises(ValueError, match=next(iter(bad))):
            live_settings_from_env(bad)


def test_scaling_and_encoding_a_head_frame_costs_a_few_milliseconds():
    rng = np.random.default_rng(0)
    picture = cv2.GaussianBlur(rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8), (0, 0), 4)
    live = LiveStream(SimulationBackend())
    costs = []
    for _ in range(60):
        began = time.perf_counter()
        jpeg, width, height = live._encode(picture, 'bgr')
        costs.append((time.perf_counter() - began) * 1000.)
    assert (width, height) == (640, 360)
    assert 2000 < len(jpeg) < 200000
    assert statistics.median(costs) < 8., f'median {statistics.median(costs):.2f} ms per 1280x720 frame'


# ------------------------------------------------------------ PiPER-X backend on the kinematic simulator

def test_without_a_camera_stream_the_head_camera_is_captured_at_most_three_times_per_second(piper_backend):
    cameras = piper_backend.runtime.cameras
    assert cameras.streaming('head') is False
    with pytest.raises(TimeoutError):
        cameras.wait_color('head', after_seq=None, timeout_s=.1)
    live = LiveStream(piper_backend, idle_timeout_s=.1)
    with live.subscription():
        first = live.wait_frame(after=0, timeout=5.)
        began = time.monotonic()
        time.sleep(1.5)
        published = latest_index(live) - first.index
        elapsed = time.monotonic() - began
    assert first.source == 'capture' and live.status()['source'] == 'capture'
    assert 1 <= published <= CAPTURE_FPS * elapsed + 1
    assert (first.width, first.height) == (640, 360)
    # The arms stand still, so a fresh capture renders the same picture: the frame is its RGB pixels, encoded.
    rgb = piper_backend.runtime.capture('head').rgb
    assert first.jpeg == encode_jpeg(cv2.cvtColor(scale_to_width(rgb, 640), cv2.COLOR_RGB2BGR), 70)
    live.close()


def test_the_capture_source_waits_for_the_lock_that_observations_hold(piper_backend):
    live = LiveStream(piper_backend, idle_timeout_s=.1)
    service = Service(piper_backend)
    with piper_backend.capture_lock:
        with live.subscription():
            assert live.wait_frame(after=0, timeout=.5) is None, 'no capture while an observation holds the camera'
        observing = threading.Thread(target=service.observe)
        observing.start()
        observing.join(.5)
        assert observing.is_alive() and service.frame is None, 'Service.observe waits for the same lock'
    observing.join(10)
    assert service.frame is not None
    with live.subscription():
        assert live.wait_frame(after=0, timeout=5.) is not None
    live.close()


def test_a_camera_held_too_long_is_reported_and_the_preview_recovers(piper_backend):
    live = LiveStream(piper_backend, idle_timeout_s=.5)
    with live.subscription():
        with piper_backend.capture_lock:
            assert eventually(lambda: live.status()['error'] is not None, 5.)
            assert live.status()['error'].startswith('TimeoutError')
            held = latest_index(live)
        assert live.wait_frame(after=held, timeout=5.) is not None
        assert live.status()['error'] is None
    live.close()


def test_the_simulator_preview_leaves_a_50_hz_loop_undisturbed(piper_backend):
    """Rendering the simulated head camera shares the process with the 50 Hz control loop, which aborts a motion
    after a 250 ms scheduling gap."""
    live = LiveStream(piper_backend, idle_timeout_s=.1)
    stop = threading.Event()
    gaps = []

    def control_loop():
        last = time.monotonic()
        while not stop.is_set():
            time.sleep(.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    loop = threading.Thread(target=control_loop)
    loop.start()
    with live.subscription():
        time.sleep(2.)
        status = live.status()
    stop.set()
    loop.join(2.)
    live.close()
    assert status['source'] == 'capture' and status['fps'] > 1., status
    assert max(gaps) < .25, f'control loop stalled {max(gaps)*1000:.0f} ms'
    assert statistics.median(gaps) < .03


# ------------------------------------------------------------ HTTP

def test_mjpg_route_streams_multipart_jpeg_parts_with_latency_headers_and_releases_the_subscription(sim_service):
    # Starlette's TestClient buffers a response until the app finishes, so an endless stream needs a real server.
    app = create_app(sim_service)
    live = app.state.live
    live.idle_timeout_s = .3
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert eventually(lambda: server.started, 10.)
    try:
        body = b''
        with httpx.stream('GET', f'http://127.0.0.1:{port}/api/live.mjpg', timeout=5.) as response:
            assert response.status_code == 200
            assert response.headers['content-type'] == 'multipart/x-mixed-replace; boundary=urai-frame'
            assert response.headers['cache-control'] == 'no-store'
            for chunk in response.iter_bytes():
                body += chunk
                if body.count(b'--urai-frame\r\n') >= 3:
                    break
            assert live.subscribers == 1
        parts = body.split(b'--urai-frame\r\n')[1:3]
        indices = []
        for part in parts:
            header, _, payload = part.partition(b'\r\n\r\n')
            assert b'Content-Type: image/jpeg' in header
            length = int(re.search(rb'Content-Length: (\d+)', header).group(1))
            indices.append(int(re.search(rb'X-Frame-Index: (\d+)', header).group(1)))
            age = float(re.search(rb'X-Age-Ms: ([\d.]+)', header).group(1))
            assert 0. <= age < 5000.
            assert payload[:2] == b'\xff\xd8' and len(payload) >= length
        assert indices[1] > indices[0]
        assert eventually(lambda: live.subscribers == 0, 5.), 'client disconnect must release the subscription'
        assert eventually(lambda: not live.running, 3.)
    finally:
        server.should_exit = True
        thread.join(10.)
    assert not thread.is_alive()


def test_closing_the_multipart_generator_unsubscribes():
    live = LiveStream(SimulationBackend(), idle_timeout_s=.2)
    parts = live.multipart()
    first = next(parts)
    assert first.startswith(b'--urai-frame\r\n') and live.subscribers == 1
    parts.close()
    assert live.subscribers == 0
    assert eventually(lambda: not live.running, 3.)


def test_state_reports_live_stream_status_and_snapshot_is_a_full_resolution_jpeg(sim_service):
    client = make_client(sim_service)
    status = client.get('/api/state').json()['live']
    assert status == {'running': False, 'subscribers': 0, 'source': None, 'error': None, 'fps': 0., 'encode_ms': None,
                      'age_ms': None, 'width': None, 'height': None, 'quality': 70, 'max_fps': 30.}
    response = client.get('/api/live.jpg')
    assert response.status_code == 200 and response.headers['content-type'] == 'image/jpeg'
    source = sim_service.backend.capture().rgb
    assert Image.open(io.BytesIO(response.content)).size == (source.shape[1], source.shape[0])


def test_the_simulator_snapshot_is_the_head_camera_at_full_resolution(piper_service):
    response = make_client(piper_service).get('/api/live.jpg')
    assert response.status_code == 200 and response.headers['content-type'] == 'image/jpeg'
    rgb = piper_service.backend.runtime.capture('head').rgb
    assert Image.open(io.BytesIO(response.content)).size == (rgb.shape[1], rgb.shape[0])
