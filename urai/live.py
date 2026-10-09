"""Live RGB preview: one background worker feeds every MJPEG subscriber.

The worker runs only while at least one client is subscribed and stops itself
after ``idle_timeout_s`` without subscribers. Each camera frame is scaled and
JPEG-encoded once with OpenCV (libjpeg-turbo, GIL released) and the encoded
bytes are shared by every subscriber.

Frame sources:

* ``stream`` – the robot runtime streams the head RealSense continuously
  (:meth:`urai.robot.cameras.RealSenseRig.wait_color`). Only the colour plane is read: no depth
  alignment, no depth conversion and no BGR→RGB copy, none of which a preview needs. The stream
  serves concurrent readers, so this path does not take ``backend.capture_lock`` and never
  delays ``Service.observe``.
* ``capture`` – head camera without a stream: a full capture owns the pipeline, so it is
  serialized with observations through ``backend.capture_lock`` at a low rate.
* ``simulation`` – synthetic frames of a backend without a camera stream.
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import cv2

STREAM_FPS = 30.  # the head RealSense streams at 30 fps; the worker never asks for more
CAPTURE_FPS = 3.
STREAM_TIMEOUT_S = 1.
LOCK_TIMEOUT_S = 2.
ERROR_RETRY_S = .5
BOUNDARY = 'urai-frame'
DEFAULT_WIDTH = 640
DEFAULT_QUALITY = 70
FPS_WINDOW_S = 2.
#: Environment overrides read by :func:`live_settings_from_env`: variable → (keyword, minimum, maximum).
ENV_SETTINGS = {'URAI_LIVE_WIDTH': ('width', 160, 1920), 'URAI_LIVE_QUALITY': ('quality', 1, 100),
                'URAI_LIVE_FPS': ('max_fps', 1, 60)}


def live_settings_from_env(environ=os.environ):
    """``LiveStream`` keyword overrides from ``URAI_LIVE_WIDTH`` / ``URAI_LIVE_QUALITY`` / ``URAI_LIVE_FPS``."""
    settings = {}
    for name, (key, low, high) in ENV_SETTINGS.items():
        raw = environ.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f'{name} must be an integer, got {raw!r}') from None
        if not low <= value <= high:
            raise ValueError(f'{name} must be within [{low}, {high}], got {value}')
        settings[key] = value
    return settings


def scale_to_width(picture, width):
    """Area-filtered downscale so the picture is at most ``width`` pixels wide; narrower pictures pass through."""
    height, current = picture.shape[:2]
    if current <= width:
        return picture
    return cv2.resize(picture, (width, round(height * width / current)), interpolation=cv2.INTER_AREA)


def encode_jpeg(bgr, quality):
    """JPEG bytes of a BGR picture."""
    return cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1].tobytes()


@dataclass(frozen=True)
class LiveFrame:
    index: int
    jpeg: bytes
    width: int
    height: int
    source: str
    timestamp: float  # published, epoch seconds
    captured: float  # camera arrival (stream) or grab time, epoch seconds


class LiveStream:
    def __init__(self, backend, *, width=DEFAULT_WIDTH, quality=DEFAULT_QUALITY, idle_timeout_s=3.,
                 max_fps=STREAM_FPS, capture_fps=CAPTURE_FPS):
        self.backend = backend
        self.width = width
        self.quality = quality
        self.idle_timeout_s = idle_timeout_s
        self.max_fps = max_fps
        self.capture_fps = capture_fps
        self._condition = threading.Condition()
        self._subscribers = 0
        self._idle_since = None
        self._thread = None
        self._closed = False
        self._index = 0
        self._latest = None
        self._last_seq = None
        self._source = None
        self._error = None
        self._published = deque(maxlen=240)
        self._encode_ms = None
        self._age_ms = None

    # ------------------------------------------------------------ subscriptions

    @property
    def subscribers(self):
        with self._condition:
            return self._subscribers

    @property
    def running(self):
        thread = self._thread
        return thread is not None and thread.is_alive()

    def subscribe(self):
        with self._condition:
            if self._closed:
                raise RuntimeError('实时画面已关闭')
            self._subscribers += 1
            self._idle_since = None
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name='urai-live', daemon=True)
                self._thread.start()
            self._condition.notify_all()  # a lingering worker resumes at once instead of at its idle deadline

    def unsubscribe(self):
        with self._condition:
            self._subscribers -= 1
            if self._subscribers == 0:
                self._idle_since = time.monotonic()
            self._condition.notify_all()

    @contextmanager
    def subscription(self):
        self.subscribe()
        try:
            yield self
        finally:
            self.unsubscribe()

    def close(self):
        """Stop the worker for good (server shutdown); open generators end at their next wait."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(LOCK_TIMEOUT_S + STREAM_TIMEOUT_S)

    def status(self):
        with self._condition:
            now = time.monotonic()
            recent = [t for t in self._published if now - t <= FPS_WINDOW_S]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) >= 2 and recent[-1] > recent[0] else 0.
            latest = self._latest
            return {'running': self.running, 'subscribers': self._subscribers, 'source': self._source,
                    'error': self._error, 'fps': round(fps, 1),
                    'encode_ms': None if self._encode_ms is None else round(self._encode_ms, 2),
                    'age_ms': None if self._age_ms is None else round(self._age_ms, 1),
                    'width': None if latest is None else latest.width,
                    'height': None if latest is None else latest.height,
                    'quality': self.quality, 'max_fps': self.max_fps}

    # ------------------------------------------------------------ frames

    def wait_frame(self, after, timeout):
        """Newest frame with ``index > after``, or None on timeout or after close."""
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._index <= after and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._index <= after:
                return None
            return self._latest

    def multipart(self, boundary=BOUNDARY, keepalive_s=2.):
        """``multipart/x-mixed-replace`` body; closing the generator releases the subscription.

        Every part carries ``X-Frame-Index`` and ``X-Age-Ms`` (milliseconds between the camera frame's
        arrival at the host and this part being sent) so clients can show the server-side latency
        without sharing a clock.
        """
        marker = f'--{boundary}\r\n'.encode()
        with self.subscription():
            index = 0
            last = None
            while True:
                frame = self.wait_frame(after=index, timeout=keepalive_s)
                if frame is None:
                    if self._closed or last is None:
                        return
                    # Camera stalled: resend the last picture so the browser keeps the
                    # connection and our unsubscribe still sees its disconnect.
                    frame = last
                else:
                    index = frame.index
                    last = frame
                age_ms = (time.time() - frame.captured) * 1000.
                yield (marker + b'Content-Type: image/jpeg\r\nContent-Length: %d\r\nX-Frame-Index: %d\r\nX-Age-Ms: %.1f\r\n\r\n'
                       % (len(frame.jpeg), frame.index, age_ms) + frame.jpeg + b'\r\n')

    def snapshot(self):
        """Full-resolution JPEG of a fresh capture (``GET /api/live.jpg``), through the backend's capture lock."""
        return encode_jpeg(cv2.cvtColor(self.backend.capture().rgb, cv2.COLOR_RGB2BGR), self.quality)

    # ------------------------------------------------------------ worker

    def _run(self):
        self._last_seq = None
        while True:
            with self._condition:
                while self._subscribers == 0 and not self._closed:
                    remaining = self.idle_timeout_s - (time.monotonic() - self._idle_since)
                    if remaining <= 0:
                        self._thread = None
                        return
                    self._condition.wait(remaining)
                if self._closed:
                    self._thread = None
                    return
            began = time.monotonic()
            try:
                picture, order, source, captured, holder = self._grab()
            except Exception as exc:
                self._last_seq = None  # resync to the newest frame after a stall or a camera restart
                with self._condition:
                    self._error = f'{type(exc).__name__}: {exc}'
                    self._condition.wait(ERROR_RETRY_S)
                continue
            encode_began = time.monotonic()
            jpeg, width, height = self._encode(picture, order)
            del picture, holder
            published = time.monotonic()
            with self._condition:
                self._index += 1
                self._latest = LiveFrame(self._index, jpeg, width, height, source, time.time(), captured)
                self._source = source
                self._error = None
                self._published.append(published)
                self._encode_ms = (published - encode_began) * 1000.
                self._age_ms = (self._latest.timestamp - captured) * 1000.
                self._condition.notify_all()
                pause = 1. / (self.capture_fps if source == 'capture' else self.max_fps) - (published - began)
                if pause > 0 and self._subscribers and not self._closed:
                    self._condition.wait(pause)

    def _grab(self):
        """One full-resolution picture: ``(pixels, 'bgr'|'rgb', source, captured_epoch_s, keepalive)``."""
        backend = self.backend
        if backend.mode == 'simulation':
            return backend.capture().rgb, 'rgb', 'simulation', time.time(), None
        cameras = backend.runtime.cameras
        if cameras.streaming('head'):
            colour = cameras.wait_color('head', after_seq=self._last_seq, timeout_s=STREAM_TIMEOUT_S)
            self._last_seq = colour.seq
            return colour.bgr, 'bgr', 'stream', colour.arrival_ms / 1000., colour.keepalive
        if not backend.capture_lock.acquire(timeout=LOCK_TIMEOUT_S):
            raise TimeoutError('相机采集锁被观测占用超过 2 秒')
        try:
            frame = backend.runtime.capture('head')
        finally:
            backend.capture_lock.release()
        return frame.rgb, 'rgb', 'capture', time.time(), None

    def _encode(self, picture, order):
        bgr = scale_to_width(picture, self.width)
        if order == 'rgb':
            bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        return encode_jpeg(bgr, self.quality), bgr.shape[1], bgr.shape[0]
