"""Intel RealSense RGB-D cameras of the rig, over ``pyrealsense2``.

Each :class:`RealSenseCamera` streams colour (BGR8) and depth (Z16) from one device selected by serial
number; depth is aligned to the colour stream, so both images share the colour intrinsics. Depth is
returned in metres with invalid pixels set to NaN.

Stale-frame protection is the main job of this module. The cameras stream continuously while the arms
move for seconds between two pictures, so the frame at the head of the librealsense queue usually shows
the previous pose. Every capture therefore returns a frame that arrived at the host *after* the call
(``time_of_arrival`` metadata, host epoch milliseconds, at least one frame period later than the call),
and a picture byte-identical to the previous one is treated as stale.

A camera can optionally run a *frame pump* (:meth:`RealSenseCamera.start_stream`): one thread keeps
calling ``wait_for_frames`` and publishes the newest frameset with a sequence number. Captures then wait
on that slot instead of reading the pipeline (a pipeline must not be read from two threads), and other
consumers such as a live preview read colour frames with :meth:`RealSenseCamera.wait_color`.
"""
from __future__ import annotations

import dataclasses
import hashlib
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = [
    "CameraFrame",
    "CameraSpec",
    "ColorFrame",
    "RealSenseCamera",
    "RealSenseRig",
    "connected_serials",
]

#: Frames dropped after a pipeline (re)start while auto exposure and white balance settle.
WARMUP_FRAMES = 30
#: Upper bound on queued frames dropped while waiting for a frame that arrived after the call.
MAX_FLUSH_FRAMES = 30
#: Frames published by a pump during this time after a (re)start are discarded (exposure settling).
STREAM_WARMUP_S = 1.0


@dataclasses.dataclass(frozen=True)
class CameraFrame:
    """One RGB-D picture in world coordinates.

    Attributes:
        rgb: ``(H, W, 3)`` uint8 colour image, RGB order.
        depth_m: ``(H, W)`` float32 depth along the optical axis in metres, NaN where invalid.
        k: ``(3, 3)`` pinhole intrinsics of the colour image (depth is aligned to it).
        t_world_cam: ``(4, 4)`` camera-to-world transform in metres at the time of the picture.
        camera: Camera name in the rig.
        timestamp: Host epoch seconds at which the picture was taken.
        seq: Frame-pump sequence number, -1 when the camera is not streaming.
        arrival_ms: Colour-frame arrival at the host (epoch milliseconds), 0 when unknown.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    k: np.ndarray
    t_world_cam: np.ndarray
    camera: str = ""
    timestamp: float = 0.0
    seq: int = -1
    arrival_ms: float = 0.0

    def points_world(self) -> np.ndarray:
        """Back-project the depth image into world coordinates, ``(H, W, 3)`` with NaN where invalid."""
        h, w = self.depth_m.shape
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        z = self.depth_m.astype(np.float64)
        points = np.stack([(us - self.k[0, 2]) / self.k[0, 0] * z, (vs - self.k[1, 2]) / self.k[1, 1] * z, z], axis=-1)
        return points @ self.t_world_cam[:3, :3].T + self.t_world_cam[:3, 3]


@dataclasses.dataclass(frozen=True)
class ColorFrame:
    """A colour frame read straight from a frame pump.

    Attributes:
        bgr: ``(H, W, 3)`` uint8 BGR view into the frame buffer; valid while ``keepalive`` is referenced.
        seq: Pump sequence number.
        arrival_ms: Arrival at the host, epoch milliseconds.
        keepalive: Object that owns the pixel buffer.
    """

    bgr: np.ndarray
    seq: int
    arrival_ms: float
    keepalive: Any


@dataclasses.dataclass(frozen=True)
class CameraSpec:
    """Stream configuration of one RealSense device.

    Attributes:
        serial: Device serial number.
        width: Colour width in pixels.
        height: Colour height in pixels.
        fps: Frame rate of both streams.
        depth_width: Depth stream width before alignment.
        depth_height: Depth stream height before alignment.
        stream: Run a frame pump for this camera.
    """

    serial: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    depth_width: int = 848
    depth_height: int = 480
    stream: bool = False


@dataclasses.dataclass(frozen=True)
class _Image:
    rgb: np.ndarray
    depth_m: np.ndarray
    k: np.ndarray
    timestamp: float
    seq: int = -1
    arrival_ms: float = 0.0


def connected_serials() -> list[str]:
    """Serial numbers of the RealSense devices attached to this host."""
    import pyrealsense2 as rs

    return [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]


def _arrival_ms(frames: Any) -> float:
    """Host arrival time (epoch ms) of the colour frame of a frameset."""
    import pyrealsense2 as rs

    return float(frames.get_color_frame().get_frame_metadata(rs.frame_metadata_value.time_of_arrival))


def _signature(rgb: np.ndarray) -> str:
    return hashlib.md5(rgb.tobytes()).hexdigest()


class RealSenseCamera:
    """Colour + aligned depth from one RealSense device, with stale-frame protection.

    Args:
        spec: Device serial number and stream configuration.
    """

    def __init__(self, spec: CameraSpec) -> None:
        self.spec = spec
        self.stream_error: str | None = None
        self._pipe: Any = None
        self._depth_scale = 0.0
        self._k: np.ndarray | None = None
        self._aligners = threading.local()
        self._warm = False
        self._last_signature: str | None = None
        self._capture_lock = threading.Lock()
        self._pump_thread: threading.Thread | None = None
        self._pump_stop = threading.Event()
        self._slot_cond = threading.Condition()
        self._slot: tuple[int, float, Any] | None = None
        self._seq = 0

    @property
    def serial(self) -> str:
        return self.spec.serial

    @property
    def k(self) -> np.ndarray:
        """Colour intrinsics of the running pipeline."""
        if self._k is None:
            raise RuntimeError(f"camera {self.serial} is not open")
        return self._k

    # ------------------------------------------------------------------ lifecycle

    def open(self) -> RealSenseCamera:
        """Start the pipeline, and the frame pump when the spec asks for one."""
        self._start_pipeline()
        if self.spec.stream:
            self.start_stream()
        return self

    def close(self) -> None:
        """Stop the frame pump and the pipeline."""
        self.stop_stream()
        if self._pipe is not None:
            pipe, self._pipe = self._pipe, None
            pipe.stop()

    def _start_pipeline(self) -> None:
        import pyrealsense2 as rs

        spec = self.spec
        config = rs.config()
        config.enable_device(spec.serial)
        config.enable_stream(rs.stream.color, spec.width, spec.height, rs.format.bgr8, spec.fps)
        config.enable_stream(rs.stream.depth, spec.depth_width, spec.depth_height, rs.format.z16, spec.fps)
        pipe = rs.pipeline()
        profile = pipe.start(config)
        self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self._k = np.array([[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self._aligners = threading.local()
        self._warm = False
        self._last_signature = None
        self._pipe = pipe

    def _restart_pipeline(self) -> None:
        if self._pipe is not None:
            pipe, self._pipe = self._pipe, None
            pipe.stop()
        self._start_pipeline()

    # ------------------------------------------------------------------ frame pump

    @property
    def streaming(self) -> bool:
        """Whether the frame pump is running."""
        thread = self._pump_thread
        return thread is not None and thread.is_alive()

    def start_stream(self) -> None:
        """Start the frame pump; no-op when it already runs."""
        if self.streaming:
            return
        if self._pipe is None:
            self._start_pipeline()
        self._pump_stop.clear()
        self.stream_error = None
        with self._slot_cond:
            self._slot = None
        self._pump_thread = threading.Thread(target=self._pump, name=f"realsense-pump-{self.serial}", daemon=True)
        self._pump_thread.start()

    def stop_stream(self, timeout_s: float = 10.0) -> None:
        """Stop the frame pump and clear its slot; no-op when it is not running."""
        thread = self._pump_thread
        if thread is None:
            return
        self._pump_stop.set()
        thread.join(timeout=timeout_s)
        self._pump_thread = None
        with self._slot_cond:
            self._slot = None
            self._slot_cond.notify_all()

    def _pump(self, timeout_ms: int = 8000) -> None:
        warm_until = time.monotonic() + STREAM_WARMUP_S
        while not self._pump_stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms)
                arrival_ms = _arrival_ms(frames)
            except RuntimeError as exc:
                if self._pump_stop.is_set():
                    break
                # A stalled or re-enumerated device: rebuild the pipeline in place and settle again.
                self.stream_error = f"{type(exc).__name__}: {exc}"
                try:
                    self._restart_pipeline()
                except RuntimeError as restart_exc:
                    self.stream_error = f"pipeline restart failed: {restart_exc}"
                    self._pump_stop.wait(0.5)
                warm_until = time.monotonic() + STREAM_WARMUP_S
                continue
            if time.monotonic() < warm_until:
                continue
            with self._slot_cond:
                self._seq += 1
                self._slot = (self._seq, arrival_ms, frames)
                self._warm = True
                self._slot_cond.notify_all()

    def _wait_slot(self, after_seq: int | None, min_arrival_ms: float | None, timeout_s: float) -> tuple[int, float, Any]:
        deadline = time.monotonic() + float(timeout_s)
        with self._slot_cond:
            while True:
                slot = self._slot
                if (slot is not None and (after_seq is None or slot[0] > after_seq)
                        and (min_arrival_ms is None or slot[1] >= min_arrival_ms)):
                    return slot
                if not self.streaming:
                    raise TimeoutError(f"camera {self.serial} is not streaming ({self.stream_error or 'pump stopped'})")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"camera {self.serial}: no new frame within {timeout_s:.1f} s "
                        f"(after_seq={after_seq}, min_arrival_ms={min_arrival_ms}, error={self.stream_error})"
                    )
                self._slot_cond.wait(min(remaining, 0.5))

    def wait_color(self, after_seq: int | None, timeout_s: float) -> ColorFrame:
        """Newest pumped colour frame with ``seq > after_seq``, without depth alignment or copies.

        Raises:
            TimeoutError: The pump is not running or no such frame arrives within ``timeout_s``.
        """
        seq, arrival_ms, frames = self._wait_slot(after_seq, None, timeout_s)
        color = frames.get_color_frame()
        return ColorFrame(np.asanyarray(color.get_data()), seq, arrival_ms, color)

    def wait_stream_image(self, after_seq: int | None = None, min_arrival_ms: float | None = None,
                          timeout_s: float = 2.0) -> _Image:
        """Newest pumped frame meeting both conditions, aligned and copied in the calling thread."""
        seq, arrival_ms, frames = self._wait_slot(after_seq, min_arrival_ms, timeout_s)
        return self._materialize(frames, seq=seq, arrival_ms=arrival_ms)

    # ------------------------------------------------------------------ capture

    def capture(self, timeout_ms: int = 8000, tries: int = 3) -> _Image:
        """One colour + depth picture that arrived after this call and differs from the previous one.

        Raises:
            RuntimeError: ``tries`` attempts failed or kept returning the previous picture.
        """
        with self._capture_lock:
            if self.streaming:
                return self._capture_streaming(timeout_ms, tries)
            for _ in range(tries):
                try:
                    frames = self._fresh_frames(timeout_ms)
                except RuntimeError:
                    self._restart_pipeline()
                    continue
                image = self._materialize(frames, arrival_ms=_arrival_ms(frames))
                signature = _signature(image.rgb)
                if signature == self._last_signature:
                    self._restart_pipeline()
                    continue
                self._last_signature = signature
                return image
            raise RuntimeError(f"camera {self.serial}: {tries} captures failed or returned the previous picture")

    def _capture_streaming(self, timeout_ms: int, tries: int) -> _Image:
        threshold_ms = time.time() * 1000.0 + 1000.0 / self.spec.fps
        after_seq = None
        for _ in range(tries):
            image = self.wait_stream_image(after_seq, threshold_ms, timeout_ms / 1000.0)
            signature = _signature(image.rgb)
            if signature == self._last_signature:
                after_seq = image.seq
                continue
            self._last_signature = signature
            return image
        raise RuntimeError(f"camera {self.serial}: {tries} streamed frames repeated the previous picture")

    def _fresh_frames(self, timeout_ms: int) -> Any:
        """Drop the warm-up frames after a (re)start, otherwise the frames queued before the call."""
        if not self._warm:
            for _ in range(WARMUP_FRAMES):
                self._pipe.wait_for_frames(timeout_ms)
            self._warm = True
            return self._pipe.wait_for_frames(timeout_ms)
        threshold_ms = time.time() * 1000.0 + 1000.0 / self.spec.fps
        for _ in range(MAX_FLUSH_FRAMES):
            frames = self._pipe.wait_for_frames(timeout_ms)
            if _arrival_ms(frames) >= threshold_ms:
                return frames
        return self._pipe.wait_for_frames(timeout_ms)

    def _aligner(self) -> Any:
        """A per-thread ``rs.align`` to the colour stream (processing blocks are not shared across threads)."""
        aligner = getattr(self._aligners, "align", None)
        if aligner is None:
            import pyrealsense2 as rs

            aligner = rs.align(rs.stream.color)
            self._aligners.align = aligner
        return aligner

    def _materialize(self, frames: Any, *, arrival_ms: float, seq: int = -1) -> _Image:
        aligned = self._aligner().process(frames)
        bgr = np.asanyarray(aligned.get_color_frame().get_data())
        raw = np.asanyarray(aligned.get_depth_frame().get_data())
        depth = raw.astype(np.float32) * np.float32(self._depth_scale)
        depth[raw == 0] = np.nan
        return _Image(rgb=np.ascontiguousarray(bgr[..., ::-1]), depth_m=depth, k=self.k.copy(),
                      timestamp=arrival_ms / 1000.0, seq=seq, arrival_ms=arrival_ms)


class RealSenseRig:
    """The named RealSense cameras of the robot (``head``, ``hand_left``, ``hand_right``).

    Args:
        specs: Camera name to stream configuration, in opening order.
    """

    def __init__(self, specs: Mapping[str, CameraSpec]) -> None:
        self.cameras = {name: RealSenseCamera(spec) for name, spec in specs.items()}

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.cameras)

    def open(self) -> RealSenseRig:
        """Open every camera in order; on failure close the ones already opened and re-raise."""
        opened: list[RealSenseCamera] = []
        try:
            for camera in self.cameras.values():
                camera.open()
                opened.append(camera)
        except BaseException:
            for camera in opened:
                camera.close()
            raise
        return self

    def camera(self, name: str) -> RealSenseCamera:
        if name not in self.cameras:
            raise ValueError(f"unknown camera {name!r}; the rig has {self.names}")
        return self.cameras[name]

    def capture(self, name: str, extrinsics: Callable[[], np.ndarray]) -> CameraFrame:
        """One fresh picture of camera ``name``; ``extrinsics()`` is read right after the frame arrived."""
        image = self.camera(name).capture()
        return CameraFrame(rgb=image.rgb, depth_m=image.depth_m, k=image.k, t_world_cam=extrinsics(), camera=name,
                           timestamp=image.timestamp, seq=image.seq, arrival_ms=image.arrival_ms)

    def streaming(self, name: str) -> bool:
        """Whether camera ``name`` runs a frame pump."""
        return self.camera(name).streaming

    def wait_color(self, name: str, after_seq: int | None, timeout_s: float) -> ColorFrame:
        """Newest pumped colour frame of camera ``name`` with ``seq > after_seq``."""
        return self.camera(name).wait_color(after_seq, timeout_s)

    def close(self) -> None:
        """Close every camera; the first error is re-raised after all cameras were closed."""
        errors: list[BaseException] = []
        for camera in self.cameras.values():
            try:
                camera.close()
            except RuntimeError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
