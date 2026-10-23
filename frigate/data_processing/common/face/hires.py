"""Recognize small faces from the camera's high resolution recording stream.

Face recognition runs on the detect stream. On a low resolution detect
stream a distant face can be a dozen pixels tall, far below what the
embedding models were trained on, and no amount of upscaling brings the
detail back. The same frame exists at full resolution in the camera's
record stream, which Frigate copies to disk without decoding it.

This module waits for the recording segment that contains a frame, decodes
that single frame, finds the face again inside the person's box at full
resolution and hands the crop back to the recognizer. Recordings are
written in segments, so the result arrives seconds after the detect frame
and is applied to the tracked object (and its event) when it is ready.

The detect and record streams are not guaranteed to be in sync, so the face
is re-detected inside a padded copy of the person's box instead of trusting
the scaled detect-frame coordinates, and the detect crop is kept so the
regular path can run if the recording never shows up or has no face in it.
"""

import logging
import subprocess as sp
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from peewee import DoesNotExist

from frigate.models import Recordings
from frigate.util.builtin import get_ffmpeg_arg_list

logger = logging.getLogger(__name__)

# recordings are written in segments (10 s by default) and inserted in the
# database once the segment is closed, so a frame is on disk at most about
# one segment plus a little after it was seen. give up after this.
SEGMENT_WAIT_SECONDS = 30.0

# how often pending candidates look for their segment in the database
POLL_INTERVAL_SECONDS = 1.0

# the record stream may lag or lead the detect stream, so look for the face
# in a padded copy of the person's box rather than at its exact position
PERSON_BOX_PAD_RATIO = 0.35
PERSON_BOX_MIN_PAD = 16

# the re-detected face must sit inside the scaled person box widened by this
# ratio, otherwise it is probably a different person
FACE_MATCH_SLACK_RATIO = 0.15

# small faces are seen on many detect frames while the object is tracked.
# keep only the largest few of them per object; more add cost, not detail.
MAX_CANDIDATES_PER_OBJECT = 3

FFMPEG_TIMEOUT_SECONDS = 10

STREAM_TYPE_MAIN = "main"

Box = tuple[int, int, int, int]


@dataclass
class HiResCandidate:
    """A small face seen on the detect stream, waiting for its recording."""

    object_id: str
    camera: str
    frame_time: float
    person_box: Box
    face_box: Box
    detect_shape: tuple[int, int]
    detect_crop: np.ndarray
    created: float
    last_poll: float = 0.0
    future: Future | None = None
    recording_path: str | None = None
    seek: float = 0.0

    @property
    def detect_face_px(self) -> int:
        """Shorter side of the face on the detect stream, in pixels."""
        return min(
            self.face_box[2] - self.face_box[0], self.face_box[3] - self.face_box[1]
        )


@dataclass
class HiResOutcome:
    """What the fetcher produced for a candidate."""

    candidate: HiResCandidate
    # full resolution region around the person, or None when it could not be
    # fetched and the detect crop should be used instead
    region: np.ndarray | None
    # offset of region inside the full frame, and the expected face position
    # inside region, both in record stream pixels
    region_origin: tuple[int, int] = (0, 0)
    expected_face: Box | None = None
    scale: tuple[float, float] = (1.0, 1.0)
    reason: str = ""


@dataclass
class _ObjectState:
    candidates: list[HiResCandidate] = field(default_factory=list)
    ended: bool = False


def scale_box(box: Box, sx: float, sy: float) -> Box:
    """Scale a box from one resolution to another."""
    return (
        int(round(box[0] * sx)),
        int(round(box[1] * sy)),
        int(round(box[2] * sx)),
        int(round(box[3] * sy)),
    )


def pad_box(
    box: Box,
    width: int,
    height: int,
    ratio: float = PERSON_BOX_PAD_RATIO,
    min_pad: int = PERSON_BOX_MIN_PAD,
) -> Box:
    """Widen a box by a ratio of its size on every side, clamped to the frame."""
    pad_x = max(min_pad, round((box[2] - box[0]) * ratio))
    pad_y = max(min_pad, round((box[3] - box[1]) * ratio))
    return (
        max(0, box[0] - pad_x),
        max(0, box[1] - pad_y),
        min(width, box[2] + pad_x),
        min(height, box[3] + pad_y),
    )


def box_center_within(
    box: Box, bounds: Box, slack: float = FACE_MATCH_SLACK_RATIO
) -> bool:
    """True when the center of box lies inside bounds widened by slack."""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    sx = (bounds[2] - bounds[0]) * slack
    sy = (bounds[3] - bounds[1]) * slack
    return (
        bounds[0] - sx <= cx <= bounds[2] + sx
        and bounds[1] - sy <= cy <= bounds[3] + sy
    )


def strip_hwaccel_output_format(args: list[str]) -> list[str]:
    """Drop -hwaccel_output_format so decoded frames land in system memory.

    The decode presets keep frames on the device for the scale filter that
    follows them in the detect pipeline. A single frame written to a pipe has
    no such filter, so letting ffmpeg download it is both simpler and works
    for every hwaccel type.
    """
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg == "-hwaccel_output_format":
            skip = True
            continue
        out.append(arg)
    return out


def build_frame_command(
    ffmpeg_path: str,
    recording_path: str,
    seek: float,
    hwaccel_args: list[str] | None = None,
) -> list[str]:
    """ffmpeg command that writes one frame at seek seconds as a bmp to stdout."""
    cmd = [ffmpeg_path, "-hide_banner", "-loglevel", "warning", "-nostdin"]

    if hwaccel_args:
        cmd.extend(strip_hwaccel_output_format(hwaccel_args))

    cmd.extend(
        [
            "-ss",
            f"{max(0.0, seek):.3f}",
            "-i",
            recording_path,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "bmp",
            "-",
        ]
    )
    return cmd


def extract_frame(
    ffmpeg_path: str,
    recording_path: str,
    seek: float,
    hwaccel_args: list[str] | None,
    timeout: float = FFMPEG_TIMEOUT_SECONDS,
) -> tuple[np.ndarray | None, str]:
    """Decode a single BGR frame from a recording.

    Returns the frame and an empty string, or None and the reason.
    """
    cmd = build_frame_command(ffmpeg_path, recording_path, seek, hwaccel_args)

    try:
        process = sp.run(cmd, capture_output=True, timeout=timeout)
    except sp.TimeoutExpired:
        return None, "timeout"
    except OSError as e:
        return None, str(e)

    if process.returncode != 0 or not process.stdout:
        stderr = (
            process.stderr.decode(errors="replace").strip() if process.stderr else ""
        )
        return None, stderr or f"ffmpeg exited {process.returncode}"

    frame = cv2.imdecode(
        np.frombuffer(process.stdout, dtype=np.uint8), cv2.IMREAD_COLOR
    )

    if frame is None:
        return None, "undecodable frame"

    return frame, ""


def find_recording(camera: str, frame_time: float) -> tuple[str, float] | None:
    """Path of the main stream segment holding frame_time and the offset into it."""
    try:
        recording: Recordings = (
            Recordings.select(Recordings.path, Recordings.start_time)
            .where(
                (Recordings.camera == camera)
                & (Recordings.stream_type == STREAM_TYPE_MAIN)
                & (Recordings.start_time <= frame_time)
                & (Recordings.end_time >= frame_time)
            )
            .order_by(Recordings.start_time.desc())
            .limit(1)
            .get()
        )
    except DoesNotExist:
        return None

    return recording.path, frame_time - recording.start_time


class HiResFaceFetcher:
    """Queue of small faces waiting for their recording segment.

    process_frame adds candidates; the embeddings maintainer drains outcomes
    once per loop. Database lookups and all inference stay on the caller's
    thread, only the ffmpeg decode runs on a single worker so a slow disk or
    decoder never blocks the real time path.
    """

    def __init__(
        self,
        ffmpeg_path: str,
        hwaccel_args_for_camera: Callable[[str], list[str]],
        find_recording_fn: Callable[
            [str, float], tuple[str, float] | None
        ] = find_recording,
        extract_fn: Callable[..., tuple[np.ndarray | None, str]] = extract_frame,
        clock: Callable[[], float] = time.time,
        segment_wait: float = SEGMENT_WAIT_SECONDS,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        max_candidates: int = MAX_CANDIDATES_PER_OBJECT,
    ) -> None:
        self.ffmpeg_path = ffmpeg_path
        self.hwaccel_args_for_camera = hwaccel_args_for_camera
        self.find_recording = find_recording_fn
        self.extract = extract_fn
        self.clock = clock
        self.segment_wait = segment_wait
        self.poll_interval = poll_interval
        self.max_candidates = max_candidates
        self.objects: dict[str, _ObjectState] = {}
        # cameras whose record stream is not larger than detect, or whose
        # hwaccel decode failed, learned at runtime
        self.no_gain_cameras: set[str] = set()
        self.software_decode_cameras: set[str] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="face_hires_ffmpeg"
        )
        self._lock = threading.Lock()

    def add(
        self,
        object_id: str,
        camera: str,
        frame_time: float,
        person_box: Box,
        face_box: Box,
        detect_shape: tuple[int, int],
        detect_crop: np.ndarray,
    ) -> bool:
        """Remember a small face for later. Returns False when it was not kept."""
        if camera in self.no_gain_cameras:
            return False

        state = self.objects.setdefault(object_id, _ObjectState())

        if state.ended:
            return False

        candidate = HiResCandidate(
            object_id=object_id,
            camera=camera,
            frame_time=frame_time,
            person_box=person_box,
            face_box=face_box,
            detect_shape=detect_shape,
            detect_crop=detect_crop.copy(),
            created=self.clock(),
        )

        waiting = [c for c in state.candidates if c.future is None]

        if len(waiting) >= self.max_candidates:
            smallest = min(waiting, key=lambda c: c.detect_face_px)

            if candidate.detect_face_px <= smallest.detect_face_px:
                return False

            state.candidates.remove(smallest)

        state.candidates.append(candidate)
        return True

    def pending_count(self, object_id: str | None = None) -> int:
        if object_id is not None:
            state = self.objects.get(object_id)
            return len(state.candidates) if state else 0

        return sum(len(s.candidates) for s in self.objects.values())

    def expire(self, object_id: str) -> bool:
        """Mark an object as ended. Returns True when nothing is left for it."""
        state = self.objects.get(object_id)

        if state is None:
            return True

        if not state.candidates:
            self.objects.pop(object_id, None)
            return True

        state.ended = True
        return False

    def poll(self) -> list[HiResOutcome]:
        """Advance pending candidates and return the ones that are finished."""
        now = self.clock()
        outcomes: list[HiResOutcome] = []

        for object_id in list(self.objects):
            state = self.objects[object_id]

            for candidate in list(state.candidates):
                outcome = self._advance(candidate, now)

                if outcome is not None:
                    state.candidates.remove(candidate)
                    outcomes.append(outcome)

            if state.ended and not state.candidates:
                self.objects.pop(object_id, None)

        return outcomes

    def _advance(self, candidate: HiResCandidate, now: float) -> HiResOutcome | None:
        if candidate.future is not None:
            if not candidate.future.done():
                return None

            frame, reason = candidate.future.result()

            if frame is None:
                return HiResOutcome(candidate, None, reason=reason)

            return self._outcome_from_frame(candidate, frame)

        if now - candidate.last_poll < self.poll_interval:
            return None

        candidate.last_poll = now
        found = self.find_recording(candidate.camera, candidate.frame_time)

        if found is None:
            if now - candidate.created > self.segment_wait:
                return HiResOutcome(candidate, None, reason="no recording")

            return None

        candidate.recording_path, candidate.seek = found
        candidate.future = self._executor.submit(self._fetch, candidate)
        return None

    def _fetch(self, candidate: HiResCandidate) -> tuple[np.ndarray | None, str]:
        """Worker side: decode the frame, trying hwaccel first then software."""
        assert candidate.recording_path is not None
        hwaccel = self.hwaccel_args_for_camera(candidate.camera)
        started = time.monotonic()

        if hwaccel and candidate.camera not in self.software_decode_cameras:
            frame, reason = self.extract(
                self.ffmpeg_path, candidate.recording_path, candidate.seek, hwaccel
            )

            if frame is not None:
                logger.debug(
                    f"{candidate.camera}: decoded recording frame with hwaccel in {(time.monotonic() - started) * 1000:.0f}ms"
                )
                return frame, ""

            logger.debug(
                f"{candidate.camera}: hwaccel decode of {candidate.recording_path} failed ({reason}), using software decode from now on"
            )
            with self._lock:
                self.software_decode_cameras.add(candidate.camera)

        frame, reason = self.extract(
            self.ffmpeg_path, candidate.recording_path, candidate.seek, None
        )

        if frame is not None:
            logger.debug(
                f"{candidate.camera}: decoded recording frame in software in {(time.monotonic() - started) * 1000:.0f}ms"
            )

        return frame, reason

    def _outcome_from_frame(
        self, candidate: HiResCandidate, frame: np.ndarray
    ) -> HiResOutcome:
        rec_h, rec_w = frame.shape[:2]
        det_h, det_w = candidate.detect_shape

        if rec_w <= det_w and rec_h <= det_h:
            logger.debug(
                f"{candidate.camera}: record stream {rec_w}x{rec_h} is not larger than detect {det_w}x{det_h}, not using it for faces"
            )
            self.no_gain_cameras.add(candidate.camera)
            return HiResOutcome(candidate, None, reason="record stream not larger")

        sx = rec_w / det_w
        sy = rec_h / det_h
        person = scale_box(candidate.person_box, sx, sy)
        region_box = pad_box(person, rec_w, rec_h)
        region = frame[region_box[1] : region_box[3], region_box[0] : region_box[2]]

        if region.size == 0:
            return HiResOutcome(candidate, None, reason="empty region")

        expected = scale_box(candidate.face_box, sx, sy)
        return HiResOutcome(
            candidate,
            region,
            region_origin=(region_box[0], region_box[1]),
            expected_face=expected,
            scale=(sx, sy),
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.objects.clear()


def hwaccel_args_for(camera_config: Any) -> list[str]:
    """Decode arguments the camera's record input (or camera) is configured with.

    Presets are resolved the same way the detect pipeline resolves them; the
    detect geometry only matters to presets that scale in the decoder.
    """
    from frigate.ffmpeg_presets import parse_preset_hardware_acceleration_decode

    record_input = next(
        (
            i
            for i in camera_config.ffmpeg.inputs
            if "record" in [str(r.value) if hasattr(r, "value") else r for r in i.roles]
        ),
        None,
    )
    candidates: list[Any] = []

    if record_input is not None and record_input.hwaccel_args:
        candidates.append(record_input.hwaccel_args)

    if (
        camera_config.ffmpeg.hwaccel_args
        and camera_config.ffmpeg.hwaccel_args != "auto"
    ):
        candidates.append(camera_config.ffmpeg.hwaccel_args)

    for arg in candidates:
        preset = parse_preset_hardware_acceleration_decode(
            arg,
            camera_config.detect.fps,
            camera_config.detect.width,
            camera_config.detect.height,
            camera_config.ffmpeg.gpu,
        )
        resolved = preset or arg

        if resolved:
            return get_ffmpeg_arg_list(resolved)

    return []


def camera_records_main_stream(camera_config: Any) -> bool:
    """True when the camera has a record role and recording is enabled."""
    if not camera_config.record.enabled:
        return False

    return any(
        "record" in [str(r.value) if hasattr(r, "value") else r for r in i.roles]
        for i in camera_config.ffmpeg.inputs
    )
