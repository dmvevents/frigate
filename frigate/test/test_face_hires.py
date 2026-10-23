"""Tests for recognizing small faces from the recording stream."""

import unittest
from unittest.mock import MagicMock

import numpy as np
from peewee import SqliteDatabase
from pydantic import ValidationError

from frigate.config import FrigateConfig
from frigate.data_processing.common.face.detector import DetectionResult
from frigate.data_processing.common.face.hires import (
    HiResCandidate,
    HiResFaceFetcher,
    HiResOutcome,
    box_center_within,
    build_frame_command,
    find_recording,
    pad_box,
    scale_box,
    strip_hwaccel_output_format,
)
from frigate.data_processing.real_time.face import FaceRealTimeProcessor
from frigate.models import Recordings

DETECT_SHAPE = (360, 640)
RECORD_SHAPE = (1080, 1920)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ImmediateFuture:
    """Stands in for the executor future so poll() sees the result at once."""

    def __init__(self, value):
        self._value = value

    def done(self) -> bool:
        return True

    def result(self):
        return self._value


def make_fetcher(**overrides) -> HiResFaceFetcher:
    kwargs = dict(
        ffmpeg_path="ffmpeg",
        hwaccel_args_for_camera=lambda camera: [],
        find_recording_fn=lambda camera, frame_time: None,
        extract_fn=lambda *args, **kwargs: (None, "not stubbed"),
        clock=FakeClock(),
        poll_interval=0.0,
    )
    kwargs.update(overrides)
    fetcher = HiResFaceFetcher(**kwargs)
    # run the decode inline so tests are deterministic
    fetcher._executor = MagicMock()
    fetcher._executor.submit = lambda fn, *args: ImmediateFuture(fn(*args))
    return fetcher


def add_candidate(
    fetcher: HiResFaceFetcher,
    object_id: str = "obj1",
    face_px: int = 20,
    frame_time: float = 1000.0,
    camera: str = "front",
) -> bool:
    person_box = (300, 100, 380, 340)
    face_box = (330, 110, 330 + face_px, 110 + face_px)
    crop = np.zeros((face_px, face_px, 3), dtype=np.uint8)
    return fetcher.add(
        object_id, camera, frame_time, person_box, face_box, DETECT_SHAPE, crop
    )


class TestBoxMath(unittest.TestCase):
    def test_scale_box_rounds_to_record_resolution(self):
        self.assertEqual(scale_box((100, 50, 200, 150), 3.0, 3.0), (300, 150, 600, 450))
        self.assertEqual(scale_box((1, 1, 3, 3), 1.5, 2.0), (2, 2, 4, 6))

    def test_pad_box_uses_ratio_and_clamps_to_frame(self):
        self.assertEqual(
            pad_box((100, 100, 300, 500), 1920, 1080, 0.25, 0), (50, 0, 350, 600)
        )
        self.assertEqual(
            pad_box((0, 0, 1900, 1070), 1920, 1080, 0.25, 0), (0, 0, 1920, 1080)
        )

    def test_pad_box_has_a_minimum_pad_for_small_boxes(self):
        self.assertEqual(
            pad_box((100, 100, 110, 120), 1920, 1080, 0.1, 16), (84, 84, 126, 136)
        )

    def test_box_center_within_allows_slack(self):
        bounds = (100, 100, 300, 500)
        self.assertTrue(box_center_within((150, 120, 200, 170), bounds))
        # center at x=320 is 20px outside, slack is 15% of 200px = 30px
        self.assertTrue(box_center_within((300, 120, 340, 170), bounds))
        self.assertFalse(box_center_within((330, 120, 400, 170), bounds))
        self.assertFalse(box_center_within((150, 600, 200, 650), bounds))


class TestFfmpegCommand(unittest.TestCase):
    def test_strip_hwaccel_output_format_removes_only_that_pair(self):
        args = [
            "-hwaccel_device",
            "0",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
        ]
        self.assertEqual(
            strip_hwaccel_output_format(args),
            ["-hwaccel_device", "0", "-hwaccel", "cuda"],
        )
        self.assertEqual(strip_hwaccel_output_format([]), [])

    def test_build_frame_command_seeks_before_input(self):
        cmd = build_frame_command("/usr/lib/ffmpeg/7.0/bin/ffmpeg", "/media/a.mp4", 4.2)
        self.assertEqual(cmd[0], "/usr/lib/ffmpeg/7.0/bin/ffmpeg")
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))
        self.assertEqual(cmd[cmd.index("-ss") + 1], "4.200")
        self.assertEqual(cmd[cmd.index("-i") + 1], "/media/a.mp4")
        self.assertEqual(cmd[-1], "-")
        self.assertIn("bmp", cmd)

    def test_build_frame_command_places_hwaccel_before_seek(self):
        cmd = build_frame_command(
            "ffmpeg",
            "/media/a.mp4",
            -1.0,
            ["-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi"],
        )
        self.assertLess(cmd.index("-hwaccel"), cmd.index("-ss"))
        self.assertNotIn("-hwaccel_output_format", cmd)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "0.000")


class TestFindRecording(unittest.TestCase):
    def setUp(self):
        self.db = SqliteDatabase(":memory:")
        self.db.bind([Recordings])
        self.db.connect()
        self.db.create_tables([Recordings])
        rows = [
            ("a", "front", "/media/front/00.00.mp4", 100.0, 110.0, "main"),
            ("b", "front", "/media/front/00.10.mp4", 110.0, 120.0, "main"),
            ("c", "front", "/media/front/sub/00.10.mp4", 110.0, 120.0, "sub"),
            ("d", "back", "/media/back/00.10.mp4", 110.0, 120.0, "main"),
        ]
        for id, camera, path, start, end, stream_type in rows:
            Recordings.create(
                id=id,
                camera=camera,
                path=path,
                start_time=start,
                end_time=end,
                duration=end - start,
                stream_type=stream_type,
            )

    def tearDown(self):
        self.db.drop_tables([Recordings])
        self.db.close()

    def test_returns_main_segment_and_offset(self):
        self.assertEqual(
            find_recording("front", 114.5), ("/media/front/00.10.mp4", 4.5)
        )
        self.assertEqual(
            find_recording("front", 105.0), ("/media/front/00.00.mp4", 5.0)
        )

    def test_ignores_other_cameras_and_unflushed_times(self):
        self.assertEqual(find_recording("back", 114.5), ("/media/back/00.10.mp4", 4.5))
        self.assertIsNone(find_recording("front", 125.0))
        self.assertIsNone(find_recording("garage", 114.5))


class TestHiResFaceFetcherQueue(unittest.TestCase):
    def test_keeps_the_largest_candidates_per_object(self):
        fetcher = make_fetcher(max_candidates=2)
        self.assertTrue(add_candidate(fetcher, face_px=20))
        self.assertTrue(add_candidate(fetcher, face_px=24))
        # smaller than both waiting faces, not worth a decode
        self.assertFalse(add_candidate(fetcher, face_px=18))
        # larger replaces the smallest
        self.assertTrue(add_candidate(fetcher, face_px=30))
        self.assertEqual(fetcher.pending_count("obj1"), 2)
        sizes = sorted(c.detect_face_px for c in fetcher.objects["obj1"].candidates)
        self.assertEqual(sizes, [24, 30])

    def test_rejects_cameras_without_gain_and_ended_objects(self):
        fetcher = make_fetcher()
        fetcher.no_gain_cameras.add("front")
        self.assertFalse(add_candidate(fetcher))
        self.assertEqual(fetcher.pending_count(), 0)

        fetcher = make_fetcher()
        self.assertTrue(add_candidate(fetcher))
        self.assertFalse(fetcher.expire("obj1"))
        self.assertFalse(add_candidate(fetcher, face_px=40))
        self.assertEqual(fetcher.pending_count("obj1"), 1)

    def test_expire_without_pending_work_forgets_the_object(self):
        fetcher = make_fetcher()
        self.assertTrue(fetcher.expire("never-seen"))
        add_candidate(fetcher)
        fetcher.objects["obj1"].candidates.clear()
        self.assertTrue(fetcher.expire("obj1"))
        self.assertNotIn("obj1", fetcher.objects)

    def test_waits_for_the_segment_then_gives_up(self):
        clock = FakeClock()
        lookups: list[tuple[str, float]] = []

        def find(camera, frame_time):
            lookups.append((camera, frame_time))
            return None

        fetcher = make_fetcher(
            clock=clock, find_recording_fn=find, segment_wait=30.0, poll_interval=1.0
        )
        add_candidate(fetcher, frame_time=1000.0)

        self.assertEqual(fetcher.poll(), [])
        self.assertEqual(lookups, [("front", 1000.0)])

        # inside the poll interval nothing is looked up again
        clock.advance(0.5)
        self.assertEqual(fetcher.poll(), [])
        self.assertEqual(len(lookups), 1)

        clock.advance(10)
        self.assertEqual(fetcher.poll(), [])
        self.assertEqual(len(lookups), 2)

        clock.advance(25)
        outcomes = fetcher.poll()
        self.assertEqual(len(outcomes), 1)
        self.assertIsNone(outcomes[0].region)
        self.assertEqual(outcomes[0].reason, "no recording")
        self.assertEqual(outcomes[0].candidate.detect_crop.shape, (20, 20, 3))
        self.assertEqual(fetcher.pending_count(), 0)

    def test_returns_scaled_region_from_the_recording_frame(self):
        frame = np.zeros((*RECORD_SHAPE, 3), dtype=np.uint8)
        frame[:, :, 1] = 200
        calls: list[tuple] = []

        def extract(ffmpeg_path, path, seek, hwaccel):
            calls.append((ffmpeg_path, path, seek, hwaccel))
            return frame, ""

        fetcher = make_fetcher(
            find_recording_fn=lambda c, t: ("/media/front/00.10.mp4", 3.25),
            extract_fn=extract,
        )
        add_candidate(fetcher)
        # first poll submits the decode, the inline future is read on the next
        self.assertEqual(fetcher.poll(), [])
        outcomes = fetcher.poll()

        self.assertEqual(calls, [("ffmpeg", "/media/front/00.10.mp4", 3.25, None)])
        self.assertEqual(len(outcomes), 1)
        outcome = outcomes[0]
        self.assertEqual(outcome.scale, (3.0, 3.0))
        # person box (300,100,380,340) scaled x3 is (900,300,1140,1020),
        # padded by 35% of 240x720 = 84x252 and clamped to the frame
        self.assertEqual(outcome.region_origin, (816, 48))
        self.assertEqual(outcome.region.shape, (1032, 408, 3))
        self.assertEqual(outcome.expected_face, (990, 330, 1050, 390))
        self.assertEqual(int(outcome.region[0, 0, 1]), 200)
        self.assertEqual(fetcher.pending_count(), 0)

    def test_record_stream_not_larger_disables_camera(self):
        small = np.zeros((360, 640, 3), dtype=np.uint8)
        fetcher = make_fetcher(
            find_recording_fn=lambda c, t: ("/media/front/00.10.mp4", 1.0),
            extract_fn=lambda *a: (small, ""),
        )
        add_candidate(fetcher)
        fetcher.poll()
        outcomes = fetcher.poll()
        self.assertIsNone(outcomes[0].region)
        self.assertIn("front", fetcher.no_gain_cameras)
        self.assertFalse(add_candidate(fetcher))

    def test_decode_failure_falls_back_to_detect_crop(self):
        fetcher = make_fetcher(
            find_recording_fn=lambda c, t: ("/media/front/00.10.mp4", 1.0),
            extract_fn=lambda *a: (None, "timeout"),
        )
        add_candidate(fetcher)
        fetcher.poll()
        outcomes = fetcher.poll()
        self.assertIsNone(outcomes[0].region)
        self.assertEqual(outcomes[0].reason, "timeout")

    def test_hwaccel_failure_switches_camera_to_software_decode(self):
        frame = np.zeros((*RECORD_SHAPE, 3), dtype=np.uint8)
        calls: list[list[str] | None] = []

        def extract(ffmpeg_path, path, seek, hwaccel):
            calls.append(hwaccel)
            if hwaccel:
                return None, "no cuda"
            return frame, ""

        fetcher = make_fetcher(
            hwaccel_args_for_camera=lambda c: ["-hwaccel", "cuda"],
            find_recording_fn=lambda c, t: ("/media/front/00.10.mp4", 1.0),
            extract_fn=extract,
        )
        add_candidate(fetcher)
        fetcher.poll()
        outcomes = fetcher.poll()
        self.assertIsNotNone(outcomes[0].region)
        self.assertEqual(calls, [["-hwaccel", "cuda"], None])
        self.assertIn("front", fetcher.software_decode_cameras)

        add_candidate(fetcher, object_id="obj2")
        fetcher.poll()
        fetcher.poll()
        self.assertEqual(calls[2:], [None])

    def test_ended_object_is_forgotten_once_drained(self):
        fetcher = make_fetcher(
            find_recording_fn=lambda c, t: ("/media/front/00.10.mp4", 1.0),
            extract_fn=lambda *a: (np.zeros((*RECORD_SHAPE, 3), dtype=np.uint8), ""),
        )
        add_candidate(fetcher)
        self.assertFalse(fetcher.expire("obj1"))
        fetcher.poll()
        self.assertEqual(len(fetcher.poll()), 1)
        self.assertNotIn("obj1", fetcher.objects)


def make_processor(hires_enabled: bool = True, min_px: int = 80, records: bool = True):
    """A FaceRealTimeProcessor without models, enough for the gating and finish paths."""
    processor = FaceRealTimeProcessor.__new__(FaceRealTimeProcessor)
    camera = MagicMock()
    camera.record.enabled = records
    record_input = MagicMock()
    record_input.roles = ["detect", "record"] if records else ["detect"]
    record_input.hwaccel_args = []
    camera.ffmpeg.inputs = [record_input]
    camera.ffmpeg.hwaccel_args = []
    camera.face_recognition.min_area = 0

    config = MagicMock()
    config.cameras = {"front": camera}
    config.ffmpeg.ffmpeg_path = "ffmpeg"
    config.face_recognition.save_attempts = 0
    config.face_recognition.hires_from_record = hires_enabled
    config.face_recognition.min_face_px_for_detect_crop = min_px
    config.face_recognition.detection_threshold = 0.7
    config.face_recognition.unknown_score = 0.8
    config.face_recognition.recognition_threshold = 0.9
    config.face_recognition.min_faces = 1

    processor.config = config
    processor.face_config = config.face_recognition
    processor.metrics = MagicMock()
    processor.requestor = MagicMock()
    processor.sub_label_publisher = MagicMock()
    processor.face_detector = MagicMock()
    processor.recognizer = MagicMock()
    processor.recognizer.classify.return_value = ("alice", 0.95)
    processor.faces_per_second = MagicMock()
    processor.inference_speed = MagicMock()
    processor.person_face_history = {}
    processor.camera_current_people = {}
    processor.pending_expirations = {}
    processor.hires_skipped_cameras = set()
    processor.hires = None
    return processor


def obj_data(face_box=None) -> dict:
    return {
        "id": "1700000000.1-abcd",
        "camera": "front",
        "label": "person",
        "frame_time": 1700000000.5,
        "box": (300, 100, 380, 340),
    }


class TestFaceProcessorHiResGating(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros(
            (DETECT_SHAPE[0] * 3 // 2, DETECT_SHAPE[1]), dtype=np.uint8
        )

    def queue(self, processor, face_box):
        crop = np.zeros(
            (face_box[3] - face_box[1], face_box[2] - face_box[0], 3), dtype=np.uint8
        )
        return processor._FaceRealTimeProcessor__queue_hires(
            obj_data(), self.frame, face_box, crop
        )

    def test_disabled_never_queues(self):
        processor = make_processor(hires_enabled=False)
        self.assertFalse(self.queue(processor, (330, 110, 350, 130)))
        self.assertIsNone(processor.hires)

    def test_large_faces_use_the_detect_crop(self):
        processor = make_processor(min_px=80)
        self.assertFalse(self.queue(processor, (330, 110, 410, 190)))
        self.assertIsNone(processor.hires)

    def test_small_faces_are_queued_with_frame_geometry(self):
        processor = make_processor(min_px=80)
        self.assertTrue(self.queue(processor, (330, 110, 350, 130)))
        self.assertIsNotNone(processor.hires)
        candidate = processor.hires.objects[obj_data()["id"]].candidates[0]
        self.assertEqual(candidate.detect_shape, DETECT_SHAPE)
        self.assertEqual(candidate.frame_time, 1700000000.5)
        self.assertEqual(candidate.person_box, (300, 100, 380, 340))
        self.assertEqual(candidate.detect_crop.shape, (20, 20, 3))

    def test_camera_without_recording_uses_the_detect_crop(self):
        processor = make_processor(records=False)
        self.assertFalse(self.queue(processor, (330, 110, 350, 130)))

    def test_expire_waits_for_pending_faces(self):
        processor = make_processor()
        processor.person_face_history["obj1"] = [("alice", 0.95, 400)]
        processor.camera_current_people["front"] = ["obj1"]
        self.assertTrue(self.queue(processor, (330, 110, 350, 130)))
        object_id = obj_data()["id"]
        processor.person_face_history[object_id] = []

        processor.expire_object(object_id, "front")
        self.assertIn(object_id, processor.person_face_history)
        self.assertEqual(processor.pending_expirations, {object_id: "front"})

        processor.expire_object("obj1", "front")
        self.assertNotIn("obj1", processor.person_face_history)


class TestFaceProcessorHiResFinish(unittest.TestCase):
    def candidate(self) -> HiResCandidate:
        return HiResCandidate(
            object_id="obj1",
            camera="front",
            frame_time=1700000000.5,
            person_box=(300, 100, 380, 340),
            face_box=(330, 110, 350, 130),
            detect_shape=DETECT_SHAPE,
            detect_crop=np.full((20, 20, 3), 7, dtype=np.uint8),
            created=0.0,
        )

    def finish(self, processor, outcome):
        processor._FaceRealTimeProcessor__finish_hires(outcome)

    def test_recognizes_the_hires_face_and_publishes(self):
        processor = make_processor()
        region = np.full((1032, 408, 3), 9, dtype=np.uint8)
        # face found by the detector inside the region, 90px tall
        processor.face_detector.detect.return_value = DetectionResult(
            face=(160, 270, 250, 360), landmarks=()
        )
        outcome = HiResOutcome(
            self.candidate(),
            region,
            region_origin=(816, 48),
            expected_face=(990, 330, 1050, 390),
            scale=(3.0, 3.0),
        )

        self.finish(processor, outcome)

        crop = processor.recognizer.classify.call_args[0][0]
        self.assertEqual(crop.shape, (90, 90, 3))
        self.assertEqual(int(crop[0, 0, 0]), 9)
        self.assertEqual(processor.person_face_history["obj1"], [("alice", 0.95, 8100)])
        processor.requestor.send_data.assert_called_once()
        processor.sub_label_publisher.publish.assert_called_once()
        self.assertEqual(
            processor.sub_label_publisher.publish.call_args[0][0],
            ("obj1", "alice", 0.95),
        )

    def test_face_outside_person_box_falls_back_to_detect_crop(self):
        processor = make_processor()
        region = np.full((1032, 408, 3), 9, dtype=np.uint8)
        # a face near the far edge of the padded region: another person
        processor.face_detector.detect.return_value = DetectionResult(
            face=(0, 900, 60, 960), landmarks=()
        )
        outcome = HiResOutcome(
            self.candidate(), region, region_origin=(816, 48), scale=(3.0, 3.0)
        )

        self.finish(processor, outcome)

        crop = processor.recognizer.classify.call_args[0][0]
        self.assertEqual(crop.shape, (20, 20, 3))
        self.assertEqual(int(crop[0, 0, 0]), 7)

    def test_no_face_or_no_frame_falls_back_to_detect_crop(self):
        for outcome_kwargs in ({"region": None, "reason": "no recording"},):
            processor = make_processor()
            self.finish(processor, HiResOutcome(self.candidate(), **outcome_kwargs))
            crop = processor.recognizer.classify.call_args[0][0]
            self.assertEqual(crop.shape, (20, 20, 3))

        processor = make_processor()
        processor.face_detector.detect.return_value = None
        self.finish(
            processor,
            HiResOutcome(
                self.candidate(),
                np.zeros((1032, 408, 3), dtype=np.uint8),
                region_origin=(816, 48),
                scale=(3.0, 3.0),
            ),
        )
        crop = processor.recognizer.classify.call_args[0][0]
        self.assertEqual(crop.shape, (20, 20, 3))
        self.assertEqual(processor.person_face_history["obj1"][0][2], 400)


class TestFaceHiResConfig(unittest.TestCase):
    def setUp(self):
        self.minimal = {
            "mqtt": {"host": "mqtt"},
            "cameras": {
                "back": {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://10.0.0.1:554/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": {"height": 1080, "width": 1920, "fps": 5},
                }
            },
        }

    def test_defaults_are_off(self):
        config = FrigateConfig(**self.minimal)
        self.assertFalse(config.face_recognition.hires_from_record)
        self.assertEqual(config.face_recognition.min_face_px_for_detect_crop, 80)

    def test_values_are_parsed(self):
        self.minimal["face_recognition"] = {
            "hires_from_record": True,
            "min_face_px_for_detect_crop": 64,
        }
        config = FrigateConfig(**self.minimal)
        self.assertTrue(config.face_recognition.hires_from_record)
        self.assertEqual(config.face_recognition.min_face_px_for_detect_crop, 64)

    def test_negative_size_is_rejected(self):
        self.minimal["face_recognition"] = {"min_face_px_for_detect_crop": -1}
        with self.assertRaises(ValidationError):
            FrigateConfig(**self.minimal)


if __name__ == "__main__":
    unittest.main(verbosity=2)
