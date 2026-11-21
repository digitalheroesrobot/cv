#!/usr/bin/env python3
"""Deterministic offline tests for the threaded multi-camera capture manager."""
import unittest
import numpy as np
from common import CameraCalibration, CameraFrame, DepthMap
from camera_adapters import CameraSample
from multi_camera.capture import MultiCameraCaptureManager
from multi_camera.time_sync import ClockModel


class DummyAdapter:
    def __init__(self, camera_id, timestamp):
        self.camera_id, self.timestamp = camera_id, timestamp
        self.opened = self.closed = False
    def open(self): self.opened = True
    def read(self): return None
    def close(self): self.closed = True


def sample(camera_id, timestamp, frame_id=0):
    image = np.zeros((4, 5, 3), np.uint8)
    depth = np.full((4, 5), 1.0, np.float32)
    calib = CameraCalibration(np.array([[10.,0,2.],[0,10.,2.],[0,0,1.]]),
                              5, 4, np.zeros(5))
    return CameraSample(
        CameraFrame(image, timestamp, camera_id, frame_id),
        calib, DepthMap(depth, timestamp, camera_id))


class TestMultiCameraCapture(unittest.TestCase):
    def setUp(self):
        self.adapters = {"a": DummyAdapter("a", 1.0),
                         "b": DummyAdapter("b", 1.1)}
        self.manager = MultiCameraCaptureManager(
            self.adapters,
            {"a": ClockModel(), "b": ClockModel(offset_s=-.1)},
            tolerance_s=.002, queue_size=2)

    def test_normalizes_frame_and_depth_to_master_time(self):
        normalized = self.manager._normalize_sample("b", sample("b", 1.1))
        self.assertAlmostEqual(normalized.frame.timestamp, 1.0)
        self.assertAlmostEqual(normalized.depth.timestamp, 1.0)
        self.assertAlmostEqual(
            normalized.frame.metadata["device_timestamp_s"], 1.1)

    def test_bundle_pairing(self):
        self.manager.queues["a"].append(sample("a", 1.0))
        self.manager.queues["b"].append(sample("b", 1.001))
        with self.manager._condition:
            bundle = self.manager._try_bundle_locked()
        self.assertIsNotNone(bundle)
        self.assertLessEqual(bundle.max_skew_s, .002)
        self.assertEqual(set(bundle.samples), {"a", "b"})

    def test_stale_frame_dropped(self):
        self.manager.queues["a"].append(sample("a", 1.0))
        self.manager.queues["b"].append(sample("b", 1.02))
        with self.manager._condition:
            bundle = self.manager._try_bundle_locked()
        self.assertIsNone(bundle)
        self.assertEqual(self.manager.stats["a"]["stale"], 1)

    def test_lifecycle_opens_and_closes_adapters(self):
        self.manager.start()
        self.assertTrue(all(a.opened for a in self.adapters.values()))
        self.manager.close()
        self.assertTrue(all(a.closed for a in self.adapters.values()))

    def test_queue_overflow_policy_is_bounded(self):
        for i in range(3):
            with self.manager._condition:
                q = self.manager.queues["a"]
                if len(q) >= self.manager.queue_size:
                    q.popleft()
                    self.manager.stats["a"]["queue_overflow"] += 1
                q.append(sample("a", 1.0+i*.01, i))
        self.assertEqual(len(self.manager.queues["a"]), 2)
        self.assertEqual(self.manager.stats["a"]["queue_overflow"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
