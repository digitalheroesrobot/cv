#!/usr/bin/env python3
"""Offline tests for Lesson 51 synchronized RGB-D world reconstruction."""
import unittest
import numpy as np
from common import CameraCalibration, CameraFrame, DepthMap
from camera_adapters import CameraSample
from multi_camera import MultiCameraReconstructor


def sample(camera_id, timestamp, depth_value=2.0, width=16, height=12):
    K = np.array([[20., 0, width / 2], [0, 20., height / 2], [0, 0, 1.]])
    calib = CameraCalibration(K, width, height, np.zeros(5))
    image = np.full((height, width, 3), 100, dtype=np.uint8)
    frame = CameraFrame(image, timestamp, camera_id, 0, "BGR")
    depth = DepthMap(np.full((height, width), depth_value, dtype=np.float32),
                     timestamp, camera_id)
    return CameraSample(frame, calib, depth)


class TestMultiCameraReconstruction(unittest.TestCase):
    def setUp(self):
        self.poses = {"a": np.eye(4), "b": np.eye(4)}
        self.poses["b"][0, 3] = 0.2
        self.engine = MultiCameraReconstructor(
            self.poses, max_timestamp_skew_s=.02, stride=1,
            min_depth_m=.1, max_depth_m=5., voxel_size_m=.001)

    def test_two_camera_world_fusion(self):
        result = self.engine.process({
            "a": sample("a", 1.0),
            "b": sample("b", 1.01),
        })
        self.assertTrue(result.synchronized)
        self.assertAlmostEqual(result.max_skew_s, .01)
        self.assertEqual(result.fused_cloud.frame_id, "world")
        self.assertEqual(set(result.per_camera_points), {"a", "b"})
        self.assertGreater(len(result.fused_cloud.points), 0)
        # Camera B origin is shifted by +0.2m in world coordinates.
        self.assertGreater(result.fused_cloud.points[:, 0].max(), .2)

    def test_rejects_timestamp_skew(self):
        with self.assertRaises(ValueError):
            self.engine.process({"a": sample("a", 1.0),
                                 "b": sample("b", 1.1)})

    def test_rejects_missing_calibration_or_depth(self):
        bad = CameraSample(CameraFrame(np.zeros((12,16,3), np.uint8), 1.0, "a"))
        with self.assertRaises(ValueError):
            self.engine.process({"a": bad, "b": sample("b", 1.0)})

    def test_rejects_missing_camera_pose(self):
        with self.assertRaises(ValueError):
            MultiCameraReconstructor({"a": np.eye(4)})


if __name__ == "__main__":
    unittest.main(verbosity=2)
