#!/usr/bin/env python3
"""Hardware-free tests for synchronization and multi-camera fusion."""
import unittest
import numpy as np
from common import CameraFrame, PointCloud, make_se3
from multi_camera import FrameSynchronizer, fuse_point_clouds, reprojection_rmse


def frame(camera_id, timestamp, frame_id=0):
    return CameraFrame(np.zeros((2, 2, 3), np.uint8), timestamp, camera_id, frame_id)


class TestMultiCamera(unittest.TestCase):
    def test_timestamp_synchronization(self):
        sync = FrameSynchronizer(["left", "right"], tolerance_s=0.02)
        sync.push(frame("left", 1.000))
        sync.push(frame("right", 1.012))
        result = sync.pop()
        self.assertIsNotNone(result)
        self.assertTrue(result.synchronized)
        self.assertAlmostEqual(result.max_skew_s, 0.012, places=6)

    def test_rejects_far_apart_frames(self):
        sync = FrameSynchronizer(["a", "b"], tolerance_s=0.005)
        sync.push(frame("a", 1.0))
        sync.push(frame("b", 1.1))
        self.assertIsNone(sync.pop())

    def test_drops_out_of_order_frame(self):
        sync = FrameSynchronizer(["a", "b"])
        sync.push(frame("a", 2.0))
        sync.push(frame("a", 1.0))
        self.assertEqual(sync.dropped_out_of_order["a"], 1)

    def test_point_cloud_fusion_and_voxel_filter(self):
        a = PointCloud(np.array([[0., 0., 1.], [0.01, 0., 1.]]), 1.0, "a")
        b = PointCloud(np.array([[0., 0., 1.]]), 1.0, "b")
        Ta = np.eye(4)
        Tb = make_se3(np.eye(3), np.array([1., 0., 0.]))
        fused = fuse_point_clouds({"a": a, "b": b}, {"a": Ta, "b": Tb})
        self.assertEqual(fused.frame_id, "world")
        np.testing.assert_allclose(fused.points[2], [1., 0., 1.])
        down = fuse_point_clouds({"a": a, "b": b}, {"a": Ta, "b": Tb}, voxel_size=0.1)
        self.assertLess(len(down.points), len(fused.points))

    def test_reprojection_rmse(self):
        obs = np.array([[0., 0.], [3., 4.]])
        pred = np.array([[0., 0.], [0., 0.]])
        self.assertAlmostEqual(reprojection_rmse(obs, pred), np.sqrt(12.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
