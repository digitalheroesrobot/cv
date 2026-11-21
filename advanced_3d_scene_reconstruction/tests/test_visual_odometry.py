#!/usr/bin/env python3
"""Numerical tests for local RGB-D visual odometry."""
import unittest
import numpy as np
from common.geometry import make_se3, transform_points
from visual_odometry import backproject_keypoints, estimate_rgbd_pnp


class TestRGBDVisualOdometry(unittest.TestCase):
    def setUp(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV not installed")
        self.cv2 = cv2
        self.K = np.array([[520., 0., 320.], [0., 520., 240.], [0., 0., 1.]])
        rng = np.random.default_rng(19)
        self.points = np.column_stack([
            rng.uniform(-.7, .7, 500),
            rng.uniform(-.5, .5, 500),
            rng.uniform(2., 4.5, 500)])
        self.T = make_se3(
            np.array([[.9998, 0., .02], [0., 1., 0.], [-.02, 0., .9998]]),
            np.array([.04, -.015, .02]))
        self.moved = transform_points(self.points, self.T)
        self.prev_uv = self._project(self.points)
        self.curr_uv = self._project(self.moved)
        self.depth = np.full((480, 640), np.nan, dtype=np.float32)
        for (u, v), p in zip(self.prev_uv, self.points):
            x, y = int(round(u)), int(round(v))
            if 0 <= x < 640 and 0 <= y < 480:
                self.depth[y, x] = p[2]

    def _project(self, p):
        q = p @ self.K.T
        return q[:, :2] / q[:, 2:3]

    def test_backproject_keypoints(self):
        uv = np.array([[320., 240.]])
        depth = np.full((480, 640), 2., dtype=np.float32)
        xyz, valid = backproject_keypoints(uv, depth, self.K)
        self.assertEqual(len(valid), 1)
        np.testing.assert_allclose(xyz[0], [0., 0., 2.], atol=1e-8)

    def test_pnp_recovers_metric_transform(self):
        # Use unique pixel neighbourhoods and exact 3D->2D correspondences.
        T, inliers, rmse, _ = estimate_rgbd_pnp(
            self.prev_uv, self.curr_uv, self.depth, self.K,
            min_depth_m=1., max_depth_m=5., reprojection_threshold_px=1.5,
            min_inliers=30, patch_radius=0)
        self.assertGreaterEqual(inliers, 30)
        self.assertLess(rmse, 1.0)
        np.testing.assert_allclose(T[:3, 3], self.T[:3, 3], atol=.025)

    def test_depth_sampling_rejects_invalid(self):
        depth = np.zeros((480, 640), dtype=np.float32)
        xyz, valid = backproject_keypoints(
            np.array([[320., 240.]]), depth, self.K)
        self.assertEqual(len(valid), 0)
        self.assertEqual(xyz.shape, (0, 3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
