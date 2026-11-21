#!/usr/bin/env python3
"""Numerical tests for RGB-D scene reconstruction primitives."""
import unittest
import numpy as np
from common import CameraCalibration, DepthMap, PointCloud
from scene_reconstruction import (
    depth_to_point_cloud, voxel_downsample, fit_ground_plane_ransac,
    euclidean_clusters, estimate_oriented_bbox,
)


class TestSceneReconstruction(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 0., 2.], [0., 100., 2.], [0., 0., 1.]])
        self.cal = CameraCalibration(self.K, 4, 4)

    def test_depth_backprojection_center_ray(self):
        depth = DepthMap(np.full((4, 4), 2.0, np.float32), 1.0, "cam")
        cloud = depth_to_point_cloud(depth, self.cal, stride=1)
        idx = np.argmin(np.linalg.norm(cloud.points, axis=1))
        np.testing.assert_allclose(cloud.points[idx], [0., 0., 2.], atol=1e-7)

    def test_depth_invalid_and_range_filter(self):
        z = np.array([[0., 0.5, 5., 10.]] * 4, dtype=np.float32)
        depth = DepthMap(z, 1.0)
        cloud = depth_to_point_cloud(depth, self.cal, stride=1,
                                     min_depth_m=0.4, max_depth_m=6.)
        self.assertEqual(len(cloud.points), 8)

    def test_voxel_downsample(self):
        cloud = PointCloud(np.array([[0., 0., 0.], [0.01, 0., 0.],
                                     [0.2, 0., 0.]]), 1.0)
        down = voxel_downsample(cloud, 0.1)
        self.assertEqual(len(down.points), 2)
        np.testing.assert_allclose(down.points[0], [0.005, 0., 0.])

    def test_ground_plane_ransac(self):
        rng = np.random.default_rng(2)
        x = rng.uniform(-1, 1, 300)
        z = rng.uniform(1, 3, 300)
        y = 1.0 + rng.normal(0, 0.002, 300)
        plane = fit_ground_plane_ransac(
            np.column_stack([x, y, z]), distance_threshold_m=0.01,
            up_axis=np.array([0., 1., 0.]))
        self.assertIsNotNone(plane)
        self.assertGreater(abs(plane.normal[1]), 0.99)
        self.assertLess(plane.rmse_m, 0.005)

    def test_euclidean_clustering(self):
        rng = np.random.default_rng(4)
        a = rng.normal([0, 0, 1], 0.01, (40, 3))
        b = rng.normal([1, 0, 1], 0.01, (45, 3))
        clusters = euclidean_clusters(np.vstack([a, b]), 0.08, 10)
        self.assertEqual(len(clusters), 2)
        self.assertEqual(sorted(map(len, clusters)), [40, 45])

    def test_oriented_bbox_dimensions(self):
        rng = np.random.default_rng(5)
        local = rng.uniform([-1., -0.5, -0.25], [1., 0.5, 0.25], (1000, 3))
        angle = 0.4
        R = np.array([[np.cos(angle), -np.sin(angle), 0.],
                      [np.sin(angle), np.cos(angle), 0.],
                      [0., 0., 1.]])
        points = local @ R.T + [0.3, 0.2, 2.0]
        box = estimate_oriented_bbox(points)
        np.testing.assert_allclose(box.dimensions, [2., 1., 0.5], atol=0.08)
        self.assertEqual(box.corners.shape, (8, 3))

    def test_empty_cluster(self):
        self.assertEqual(euclidean_clusters(np.empty((0, 3))), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
