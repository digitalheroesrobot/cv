#!/usr/bin/env python3
"""Unit tests for shared contracts and the geometry engine."""
import unittest
import numpy as np

from common import (
    CameraCalibration, CameraFrame, CameraPose, DepthMap, ObjectTrack,
    OccupancyGrid, PointCloud, backproject_pixels, compose_se3, invert_se3,
    make_se3, project_points, propagate_covariance, transform_points,
)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])

    def test_project_backproject_roundtrip(self):
        points = np.array([[0.2, -0.1, 2.0], [1.0, 0.5, 4.0]])
        uv, valid = project_points(points, self.K)
        recovered = backproject_pixels(uv, points[:, 2], self.K)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(recovered, points, atol=1e-10)

    def test_se3_inverse_and_composition(self):
        angle = 0.3
        R = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        T = make_se3(R, np.array([1., 2., 3.]))
        I = compose_se3(invert_se3(T), T)
        np.testing.assert_allclose(I, np.eye(4), atol=1e-10)
        p = np.array([[0., 0., 0.], [1., 2., 3.]])
        np.testing.assert_allclose(transform_points(transform_points(p, T), invert_se3(T)), p)

    def test_covariance_propagation(self):
        J = np.array([[2., 0.], [0., 3.]])
        S = np.diag([0.1, 0.2])
        np.testing.assert_allclose(propagate_covariance(J, S), np.diag([0.4, 1.8]))

    def test_data_contracts(self):
        frame = CameraFrame(np.zeros((8, 10, 3), np.uint8), 1.0)
        self.assertEqual((frame.width, frame.height), (10, 8))
        CameraCalibration(self.K, 640, 480)
        CameraPose(np.eye(4), 1.0)
        self.assertTrue(DepthMap(np.ones((2, 2)), 1.0).valid_mask.all())
        PointCloud(np.zeros((3, 3)), 1.0)
        ObjectTrack(1, "car", np.zeros(3), 1.0)
        OccupancyGrid(-np.ones((2, 2, 2), dtype=np.int8), np.zeros(3), 0.2, 1.0)

    def test_reject_invalid_pose(self):
        bad = np.eye(4); bad[0, 0] = 2
        with self.assertRaises(ValueError):
            CameraPose(bad, 0.0)

    def test_invalid_projection(self):
        uv, valid = project_points(np.array([[0., 0., -1.], [0., 0., 2.]]), self.K)
        self.assertEqual(valid.tolist(), [False, True])
        self.assertTrue(np.isnan(uv[0]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
