#!/usr/bin/env python3
"""Offline numerical tests for Lesson 44."""
import unittest
import numpy as np
from common.geometry import make_se3, transform_points
from temporal_mapping import iterative_closest_point, TemporalPointCloudMapper


class TestTemporalMapping(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.source = rng.uniform([-1, -.7, 2], [1, .8, 5], (700, 3))
        a = .035
        R = np.array([[np.cos(a), 0., np.sin(a)], [0., 1., 0.],
                      [-np.sin(a), 0., np.cos(a)]])
        self.T = make_se3(R, np.array([.08, -.015, .025]))
        self.target = transform_points(self.source, self.T)

    def test_icp_known_transform(self):
        # A close initial guess models a practical frame-to-frame odometry prior.
        a0 = .03
        R0 = np.array([[np.cos(a0), 0., np.sin(a0)], [0., 1., 0.],
                       [-np.sin(a0), 0., np.cos(a0)]])
        T0 = make_se3(R0, np.array([.075, -.01, .02]))
        result = iterative_closest_point(
            self.source, self.target, initial_T_target_source=T0,
            max_correspondence_distance_m=.3, max_iterations=60,
            tolerance_m=1e-7, trim_fraction=.95, min_correspondences=100)
        aligned = transform_points(self.source, result.T_target_source)
        error = np.sqrt(np.mean(np.sum((aligned-self.target)**2, axis=1)))
        self.assertLess(error, .025)
        np.testing.assert_allclose(result.T_target_source[:3, 3],
                                   self.T[:3, 3], atol=.025)

    def test_mapper_first_frame(self):
        mapper = TemporalPointCloudMapper(registration_voxel_size_m=.01)
        result = mapper.integrate(self.source)
        self.assertEqual(mapper.frame_count, 1)
        np.testing.assert_allclose(result.T_world_current, np.eye(4))

    def test_mapper_second_frame(self):
        mapper = TemporalPointCloudMapper(
            registration_voxel_size_m=.01, max_correspondence_distance_m=.3)
        mapper.integrate(self.source)
        result = mapper.integrate(self.target)
        self.assertEqual(mapper.frame_count, 2)
        self.assertGreater(len(result.map_points_world), 0)

    def test_invalid_cloud(self):
        with self.assertRaises(ValueError):
            TemporalPointCloudMapper().integrate(np.zeros((5, 3)))

    def test_reset(self):
        mapper = TemporalPointCloudMapper()
        mapper.integrate(self.source)
        mapper.reset()
        self.assertEqual(mapper.frame_count, 0)
        self.assertEqual(len(mapper.map_points_world), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
