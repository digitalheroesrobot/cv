#!/usr/bin/env python3
"""Numerical and failure-gate tests for Lesson 46 pose fusion."""
import unittest
import numpy as np
from common.geometry import make_se3, invert_se3, transform_points
from sensor_fusion import fuse_relative_poses, RGBDVisualICPFusion, rotation_distance_rad


class FakeVO:
    def __init__(self, T_current_previous, matches=100, inliers=90, rmse=.4):
        self.success = True
        self.T_current_previous = T_current_previous
        self.feature_matches = matches
        self.pnp_inliers = inliers
        self.reprojection_rmse_px = rmse


class TestPoseFusion(unittest.TestCase):
    def setUp(self):
        self.T = make_se3(np.eye(3), np.array([.08, -.01, .02]))

    def test_consistent_poses_are_fused(self):
        T2 = make_se3(np.eye(3), np.array([.10, -.01, .02]))
        fused, vu, iu, _, _, message = fuse_relative_poses(
            self.T, T2, .8, .6)
        self.assertTrue(vu and iu)
        self.assertEqual(message, "fused")
        self.assertGreater(fused[0, 3], self.T[0, 3])
        self.assertLess(fused[0, 3], T2[0, 3])

    def test_conflicting_pose_selects_higher_quality(self):
        T2 = make_se3(np.eye(3), np.array([.8, 0., 0.]))
        fused, vu, iu, gap, _, message = fuse_relative_poses(
            self.T, T2, .9, .2, max_translation_disagreement_m=.2)
        self.assertTrue(vu)
        self.assertFalse(iu)
        np.testing.assert_allclose(fused, self.T)
        self.assertGreater(gap, .2)
        self.assertIn("disagreement", message)

    def test_rotation_distance(self):
        angle = np.deg2rad(7.)
        R = np.array([[np.cos(angle), -np.sin(angle), 0.],
                      [np.sin(angle), np.cos(angle), 0.],
                      [0., 0., 1.]])
        self.assertAlmostEqual(np.rad2deg(rotation_distance_rad(np.eye(3), R)), 7., places=5)

    def test_fusion_mapper_initializes_and_integrates(self):
        rng = np.random.default_rng(46)
        previous = rng.uniform([-1, -.7, 2], [1, .8, 4], (1200, 3))
        T_previous_current = self.T
        current = transform_points(previous, invert_se3(T_previous_current))
        fusion = RGBDVisualICPFusion(
            registration_voxel_m=.005, map_voxel_m=.01,
            max_correspondence_m=.3, min_correspondences=30,
            max_translation_disagreement_m=.5)
        first = fusion.integrate(previous)
        self.assertEqual(first.message, "initialized")
        # The VO result is supplied in OpenCV PnP convention, inverse of target<-source.
        fake_vo = FakeVO(invert_se3(T_previous_current))
        second = fusion.integrate(current, fake_vo)
        self.assertTrue(second.success)
        self.assertTrue(second.visual_used)
        self.assertTrue(second.icp_used or second.message == "disagreement: visual estimate selected")
        self.assertEqual(fusion.frame_count, 2)
        self.assertGreater(len(fusion.map_points_world), 0)

    def test_both_estimates_missing_rejected(self):
        with self.assertRaises(ValueError):
            fuse_relative_poses(None, None, 0., 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
