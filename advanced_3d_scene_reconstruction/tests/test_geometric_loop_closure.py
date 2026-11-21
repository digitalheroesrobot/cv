#!/usr/bin/env python3
"""Offline geometry tests for RGB-D loop verification and map correction."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from common.geometry import invert_se3, transform_points
from keyframe_slam import Keyframe, RGBDLoopClosureVerifier, rebuild_global_map


def pose(x=0., y=0., yaw=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, 0.]
    return T


class TestGeometricLoopClosure(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(480)
        # Asymmetric, non-planar cloud with enough geometric structure for ICP.
        self.candidate = rng.uniform([-1., -.7, 1.5], [1., .8, 4.], (2500, 3))
        self.T_candidate_current = pose(.12, -.04, .025)
        self.current = transform_points(
            self.candidate, invert_se3(self.T_candidate_current))
        self.verifier = RGBDLoopClosureVerifier(
            voxel_size_m=.012, max_correspondence_m=.20,
            min_correspondences=80, max_rmse_m=.02,
            max_translation_gap_m=.04, max_rotation_gap_rad=.04)

    def test_consistent_pnp_and_icp_are_accepted(self):
        result = self.verifier.verify_cloud_pair(
            self.candidate, self.current,
            invert_se3(self.T_candidate_current))
        self.assertTrue(result.accepted, result.reason)
        self.assertLess(result.icp_rmse_m, .01)
        np.testing.assert_allclose(result.T_candidate_current[:3, 3],
                                   self.T_candidate_current[:3, 3], atol=.025)

    def test_missing_pnp_is_rejected(self):
        result = self.verifier.verify_cloud_pair(
            self.candidate, self.current, None)
        self.assertFalse(result.accepted)
        self.assertFalse(result.pnp_available)

    def test_inconsistent_pnp_and_icp_are_rejected(self):
        bad = pose(.8, .2, .3)
        result = self.verifier.verify_cloud_pair(
            self.candidate, self.current, invert_se3(bad))
        self.assertFalse(result.accepted)

    def test_global_map_reintegrates_with_optimized_poses(self):
        local = np.array([[0., 0., 1.], [.1, 0., 1.], [0., .1, 1.]])
        frames = [
            Keyframe(0, 0., pose(), point_cloud=local),
            Keyframe(1, 1., pose(), point_cloud=local),
        ]
        corrected = [pose(), pose(1., 0., 0.)]
        world = rebuild_global_map(frames, corrected, voxel_size_m=.001)
        self.assertEqual(len(world), 6)
        self.assertGreater(world[:, 0].max(), .9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
