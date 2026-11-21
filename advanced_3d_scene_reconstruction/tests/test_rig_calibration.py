#!/usr/bin/env python3
"""Offline numerical tests for multi-camera rig calibration."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from common import make_se3, invert_se3
from multi_camera import RigEdge, RigPoseGraph


def pose(t, rvec=(0.0, 0.0, 0.0)):
    return make_se3(Rotation.from_rotvec(rvec).as_matrix(), np.asarray(t, float))


def edge_from_poses(i, j, poses, sigma_t=.003, sigma_r=np.deg2rad(.2)):
    # T_camera_j_camera_i = inv(T_rig_camera_j) @ T_rig_camera_i
    return RigEdge(i, j, invert_se3(poses[j]) @ poses[i], sigma_t, sigma_r)


class TestRigCalibration(unittest.TestCase):
    def setUp(self):
        self.truth = {
            "a": np.eye(4),
            "b": pose([.25, .01, 0.0], [0.0, .01, 0.0]),
            "c": pose([.02, .20, .03], [.01, 0.0, -.01]),
        }
        self.edges = [
            edge_from_poses("a", "b", self.truth),
            edge_from_poses("b", "c", self.truth),
            edge_from_poses("a", "c", self.truth),
        ]

    def test_exact_graph_recovers_camera_poses(self):
        graph = RigPoseGraph(["a", "b", "c"], self.edges, "a")
        result = graph.optimize()
        self.assertTrue(result.success)
        for camera_id in self.truth:
            delta = invert_se3(self.truth[camera_id]) @ result.T_rig_camera[camera_id]
            self.assertLess(np.linalg.norm(delta[:3, 3]), 1e-6)
            self.assertLess(np.linalg.norm(Rotation.from_matrix(
                delta[:3, :3]).as_rotvec()), 1e-6)

    def test_noisy_initialization_converges(self):
        initial = {
            "a": np.eye(4),
            "b": pose([.28, -.02, .01], [.02, -.01, .01]),
            "c": pose([-.01, .23, .01], [-.01, .02, -.02]),
        }
        result = RigPoseGraph(["a", "b", "c"], self.edges, "a").optimize(initial)
        self.assertLess(result.final_cost, result.initial_cost)
        self.assertLess(np.linalg.norm(
            result.T_rig_camera["b"][:3, 3] - self.truth["b"][:3, 3]), 1e-4)

    def test_disconnected_graph_rejected(self):
        with self.assertRaises(ValueError):
            RigPoseGraph(["a", "b", "c"], [self.edges[0]], "a")

    def test_invalid_uncertainty_rejected(self):
        with self.assertRaises(ValueError):
            RigEdge("a", "b", np.eye(4), translation_sigma_m=0)

    def test_anchor_is_identity(self):
        result = RigPoseGraph(["a", "b", "c"], self.edges, "b").optimize()
        np.testing.assert_allclose(result.T_rig_camera["b"], np.eye(4), atol=1e-12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
