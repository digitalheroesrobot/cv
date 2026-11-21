#!/usr/bin/env python3
"""Deterministic tests for Lesson 47 pose graph and keyframe policy."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from keyframe_slam import PoseGraph, PoseGraphEdge, KeyframeManager


def pose(x, y=0., z=0., yaw=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def relative(T_world_i, T_world_j):
    return np.linalg.inv(T_world_i) @ T_world_j


class TestPoseGraph(unittest.TestCase):
    def test_consistent_chain_has_zero_residual(self):
        truth = [pose(0), pose(1), pose(2)]
        graph = PoseGraph()
        for T in truth:
            graph.add_node(T)
        for i in range(2):
            graph.add_edge(PoseGraphEdge(i, i + 1, relative(truth[i], truth[i + 1])))
        self.assertLess(np.linalg.norm(graph.residual_vector()), 1e-9)

    def test_loop_closure_reduces_drift(self):
        # A square path returns to its origin. Noisy odometry edges accumulate drift.
        truth = [pose(0, 0), pose(1, 0), pose(1, 1),
                 pose(0, 1), pose(0, 0)]
        noisy = [pose(0, 0), pose(1.08, .02), pose(1.13, 1.10),
                 pose(.04, 1.18), pose(.18, .16)]
        graph = PoseGraph()
        for T in noisy:
            graph.add_node(T)
        for i in range(4):
            measured = relative(truth[i], truth[i + 1])
            graph.add_edge(PoseGraphEdge(i, i + 1, measured,
                                         translation_information=1.0,
                                         rotation_information=2.0))
        # Verified loop edge: final camera pose should coincide with node zero.
        graph.add_edge(PoseGraphEdge(0, 4, relative(truth[0], truth[4]),
                                     translation_information=20.0,
                                     rotation_information=20.0,
                                     kind="loop"))
        before = np.linalg.norm(noisy[4][:3, 3] - noisy[0][:3, 3])
        result = graph.optimize(max_nfev=300, loss="huber", f_scale=.05)
        after = np.linalg.norm(result.poses[4][:3, 3] - result.poses[0][:3, 3])
        self.assertTrue(result.success)
        self.assertLess(result.final_cost, result.initial_cost)
        self.assertLess(after, before)
        np.testing.assert_allclose(result.poses[0], noisy[0])

    def test_keyframe_motion_and_frame_gap(self):
        manager = KeyframeManager(.2, .15, 5)
        self.assertTrue(manager.should_insert(0, pose(0)))
        manager.add(0, 0., pose(0))
        self.assertFalse(manager.should_insert(1, pose(.05)))
        self.assertTrue(manager.should_insert(2, pose(.25)))
        manager.add(2, .1, pose(.25))
        self.assertTrue(manager.should_insert(7, pose(.25)))

    def test_invalid_edge_rejected(self):
        graph = PoseGraph()
        graph.add_node(np.eye(4))
        with self.assertRaises(IndexError):
            graph.add_edge(PoseGraphEdge(0, 1, np.eye(4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
