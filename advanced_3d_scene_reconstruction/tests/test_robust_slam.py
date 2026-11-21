#!/usr/bin/env python3
"""Deterministic offline tests for Lesson 50 robust SLAM utilities."""
import unittest
import numpy as np

from keyframe_slam.robust_slam import (
    TrackingMonitor, TrackingState, SubmapManager, LoopClosurePolicy,
    information_from_covariance, whiten_pose_residual, MemoryBudget,
    SLAMHealthMonitor,
)


def pose(x=0.0, yaw=0.0):
    T = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[0, 3] = x
    return T


class TestTrackingMonitor(unittest.TestCase):
    def test_hysteresis_and_lost_transition(self):
        monitor = TrackingMonitor(lost_after=2, recover_after=2)
        self.assertEqual(monitor.update(.9).state, TrackingState.INITIALIZING)
        self.assertEqual(monitor.update(.9).state, TrackingState.TRACKING)
        self.assertEqual(monitor.update(.1).state, TrackingState.DEGRADED)
        decision = monitor.update(.1)
        self.assertEqual(decision.state, TrackingState.LOST)
        self.assertTrue(decision.start_new_submap)

    def test_invalid_pose_cannot_be_accepted(self):
        monitor = TrackingMonitor(lost_after=1)
        result = monitor.update(.99, pose_valid=False)
        self.assertFalse(result.accepted)


class TestSubmapsAndMemory(unittest.TestCase):
    def test_submap_limits_and_rotation(self):
        manager = SubmapManager(max_submaps=2, max_keyframes_per_submap=2,
                                max_total_points=100)
        manager.start()
        for i in range(6):
            manager.add_keyframe(i, 20)
        self.assertLessEqual(len(manager.submaps), 2)
        self.assertLessEqual(manager.total_points, 100)
        self.assertEqual(len(manager.current.keyframe_ids), 2)

    def test_memory_budget_evicts_oldest(self):
        budget = MemoryBudget(max_keyframes=2, max_points=100,
                              max_payload_bytes=1000)
        budget.register(1, 400, 20)
        budget.register(2, 400, 20)
        budget.register(3, 400, 20)
        evicted, snapshot = budget.enforce()
        self.assertEqual(evicted, [1])
        self.assertTrue(snapshot.within_budget)


class TestLoopAndUncertainty(unittest.TestCase):
    def test_loop_policy_accepts_consistent_and_rejects_duplicate(self):
        policy = LoopClosurePolicy(min_inliers=10)
        A = pose(.4, .05)
        accepted = policy.evaluate(2, 40, 20, 1.0, A, A.copy())
        duplicate = policy.evaluate(2, 40, 20, 1.0, A, A.copy())
        self.assertTrue(accepted.accepted)
        self.assertFalse(duplicate.accepted)
        self.assertEqual(duplicate.reason, "duplicate-loop")

    def test_loop_policy_rejects_geometric_disagreement(self):
        policy = LoopClosurePolicy(min_inliers=10)
        decision = policy.evaluate(1, 30, 30, .5, pose(), pose(1.0, .4))
        self.assertFalse(decision.accepted)
        self.assertIn("disagreement", decision.reason)

    def test_information_matrix_and_whitening(self):
        covariance = np.diag([.01, .02, .03, .04, .05, .06]) ** 2
        info = information_from_covariance(covariance)
        self.assertTrue(np.all(np.linalg.eigvalsh(info) > 0))
        r = np.ones(6)
        white = whiten_pose_residual(r, info)
        self.assertTrue(np.all(np.isfinite(white)))
        with self.assertRaises(ValueError):
            information_from_covariance(np.eye(3))


class TestHealthMonitor(unittest.TestCase):
    def test_frame_gap_and_failure_rate(self):
        monitor = SLAMHealthMonitor(window=4, max_failure_rate=.2,
                                    max_frame_interval_s=.1)
        monitor.update(0.0, True)
        result = monitor.update(.5, False)
        self.assertIn("frame-gap", result["warnings"])
        self.assertIn("high-frame-failure-rate", result["warnings"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
