#!/usr/bin/env python3
"""Numerical tests for Lesson 55 multi-camera fusion and 3D tracking."""
import unittest
import numpy as np
from multi_camera.object_tracking import (
    Detection3D, MultiCameraDetectionFusion, ConstantVelocityKalman,
    MultiCamera3DTracker,
)


def det(x, t, camera="cam0", label="box", embedding=None, confidence=.9):
    return Detection3D(
        np.array([x, 0., 2.]), np.array([.4, .3, .3]), label,
        t, camera, confidence=confidence, embedding=embedding)


class TestCrossCameraFusion(unittest.TestCase):
    def test_two_cameras_fuse_same_object(self):
        fused = MultiCameraDetectionFusion(.2).fuse([
            det(0.0, 1.0, "a"), det(.03, 1.001, "b")])
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].source_count, 2)
        self.assertEqual(fused[0].camera_ids, ("a", "b"))
        self.assertAlmostEqual(fused[0].center[0], .015, places=3)

    def test_same_camera_detections_are_not_merged(self):
        fused = MultiCameraDetectionFusion(.2).fuse([
            det(0.0, 1.0, "a"), det(.03, 1.0, "a")])
        self.assertEqual(len(fused), 2)

    def test_different_classes_are_not_merged(self):
        fused = MultiCameraDetectionFusion(.2).fuse([
            det(0.0, 1.0, "a", "box"), det(.01, 1.0, "b", "person")])
        self.assertEqual(len(fused), 2)

    def test_temporal_skew_rejected(self):
        with self.assertRaises(ValueError):
            MultiCameraDetectionFusion().fuse([
                det(0, 1.0, "a"), det(0, 1.1, "b")])


class TestKalman(unittest.TestCase):
    def test_predict_constant_velocity(self):
        kf = ConstantVelocityKalman()
        state = np.array([0., 0., 0., 1., 0., 0.])
        predicted, covariance = kf.predict(state, np.eye(6), .5)
        self.assertAlmostEqual(predicted[0], .5)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)

    def test_update_reduces_position_uncertainty(self):
        kf = ConstantVelocityKalman()
        state = np.zeros(6)
        covariance = np.eye(6)
        updated, P, _, _ = kf.update(
            state, covariance, np.array([1., 0., 0.]), .9)
        self.assertLess(P[0, 0], covariance[0, 0])
        self.assertGreater(updated[0], 0)


class TestMultiCamera3DTracker(unittest.TestCase):
    def test_persistent_id_and_velocity(self):
        tracker = MultiCamera3DTracker(min_hits=1, measurement_noise_m=.02)
        ids = []
        for frame in range(12):
            t = frame / 10
            tracks = tracker.update([det(.1*frame, t)], t)
            ids.append(tracks[0].track_id)
        self.assertEqual(len(set(ids)), 1)
        self.assertAlmostEqual(tracks[0].velocity[0], 1.0, delta=.25)

    def test_short_occlusion_keeps_track_id(self):
        tracker = MultiCamera3DTracker(min_hits=1, max_missed_frames=3)
        first = tracker.update([det(0, 0)], 0)[0].track_id
        tracker.update([], .1)
        tracker.update([], .2)
        returned = tracker.update([det(.3, .3)], .3)
        self.assertEqual(returned[0].track_id, first)

    def test_class_mismatch_creates_new_track(self):
        tracker = MultiCamera3DTracker(min_hits=1)
        first = tracker.update([det(0, 0, label="box")], 0)[0].track_id
        second = tracker.update([det(0, .1, label="person")], .1)
        self.assertNotEqual(second[0].track_id, first)

    def test_track_expires_after_max_age_without_observations(self):
        tracker = MultiCamera3DTracker(min_hits=1, max_track_age_s=.25,
                                       max_missed_frames=20)
        tracker.update([det(0, 0)], 0)
        tracker.update([], .1)
        tracker.update([], .2)
        tracker.update([], .3)
        self.assertEqual(tracker.diagnostics()["active_tracks"], 0)

    def test_timestamp_must_increase(self):
        tracker = MultiCamera3DTracker(min_hits=1)
        tracker.update([det(0, 1)], 1)
        with self.assertRaises(ValueError):
            tracker.update([], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
