#!/usr/bin/env python3
"""Numerical checks for Lesson 57 motion estimation and prediction."""
import unittest
import numpy as np
from common import ObjectTrack
from multi_camera.motion_prediction import (
    estimate_local_scene_flow, decompose_rigid_nonrigid,
    register_dynamic_cloud, predict_track, predict_pair_collision,
    motion_vector_field,
)


class TestLocalSceneFlow(unittest.TestCase):
    def test_translation_flow(self):
        previous = np.array([[0.,0.,1.],[.2,0.,1.],[0.,.2,1.],[.2,.2,1.]])
        current = previous + [.05, 0., 0.]
        result = estimate_local_scene_flow(previous, current, .1,
                                            max_distance_m=.1, neighbors=3)
        self.assertTrue(np.all(result.valid))
        self.assertTrue(np.allclose(result.flow, [.05,0.,0.], atol=1e-8))
        self.assertTrue(np.allclose(result.speed_mps, .5))

    def test_empty_cloud_is_valid_empty_result(self):
        result = estimate_local_scene_flow(np.empty((0,3)), np.empty((0,3)), .1)
        self.assertEqual(result.flow.shape, (0,3))
        self.assertEqual(result.valid.size, 0)

    def test_invalid_dt_rejected(self):
        with self.assertRaises(ValueError):
            estimate_local_scene_flow(np.zeros((1,3)), np.zeros((1,3)), 0)


class TestMotionDecomposition(unittest.TestCase):
    def test_coherent_translation_is_rigid_like(self):
        x, y = np.meshgrid(np.linspace(0,.2,5), np.linspace(0,.2,5))
        points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
        flow = np.tile([.1,0.,0.], (len(points),1))
        result = decompose_rigid_nonrigid(
            points, flow, neighborhood_radius_m=.08,
            rigid_residual_threshold_mps=.01, min_neighbors=2)
        self.assertTrue(np.all(result.rigid_mask))
        self.assertFalse(np.any(result.nonrigid_mask))

    def test_inconsistent_flow_is_nonrigid_like(self):
        x, y = np.meshgrid(np.linspace(0,.2,5), np.linspace(0,.2,5))
        points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
        flow = np.column_stack((x.ravel(), np.zeros(x.size), np.zeros(x.size)))
        result = decompose_rigid_nonrigid(
            points, flow, neighborhood_radius_m=.08,
            rigid_residual_threshold_mps=.001, min_neighbors=2)
        self.assertGreater(result.nonrigid_mask.sum(), 0)


class TestRegistrationAndPrediction(unittest.TestCase):
    def test_icp_recovers_translation(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(100,3))*.2
        translation = np.array([.04,-.02,.03])
        target = source + translation
        result = register_dynamic_cloud(
            source, target, max_correspondence_m=.2,
            max_iterations=30, min_correspondences=10)
        self.assertGreaterEqual(result.correspondences, 10)
        self.assertTrue(np.allclose(result.T_target_source[:3,3],
                                    translation, atol=1e-5))

    def test_constant_velocity_prediction(self):
        track = ObjectTrack(4, "box", np.array([1.,2.,3.]), 2.,
                            velocity=np.array([.5,0.,-.2]))
        result = predict_track(track, 2.)
        self.assertTrue(np.allclose(result.position, [2.,2.,2.6]))
        self.assertEqual(result.timestamp, 4.)

    def test_collision_prediction_for_approaching_tracks(self):
        a = ObjectTrack(1, "robot", np.array([-1.,0.,0.]), 0.,
                        velocity=np.array([1.,0.,0.]),
                        dimensions=np.array([.2,.2,.2]))
        b = ObjectTrack(2, "box", np.array([1.,0.,0.]), 0.,
                        velocity=np.array([-1.,0.,0.]),
                        dimensions=np.array([.2,.2,.2]))
        result = predict_pair_collision(a, b, horizon_s=2.)
        self.assertTrue(result.collision_predicted)
        self.assertAlmostEqual(result.time_to_closest_s, 1.)

    def test_diverging_tracks_no_collision(self):
        a = ObjectTrack(1, "a", np.array([-1.,0.,0.]), 0.,
                        velocity=np.array([-1.,0.,0.]))
        b = ObjectTrack(2, "b", np.array([1.,0.,0.]), 0.,
                        velocity=np.array([1.,0.,0.]))
        result = predict_pair_collision(a, b, horizon_s=2.)
        self.assertFalse(result.collision_predicted)

    def test_motion_vector_subsampling(self):
        points = np.arange(30,dtype=float).reshape(10,3)
        flow = np.ones_like(points)
        p, f = motion_vector_field(points, flow, max_vectors=4)
        self.assertEqual(p.shape, (4,3))
        self.assertTrue(np.all(f == 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
