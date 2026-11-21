#!/usr/bin/env python3
"""Numerical tests for Lesson 56 temporal scene-flow and static/dynamic split."""
import unittest
import numpy as np
from multi_camera.dynamic_scene import (
    estimate_scene_flow, DynamicSceneModel, points_inside_track_boxes,
)
from common import ObjectTrack


class TestSceneFlow(unittest.TestCase):
    def test_stationary_world_points_are_static(self):
        points = np.array([[0., 0., 1.], [.2, 0., 1.], [0., .2, 1.]])
        result = estimate_scene_flow(points, points.copy(), .1,
                                      max_correspondence_distance_m=.01)
        self.assertTrue(np.all(result.labels == 0))
        self.assertTrue(np.allclose(result.flow, 0))

    def test_translated_points_have_correct_flow(self):
        previous = np.array([[0., 0., 1.], [.3, 0., 1.], [0., .3, 1.]])
        current = previous + np.array([.1, 0., 0.])
        result = estimate_scene_flow(previous, current, .1,
                                      max_correspondence_distance_m=.15,
                                      static_speed_threshold_mps=.2)
        self.assertTrue(np.all(result.labels == 1))
        self.assertTrue(np.allclose(result.flow, [.1, 0., 0.]))
        self.assertTrue(np.allclose(result.speed_mps, 1.0))

    def test_unmatched_points_are_unknown(self):
        previous = np.array([[0., 0., 1.]])
        current = np.array([[2., 0., 1.]])
        result = estimate_scene_flow(previous, current, .1,
                                      max_correspondence_distance_m=.1)
        self.assertEqual(result.labels[0], -1)
        self.assertEqual(result.previous_indices[0], -1)

    def test_reciprocal_matching_rejects_ambiguous_many_to_one(self):
        previous = np.array([[0., 0., 1.], [.02, 0., 1.]])
        current = np.array([[.001, 0., 1.]])
        result = estimate_scene_flow(previous, current, .1,
                                      max_correspondence_distance_m=.1)
        self.assertEqual(result.labels[0], -1)


class TestDynamicSceneModel(unittest.TestCase):
    def test_first_frame_is_unknown_and_not_added_to_static_map(self):
        model = DynamicSceneModel()
        points = np.array([[0., 0., 1.], [.1, 0., 1.]])
        result = model.update(points, 0.)
        self.assertEqual(len(result.unknown_points), 2)
        self.assertEqual(len(result.static_map), 0)

    def test_static_background_accumulates(self):
        model = DynamicSceneModel(voxel_size_m=.02,
                                  max_correspondence_distance_m=.05)
        points = np.array([[0., 0., 1.], [.1, 0., 1.], [0., .1, 1.]])
        model.update(points, 0.)
        result = model.update(points.copy(), .1)
        self.assertEqual(len(result.static_points), 3)
        self.assertGreaterEqual(len(result.static_map), 3)

    def test_moving_points_do_not_enter_static_map(self):
        model = DynamicSceneModel(
            voxel_size_m=.02, max_correspondence_distance_m=.2,
            static_speed_threshold_mps=.1)
        background = np.array([[0., 0., 1.], [.3, 0., 1.], [0., .3, 1.]])
        moving = np.array([[1., 0., 1.], [1.1, 0., 1.], [1., .1, 1.]])
        model.update(np.concatenate((background, moving)), 0.)
        current = np.concatenate((background, moving + [.2, 0., 0.] ))
        result = model.update(current, .1)
        self.assertEqual(len(result.dynamic_points), 3)
        self.assertTrue(np.all(result.dynamic_points[:, 0] > .9))
        self.assertFalse(np.any(result.static_map[:, 0] > .9))

    def test_track_box_masks_points(self):
        track = ObjectTrack(
            1, "box", np.array([0., 0., 1.]), 0.,
            dimensions=np.array([.4, .4, .4]))
        points = np.array([[0., 0., 1.], [1., 0., 1.]])
        mask = points_inside_track_boxes(points, [track])
        self.assertTrue(mask[0])
        self.assertFalse(mask[1])

    def test_nonmonotonic_timestamps_rejected(self):
        model = DynamicSceneModel()
        model.update(np.array([[0., 0., 1.]]), 1.)
        with self.assertRaises(ValueError):
            model.update(np.array([[0., 0., 1.]]), 1.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
