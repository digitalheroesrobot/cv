#!/usr/bin/env python3
"""Hardware-free tests for clock fitting and multi-camera time alignment."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from common import CameraFrame, CameraPose, make_se3
from multi_camera.time_sync import (
    ClockModel, MultiCameraClockSynchronizer, fit_clock_model,
    interpolate_camera_pose,
)


def frame(camera, t, fid):
    return CameraFrame(np.zeros((2, 2, 3), np.uint8), t, camera, fid)


class TestClockSync(unittest.TestCase):
    def test_affine_clock_fit_recovers_offset_and_drift(self):
        device = np.linspace(100.0, 160.0, 100)
        scale, offset = 1.0 + 55e-6, -0.12
        master = scale * device + offset
        fit = fit_clock_model(device, master)
        self.assertAlmostEqual(fit.model.scale, scale, places=9)
        expected_reference_offset = offset + (scale - 1.0) * np.mean(device)
        self.assertAlmostEqual(fit.model.offset_s, expected_reference_offset, places=7)
        self.assertLess(fit.model.residual_rmse_s, 1e-8)

    def test_robust_fit_rejects_bad_pair(self):
        device = np.linspace(1.0, 10.0, 30)
        master = 1.0 + 1.00004 * (device - 1.0) + 0.2
        master[12] += 0.05
        fit = fit_clock_model(device, master, outlier_threshold_s=0.002)
        self.assertFalse(fit.inlier_mask[12])
        self.assertAlmostEqual(fit.model.drift_ppm, 40.0, places=3)

    def test_invalid_nonmonotonic_device_times(self):
        with self.assertRaises(ValueError):
            fit_clock_model([1, 2, 1.5], [1, 2, 3])

    def test_clock_mapping_and_frame_pairing(self):
        models = {
            "a": ClockModel(1.0, 0.0),
            "b": ClockModel(1.0, -0.1),
        }
        sync = MultiCameraClockSynchronizer(["a", "b"], models,
                                            tolerance_s=0.003)
        self.assertTrue(sync.push(frame("a", 1.0, 0)))
        self.assertTrue(sync.push(frame("b", 1.1, 0)))
        result = sync.pop()
        self.assertIsNotNone(result)
        self.assertLess(result.max_skew_s, 1e-9)
        self.assertAlmostEqual(result.frames["b"].metadata["device_timestamp_s"], 1.1)

    def test_duplicate_and_out_of_order_accounting(self):
        sync = MultiCameraClockSynchronizer(
            ["a", "b"], {"a": ClockModel(), "b": ClockModel()})
        self.assertTrue(sync.push(frame("a", 1.0, 1)))
        self.assertFalse(sync.push(frame("a", 1.1, 1)))
        self.assertFalse(sync.push(frame("a", 0.9, 2)))
        self.assertEqual(sync.stats["a"]["duplicate"], 1)
        self.assertEqual(sync.stats["a"]["out_of_order"], 1)

    def test_queue_overflow_is_counted(self):
        sync = MultiCameraClockSynchronizer(
            ["a", "b"], {"a": ClockModel(), "b": ClockModel()}, queue_size=2)
        for i in range(3):
            self.assertTrue(sync.push(frame("a", 1.0 + i * .01, i)))
        self.assertEqual(sync.stats["a"]["overflow"], 1)

    def test_pose_interpolation_uses_slerp_and_linear_translation(self):
        A = make_se3(np.eye(3), np.array([0., 0., 0.]))
        B = make_se3(Rotation.from_euler("z", 90, degrees=True).as_matrix(),
                     np.array([1., 0., 0.]))
        a = CameraPose(A, 1.0, "world", "cam")
        b = CameraPose(B, 1.1, "world", "cam")
        mid = interpolate_camera_pose(a, b, 1.05)
        np.testing.assert_allclose(mid.T_world_camera[:3, 3], [.5, 0, 0])
        angle = Rotation.from_matrix(mid.T_world_camera[:3, :3]).magnitude()
        self.assertAlmostEqual(np.rad2deg(angle), 45.0, places=5)

    def test_pose_interpolation_rejects_large_gap(self):
        a = CameraPose(np.eye(4), 1.0, "world", "cam")
        b = CameraPose(np.eye(4), 1.5, "world", "cam")
        with self.assertRaises(ValueError):
            interpolate_camera_pose(a, b, 1.2, max_gap_s=.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
