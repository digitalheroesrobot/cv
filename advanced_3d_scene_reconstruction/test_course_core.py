#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerical and smoke tests for the shared 3D scene reconstruction engine."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

import course_core


class CameraGeometryTests(unittest.TestCase):
    def test_projection_of_principal_axis(self):
        K = np.array([[800.0, 0.0, 320.0],
                      [0.0, 800.0, 240.0],
                      [0.0, 0.0, 1.0]])
        uv = course_core.project(np.array([[0.0, 0.0, 4.0]]), K)
        np.testing.assert_allclose(uv, [[320.0, 240.0]], atol=1e-9)

    def test_projection_scales_inversely_with_depth(self):
        K = np.array([[500.0, 0.0, 320.0],
                      [0.0, 500.0, 240.0],
                      [0.0, 0.0, 1.0]])
        near = course_core.project(np.array([[1.0, 0.0, 5.0]]), K)[0, 0]
        far = course_core.project(np.array([[1.0, 0.0, 10.0]]), K)[0, 0]
        self.assertAlmostEqual(near - 320.0, 2.0 * (far - 320.0), places=7)


class LessonSmokeTests(unittest.TestCase):
    def test_synthetic_frame_is_valid_bgr(self):
        frame = course_core.road_frame()
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)

    def test_lesson_writes_json_report_without_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = course_core.lesson_run(4, output=tmp, no_display=True)
            saved = Path(tmp) / "lesson_04_report.json"
            self.assertTrue(saved.exists())
            parsed = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(parsed["lesson"], 4)
            self.assertEqual(report["lesson"], 4)

    def test_unknown_lesson_is_rejected(self):
        with self.assertRaises(ValueError):
            course_core.lesson_run(99, no_display=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
