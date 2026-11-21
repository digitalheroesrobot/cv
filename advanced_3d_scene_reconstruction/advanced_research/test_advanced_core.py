#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests and numerical checks for the advanced research labs."""
import tempfile
import unittest
import numpy as np
import advanced_core as ac


class AdvancedGeometryTests(unittest.TestCase):
    def test_backprojection_center_pixel(self):
        d=np.full((ac.H,ac.W),5.0,np.float32)
        xyz=ac.backproject(d)
        np.testing.assert_allclose(xyz[ac.H//2,ac.W//2],[0.0,0.0,5.0],atol=1e-8)

    def test_transform_identity(self):
        p=np.array([[1.0,2.0,3.0],[-1.0,0.0,4.0]])
        np.testing.assert_allclose(ac.transform_points(p,np.eye(4)),p)

    def test_depth_scale_shift_fit(self):
        p=np.array([1.,2.,3.,4.])
        target=2.5*p+0.75
        scale,shift,rmse=ac.fit_depth_scale_shift(p,target)
        self.assertAlmostEqual(scale,2.5,places=8)
        self.assertAlmostEqual(shift,.75,places=8)
        self.assertLess(rmse,1e-8)

    def test_occupancy_unique_voxels(self):
        p=np.array([[0.1,0.1,1.1],[0.2,0.2,1.2],[1.1,0.1,1.1]])
        vox=ac.build_occupancy(p,voxel=1.0,bounds=(-1,-1,0,3,3,3))
        self.assertEqual(len(vox),2)


class AdvancedLessonSmokeTests(unittest.TestCase):
    def test_all_labs_export_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in range(31,43):
                report=ac.run_lesson(n,output=tmp,no_display=True)
                self.assertEqual(report["lesson"],n)


if __name__=="__main__":
    unittest.main(verbosity=2)
