#!/usr/bin/env python3
"""Hardware-free tests for camera adapter contracts."""
import unittest
import numpy as np
from camera_adapters.usb_camera import USBCameraAdapter
from camera_adapters.stereo_camera import StereoCameraAdapter


class FakeCapture:
    def __init__(self, image):
        self.image = image
        self.released = False
        self.read_count = 0
    def isOpened(self): return True
    def read(self):
        self.read_count += 1
        return True, self.image.copy()
    def release(self): self.released = True


class TestCameraAdapters(unittest.TestCase):
    def test_usb_adapter_maps_frame_contract(self):
        fake = FakeCapture(np.zeros((12, 16, 3), dtype=np.uint8))
        adapter = USBCameraAdapter(capture=fake, camera_id="test-camera")
        adapter.open()
        sample = adapter.read()
        self.assertEqual(sample.frame.camera_id, "test-camera")
        self.assertEqual((sample.frame.width, sample.frame.height), (16, 12))
        self.assertEqual(sample.frame.frame_id, 0)
        self.assertGreaterEqual(sample.frame.timestamp, 0)
        adapter.close()
        self.assertFalse(fake.released)  # injected capture remains caller-owned

    def test_read_before_open_fails(self):
        adapter = USBCameraAdapter(capture=None)
        with self.assertRaises(RuntimeError):
            adapter.read()

    def test_stereo_skew_configuration_validation(self):
        with self.assertRaises(ValueError):
            StereoCameraAdapter(max_timestamp_skew_s=-0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
