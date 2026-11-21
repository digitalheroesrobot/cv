#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two-camera stereo capture adapter with explicit synchronization checks.

This adapter acquires left/right frames and reports their host acquisition
timestamps. It does not pretend that sequential USB reads are hardware
synchronized. For metric stereo depth, provide a calibrated pair and compute
disparity with a rectified stereo pipeline; the baseline and rectification
must come from calibration, not guessed values.
"""
from __future__ import annotations
import time
from typing import Optional
from common import CameraCalibration, CameraFrame
from .base import CameraSample
from .usb_camera import USBCameraAdapter


class StereoCameraAdapter:
    def __init__(self, left_device=0, right_device=1,
                 left_calibration: Optional[CameraCalibration] = None,
                 right_calibration: Optional[CameraCalibration] = None,
                 max_timestamp_skew_s: float = 0.015,
                 width: int = 1280, height: int = 720, fps: float = 30.0):
        if max_timestamp_skew_s < 0:
            raise ValueError("max_timestamp_skew_s must be non-negative")
        self.left = USBCameraAdapter(left_device, width, height, fps, "stereo_left", left_calibration)
        self.right = USBCameraAdapter(right_device, width, height, fps, "stereo_right", right_calibration)
        self.max_timestamp_skew_s = max_timestamp_skew_s

    def open(self) -> None:
        self.left.open()
        try:
            self.right.open()
        except Exception:
            self.left.close()
            raise

    def read(self) -> Optional[tuple[CameraSample, CameraSample]]:
        left_sample = self.left.read()
        right_sample = self.right.read()
        if left_sample is None or right_sample is None:
            return None
        skew = abs(left_sample.frame.timestamp - right_sample.frame.timestamp)
        left_sample.frame.metadata["stereo_pair_skew_s"] = skew
        right_sample.frame.metadata["stereo_pair_skew_s"] = skew
        if skew > self.max_timestamp_skew_s:
            # Returning the pair with a quality flag is safer than silently
            # presenting it as synchronized; downstream can reject or log it.
            left_sample.frame.metadata["synchronization_valid"] = False
            right_sample.frame.metadata["synchronization_valid"] = False
        else:
            left_sample.frame.metadata["synchronization_valid"] = True
            right_sample.frame.metadata["synchronization_valid"] = True
        return left_sample, right_sample

    def close(self) -> None:
        self.left.close()
        self.right.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
