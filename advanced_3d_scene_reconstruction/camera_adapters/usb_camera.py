#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OpenCV USB/UVC camera adapter.

Example:
    with USBCameraAdapter(device=0, width=1280, height=720) as camera:
        sample = camera.read()
        if sample is not None:
            print(sample.frame.image.shape, sample.frame.timestamp)

Timestamps are host monotonic acquisition timestamps, not hardware exposure
timestamps. For precision synchronization use camera hardware timestamps or
a shared trigger and replace this adapter's timestamp source accordingly.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np

from common import CameraCalibration, CameraFrame
from .base import CameraSample


class USBCameraAdapter:
    def __init__(self, device=0, width: int = 1280, height: int = 720,
                 fps: float = 30.0, camera_id: str = "usb0",
                 calibration: Optional[CameraCalibration] = None,
                 color_order: str = "BGR", backend: Optional[int] = None,
                 capture=None):
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.camera_id, self.calibration = camera_id, calibration
        self.color_order, self.backend = color_order, backend
        self._capture = capture  # injectable fake capture for deterministic tests
        self._owns_capture = capture is None
        self._frame_id = 0

    def open(self) -> None:
        if self._capture is not None and self._capture.isOpened():
            return
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("USB camera requires opencv-python: pip install opencv-python") from exc
        if self.backend is None:
            self._capture = cv2.VideoCapture(self.device)
        else:
            self._capture = cv2.VideoCapture(self.device, self.backend)
        if not self._capture.isOpened():
            self._capture = None
            raise RuntimeError(f"Could not open camera device {self.device!r}")
        # These are requests to the driver, not guarantees. Read back actual values.
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> Optional[CameraSample]:
        if self._capture is None:
            raise RuntimeError("Camera is not open; call open() first")
        ok, image = self._capture.read()
        timestamp = time.monotonic()
        if not ok or image is None:
            return None
        if image.ndim not in (2, 3):
            raise RuntimeError(f"Unexpected camera image shape: {image.shape}")
        frame = CameraFrame(
            image=np.asarray(image), timestamp=timestamp,
            camera_id=self.camera_id, frame_id=self._frame_id,
            color_order=self.color_order,
            metadata={"source": "opencv_uvc", "requested_fps": self.fps},
        )
        self._frame_id += 1
        return CameraSample(frame=frame, calibration=self.calibration)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self) -> None:
        if self._capture is not None and self._owns_capture:
            self._capture.release()
        self._capture = None

    def actual_properties(self) -> dict:
        if self._capture is None:
            raise RuntimeError("Camera is not open")
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required") from exc
        return {
            "width": self._capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._capture.get(cv2.CAP_PROP_FPS),
        }
