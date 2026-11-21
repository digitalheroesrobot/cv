#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intel RealSense D435 adapter using the optional pyrealsense2 SDK.

Install separately according to the host OS and librealsense compatibility:
    pip install pyrealsense2

The adapter aligns depth to the color optical frame and converts the SDK's
depth units to metres. The SDK depth scale is read from the active sensor;
it is never assumed to be 1 mm. Intrinsics are taken from the active color
stream. Distortion coefficients are retained in CameraCalibration.
"""
from __future__ import annotations
import time
from typing import Optional
import numpy as np

from common import CameraCalibration, CameraFrame, DepthMap
from .base import CameraSample


class RealSenseD435Adapter:
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 camera_id: str = "realsense_d435", serial: Optional[str] = None):
        self.width, self.height, self.fps = width, height, fps
        self.camera_id, self.serial = camera_id, serial
        self._pipeline = self._profile = self._align = self._rs = None
        self._frame_id = 0
        self.calibration = None
        self.depth_scale = None

    def open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "RealSense adapter requires pyrealsense2. Install the Intel "
                "RealSense SDK and its Python binding for this platform."
            ) from exc
        self._rs = rs
        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        self._pipeline, self._profile = pipeline, profile
        self._align = rs.align(rs.stream.color)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        K = np.array([[intr.fx, 0.0, intr.ppx],
                      [0.0, intr.fy, intr.ppy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        self.calibration = CameraCalibration(
            K=K, width=intr.width, height=intr.height,
            distortion=np.asarray(intr.coeffs, dtype=np.float64),
            model="pinhole", calibration_id=f"{self.camera_id}_factory",
        )

    def read(self) -> Optional[CameraSample]:
        if self._pipeline is None:
            raise RuntimeError("RealSense pipeline is not open")
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)
        color_frame, depth_frame = aligned.get_color_frame(), aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None
        color = np.asanyarray(color_frame.get_data()).copy()
        raw_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_m = raw_depth * self.depth_scale
        # Prefer the sensor timestamp if available. Convert milliseconds to seconds.
        sensor_timestamp = float(color_frame.get_timestamp()) / 1000.0
        timestamp = sensor_timestamp if sensor_timestamp >= 0 else time.monotonic()
        valid = np.isfinite(depth_m) & (depth_m > 0)
        sample = CameraSample(
            frame=CameraFrame(
                color, timestamp, self.camera_id, self._frame_id, "BGR",
                metadata={"source": "realsense", "sensor_timestamp_ms": sensor_timestamp,
                          "depth_scale_m_per_unit": self.depth_scale},
            ),
            calibration=self.calibration,
            depth=DepthMap(depth_m, timestamp, self.camera_id, valid_mask=valid, unit="m"),
        )
        self._frame_id += 1
        return sample

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = self._profile = self._align = self._rs = None
