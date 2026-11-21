#!/usr/bin/env python3
"""Common camera adapter interfaces and samples."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable
import numpy as np
from common import CameraCalibration, CameraFrame, DepthMap


@dataclass
class CameraSample:
    """Synchronized sample from one physical camera.

    depth is optional. DepthMap values retain their declared unit; adapters
    should prefer metres whenever the sensor supplies metric depth.
    """
    frame: CameraFrame
    calibration: Optional[CameraCalibration] = None
    depth: Optional[DepthMap] = None


@runtime_checkable
class CameraAdapter(Protocol):
    """Minimal lifecycle contract shared by all camera sources."""
    def open(self) -> None: ...
    def read(self) -> Optional[CameraSample]: ...
    def close(self) -> None: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
