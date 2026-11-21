#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical, validated data contracts shared by all 3D vision lessons.

Coordinate conventions
----------------------
Camera frame: x right, y down, z forward (OpenCV optical convention).
World frame: right-handed, application-defined; document its axes at runtime.
Length units: metres. Timestamps: seconds from a monotonic or dataset clock.
Color: uint8 BGR/RGB is declared explicitly by CameraFrame.color_order.

Arrays are copied on construction to prevent accidental aliasing between stages.
This module intentionally depends only on Python's standard library and NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple
import numpy as np


def _array(value: Any, shape: Optional[Tuple[int, ...]], name: str,
           dtype: Optional[np.dtype] = None) -> np.ndarray:
    out = np.asarray(value, dtype=dtype)
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {out.shape}")
    if not np.all(np.isfinite(out)) and np.issubdtype(out.dtype, np.number):
        raise ValueError(f"{name} contains NaN or infinity")
    return out.copy()


@dataclass
class CameraFrame:
    """One timestamped camera sample; image layout is HxW or HxWxC."""
    image: np.ndarray
    timestamp: float
    camera_id: str = "camera0"
    frame_id: int = 0
    color_order: Literal["RGB", "BGR", "GRAY", "DEPTH"] = "BGR"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = _array(self.image, None, "image")
        if self.image.ndim not in (2, 3):
            raise ValueError("image must be HxW or HxWxC")
        if self.image.ndim == 3 and self.image.shape[2] not in (1, 3, 4):
            raise ValueError("image channel count must be 1, 3, or 4")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        if not self.camera_id:
            raise ValueError("camera_id cannot be empty")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


@dataclass
class CameraCalibration:
    """Pinhole intrinsics plus optional lens distortion and image dimensions."""
    K: np.ndarray
    width: int
    height: int
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    model: str = "pinhole"
    calibration_id: str = "calibration0"

    def __post_init__(self) -> None:
        self.K = _array(self.K, (3, 3), "K", np.float64)
        self.distortion = _array(self.distortion, None, "distortion", np.float64).reshape(-1)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not np.allclose(self.K[2], [0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("K must use the standard normalized third row [0,0,1]")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0:
            raise ValueError("focal lengths must be positive")
        if self.model not in ("pinhole", "fisheye"):
            raise ValueError("model must be 'pinhole' or 'fisheye'")

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])


@dataclass
class CameraPose:
    """Rigid pose T_world_camera: maps camera-coordinate points into world."""
    T_world_camera: np.ndarray
    timestamp: float
    frame_id: str = "world"
    camera_id: str = "camera0"
    covariance: Optional[np.ndarray] = None  # 6x6 tangent-space covariance

    def __post_init__(self) -> None:
        self.T_world_camera = _array(self.T_world_camera, (4, 4), "T_world_camera", np.float64)
        from .geometry import validate_se3
        validate_se3(self.T_world_camera)
        if self.covariance is not None:
            self.covariance = _array(self.covariance, (6, 6), "pose covariance", np.float64)
            if not np.allclose(self.covariance, self.covariance.T, atol=1e-8):
                raise ValueError("pose covariance must be symmetric")
            if np.min(np.linalg.eigvalsh(self.covariance)) < -1e-8:
                raise ValueError("pose covariance must be positive semidefinite")
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")


@dataclass
class DepthMap:
    """Depth in metres by default; invalid pixels are represented by NaN or mask=False."""
    depth: np.ndarray
    timestamp: float
    camera_id: str = "camera0"
    valid_mask: Optional[np.ndarray] = None
    unit: Literal["m", "mm"] = "m"
    uncertainty: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=np.float32).copy()
        if self.depth.ndim != 2:
            raise ValueError("depth must be a 2D HxW array")
        if self.unit not in ("m", "mm"):
            raise ValueError("depth unit must be 'm' or 'mm'")
        finite_positive = np.isfinite(self.depth) & (self.depth > 0)
        if self.valid_mask is None:
            self.valid_mask = finite_positive
        else:
            self.valid_mask = _array(self.valid_mask, self.depth.shape, "valid_mask", bool)
            self.valid_mask &= finite_positive
        if self.uncertainty is not None:
            self.uncertainty = _array(self.uncertainty, self.depth.shape, "uncertainty", np.float32)
            if np.any(self.uncertainty < 0):
                raise ValueError("uncertainty must be non-negative")
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")

    def in_metres(self) -> np.ndarray:
        return self.depth.copy() * (0.001 if self.unit == "mm" else 1.0)


@dataclass
class PointCloud:
    """N 3D points with optional attributes; points use metres."""
    points: np.ndarray
    timestamp: float
    frame_id: str = "camera"
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).copy()
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("points must have shape (N,3)")
        if not np.all(np.isfinite(self.points)):
            raise ValueError("points must be finite; filter invalid points before construction")
        n = len(self.points)
        for name in ("colors", "normals"):
            value = getattr(self, name)
            if value is not None:
                value = _array(value, (n, 3), name)
                setattr(self, name, value)
        if self.confidence is not None:
            self.confidence = _array(self.confidence, (n,), "confidence", np.float32)
            if np.any((self.confidence < 0) | (self.confidence > 1)):
                raise ValueError("confidence must be in [0,1]")
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")


@dataclass
class ObjectTrack:
    """One tracked object's state in a named frame, using SI units."""
    track_id: int
    label: str
    position: np.ndarray
    timestamp: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dimensions: np.ndarray = field(default_factory=lambda: np.ones(3))
    quaternion_xyzw: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    frame_id: str = "world"
    confidence: float = 1.0
    covariance: Optional[np.ndarray] = None  # position covariance, 3x3

    def __post_init__(self) -> None:
        self.position = _array(self.position, (3,), "position", np.float64)
        self.velocity = _array(self.velocity, (3,), "velocity", np.float64)
        self.dimensions = _array(self.dimensions, (3,), "dimensions", np.float64)
        self.quaternion_xyzw = _array(self.quaternion_xyzw, (4,), "quaternion", np.float64)
        norm = np.linalg.norm(self.quaternion_xyzw)
        if norm < 1e-12: raise ValueError("orientation quaternion cannot be zero")
        self.quaternion_xyzw /= norm
        if np.any(self.dimensions <= 0): raise ValueError("dimensions must be positive")
        if not 0 <= self.confidence <= 1: raise ValueError("confidence must be in [0,1]")
        if self.track_id < 0 or self.timestamp < 0: raise ValueError("invalid track id or timestamp")
        if self.covariance is not None:
            self.covariance = _array(self.covariance, (3, 3), "position covariance", np.float64)
            if not np.allclose(self.covariance, self.covariance.T, atol=1e-8):
                raise ValueError("position covariance must be symmetric")


@dataclass
class OccupancyGrid:
    """3D voxel grid with tri-state semantics: unknown=-1, free=0, occupied=1.

    Grid origin is the world-coordinate lower corner of voxel [0,0,0].
    Array axis order is (x, y, z), not image (row, column) order.
    """
    values: np.ndarray
    origin: np.ndarray
    voxel_size: float
    timestamp: float
    frame_id: str = "world"
    semantic_labels: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.int8).copy()
        if self.values.ndim != 3: raise ValueError("occupancy values must be 3D")
        if not np.all(np.isin(self.values, [-1, 0, 1])):
            raise ValueError("occupancy values must be -1 unknown, 0 free, or 1 occupied")
        self.origin = _array(self.origin, (3,), "origin", np.float64)
        if self.voxel_size <= 0 or not np.isfinite(self.voxel_size):
            raise ValueError("voxel_size must be positive and finite")
        if self.semantic_labels is not None:
            self.semantic_labels = _array(self.semantic_labels, self.values.shape, "semantic_labels")
        if self.timestamp < 0 or not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite and non-negative")
