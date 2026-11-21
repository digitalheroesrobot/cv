#!/usr/bin/env python3
"""Synchronized RGB-D multi-camera reconstruction in a shared world frame.

Extrinsics are supplied as T_world_camera. This module does not estimate rig
poses; they must come from a measured rig, stereo calibration, hand-eye
calibration, or a validated localization system.
"""
from dataclasses import dataclass
from typing import Dict, Mapping, Optional
import numpy as np

from common import CameraCalibration, PointCloud, validate_se3
from camera_adapters import CameraSample
from scene_reconstruction.geometry_pipeline import depth_to_point_cloud
from .fusion import fuse_point_clouds


@dataclass
class MultiCameraFrameResult:
    timestamp: float
    max_skew_s: float
    synchronized: bool
    per_camera_points: Dict[str, int]
    fused_cloud: PointCloud


def rectify_registered_rgbd(image, depth_m, calibration: CameraCalibration):
    """Rectify color and registered Z-depth with one pinhole camera model.

    Depth uses nearest-neighbour interpolation to avoid inventing ranges.
    The returned calibration has zero distortion and the original K.
    Fisheye models require a separately validated fisheye rectification path.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("RGB-D rectification requires opencv-python") from exc
    if calibration.model != "pinhole":
        raise ValueError("This rectifier supports pinhole distortion only")
    image = np.asarray(image)
    depth = np.asarray(depth_m, dtype=np.float32)
    if image.shape[:2] != depth.shape:
        raise ValueError("Registered color and depth dimensions must match")
    if (depth.shape[1], depth.shape[0]) != (calibration.width, calibration.height):
        raise ValueError("Frame dimensions do not match calibration")
    if np.allclose(calibration.distortion, 0):
        return image.copy(), depth.copy(), CameraCalibration(
            calibration.K.copy(), calibration.width, calibration.height,
            np.zeros(5), "pinhole", calibration.calibration_id + "_rectified")
    map_x, map_y = cv2.initUndistortRectifyMap(
        calibration.K, calibration.distortion, None, calibration.K,
        (calibration.width, calibration.height), cv2.CV_32FC1)
    rectified_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    rectified_depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0)
    return rectified_image, rectified_depth, CameraCalibration(
        calibration.K.copy(), calibration.width, calibration.height,
        np.zeros(5), "pinhole", calibration.calibration_id + "_rectified")


class MultiCameraReconstructor:
    """Fuse one synchronized bundle of RGB-D camera samples into world XYZ."""
    def __init__(self, T_world_camera: Mapping[str, np.ndarray],
                 max_timestamp_skew_s=0.015, stride=4,
                 min_depth_m=0.2, max_depth_m=5.0,
                 voxel_size_m=0.025, max_points=1_000_000):
        if len(T_world_camera) < 2:
            raise ValueError("At least two calibrated cameras are required")
        if max_timestamp_skew_s < 0 or stride < 1 or voxel_size_m <= 0 or max_points < 1:
            raise ValueError("Invalid synchronization or geometry limits")
        self.poses = {}
        for camera_id, T in T_world_camera.items():
            validate_se3(np.asarray(T, dtype=np.float64))
            self.poses[camera_id] = np.asarray(T, dtype=np.float64).copy()
        self.max_timestamp_skew_s = float(max_timestamp_skew_s)
        self.stride = int(stride)
        self.min_depth_m, self.max_depth_m = float(min_depth_m), float(max_depth_m)
        self.voxel_size_m, self.max_points = float(voxel_size_m), int(max_points)

    def process(self, samples: Mapping[str, CameraSample]) -> MultiCameraFrameResult:
        expected = set(self.poses)
        if set(samples) != expected:
            raise ValueError(f"Expected camera IDs {sorted(expected)}, got {sorted(samples)}")
        stamps = [float(samples[c].frame.timestamp) for c in self.poses]
        if not np.all(np.isfinite(stamps)):
            raise ValueError("Camera timestamps must be finite")
        skew = max(stamps) - min(stamps)
        if skew > self.max_timestamp_skew_s:
            raise ValueError(
                f"Unsynchronized camera bundle: skew {skew:.6f}s exceeds "
                f"{self.max_timestamp_skew_s:.6f}s")
        clouds = {}
        counts = {}
        for camera_id in self.poses:
            sample = samples[camera_id]
            if sample.calibration is None or sample.depth is None:
                raise ValueError(f"Camera {camera_id} must provide calibration and depth")
            if sample.frame.camera_id != camera_id or sample.depth.camera_id != camera_id:
                raise ValueError(f"Camera ID mismatch for {camera_id}")
            if abs(sample.frame.timestamp - sample.depth.timestamp) > 1e-3:
                raise ValueError(f"Color/depth timestamps disagree for {camera_id}")
            if sample.frame.color_order not in ("BGR", "RGB"):
                raise ValueError(f"Unsupported color order for {camera_id}")
            image, depth, rectified_calib = rectify_registered_rgbd(
                sample.frame.image, sample.depth.in_metres(), sample.calibration)
            cloud = depth_to_point_cloud(
                sample.depth.__class__(depth, sample.depth.timestamp, camera_id,
                                       unit="m"),
                rectified_calib, image, stride=self.stride,
                min_depth_m=self.min_depth_m, max_depth_m=self.max_depth_m)
            if sample.frame.color_order == "RGB" and cloud.colors is not None:
                cloud.colors = cloud.colors[:, ::-1].copy()
            clouds[camera_id] = cloud
            counts[camera_id] = len(cloud.points)
        fused = fuse_point_clouds(
            clouds, self.poses, target_frame="world",
            voxel_size=self.voxel_size_m)
        if len(fused.points) > self.max_points:
            # Deterministic spatially uniform cap after voxel fusion.
            indices = np.linspace(0, len(fused.points) - 1,
                                  self.max_points, dtype=np.int64)
            fused = PointCloud(
                fused.points[indices], fused.timestamp, fused.frame_id,
                fused.colors[indices] if fused.colors is not None else None,
                fused.normals[indices] if fused.normals is not None else None,
                fused.confidence[indices] if fused.confidence is not None else None)
        return MultiCameraFrameResult(
            timestamp=float(np.mean(stamps)), max_skew_s=float(skew),
            synchronized=True, per_camera_points=counts, fused_cloud=fused)
