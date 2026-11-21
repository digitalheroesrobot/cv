#!/usr/bin/env python3
"""Transform and concatenate point clouds in a common target frame."""
from __future__ import annotations
from typing import Dict, Iterable, Mapping, Optional
import numpy as np
from common import PointCloud, transform_points, validate_se3


def fuse_point_clouds(
    clouds: Mapping[str, PointCloud],
    T_world_camera: Mapping[str, np.ndarray],
    target_frame: str = "world",
    voxel_size: Optional[float] = None,
) -> PointCloud:
    """Fuse camera-local clouds using T_world_camera[camera_id].

    This function performs rigid registration using supplied poses; it does not
    estimate camera poses. Optional voxel filtering keeps one representative
    point per voxel and averages available colors/confidence. Normals are
    transformed by rotation only and renormalized.
    """
    if not clouds:
        raise ValueError("at least one point cloud is required")
    all_points, all_colors, all_conf, all_normals = [], [], [], []
    has_colors = all(c.colors is not None for c in clouds.values())
    has_conf = all(c.confidence is not None for c in clouds.values())
    has_normals = all(c.normals is not None for c in clouds.values())
    timestamps = []
    for camera_id, cloud in clouds.items():
        if camera_id not in T_world_camera:
            raise KeyError(f"Missing pose for camera {camera_id}")
        T = np.asarray(T_world_camera[camera_id], dtype=np.float64)
        validate_se3(T)
        all_points.append(transform_points(cloud.points, T))
        if has_colors: all_colors.append(cloud.colors)
        if has_conf: all_conf.append(cloud.confidence)
        if has_normals:
            n = cloud.normals @ T[:3, :3].T
            lengths = np.linalg.norm(n, axis=1, keepdims=True)
            all_normals.append(n / np.maximum(lengths, 1e-12))
        timestamps.append(cloud.timestamp)
    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0) if has_colors else None
    confidence = np.concatenate(all_conf, axis=0) if has_conf else None
    normals = np.concatenate(all_normals, axis=0) if has_normals else None
    if voxel_size is not None:
        if voxel_size <= 0: raise ValueError("voxel_size must be positive")
        keys = np.floor(points / voxel_size).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = np.bincount(inverse)
        points_out = np.column_stack([
            np.bincount(inverse, weights=points[:, axis]) / count for axis in range(3)
        ])
        if colors is not None:
            colors = np.column_stack([
                np.bincount(inverse, weights=colors[:, axis]) / count for axis in range(3)
            ])
        if confidence is not None:
            confidence = np.bincount(inverse, weights=confidence) / count
        if normals is not None:
            normals = np.column_stack([
                np.bincount(inverse, weights=normals[:, axis]) / count for axis in range(3)
            ])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        points = points_out
    return PointCloud(points, max(timestamps), target_frame, colors, normals, confidence)
