#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline sequential point-cloud registration and local mapping.

ICP estimates T_target_source, mapping source-frame points into target-frame
coordinates. This educational baseline is not a globally optimized SLAM system.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from common.geometry import make_se3, transform_points, compose_se3, validate_se3


@dataclass
class ICPResult:
    T_target_source: np.ndarray
    converged: bool
    iterations: int
    rmse_m: float
    correspondences: int


@dataclass
class FrameIntegrationResult:
    T_world_current: np.ndarray
    registration: ICPResult
    static_cloud_world: np.ndarray
    dynamic_points_current: np.ndarray
    map_points_world: np.ndarray


def voxel_downsample_points(points, voxel_size_m):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if voxel_size_m <= 0:
        raise ValueError("voxel size must be positive")
    if len(p) == 0:
        return np.empty((0, 3))
    keys = np.floor(p / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    return np.column_stack([
        np.bincount(inverse, weights=p[:, j]) / counts for j in range(3)
    ])


def _nearest(source, target):
    try:
        from scipy.spatial import cKDTree
        return cKDTree(target).query(source, k=1, workers=1)
    except ImportError:
        if len(source) * len(target) > 20_000_000:
            raise RuntimeError("Install scipy locally for large ICP clouds")
        indices = np.empty(len(source), dtype=np.int64)
        distances = np.empty(len(source))
        chunk = max(1, min(512, 2_000_000 // len(target)))
        for start in range(0, len(source), chunk):
            block = source[start:start + chunk]
            delta = block[:, None, :] - target[None, :, :]
            d2 = np.sum(delta * delta, axis=2)
            idx = np.argmin(d2, axis=1)
            indices[start:start + len(block)] = idx
            distances[start:start + len(block)] = np.sqrt(d2[np.arange(len(block)), idx])
        return distances, indices


def _fit_rigid(source, target):
    if len(source) < 3 or source.shape != target.shape:
        raise ValueError("at least three paired points are required")
    cs, ct = source.mean(axis=0), target.mean(axis=0)
    U, _, Vt = np.linalg.svd((source - cs).T @ (target - ct))
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return make_se3(R, ct - R @ cs)


def iterative_closest_point(source_points, target_points,
                             initial_T_target_source=None,
                             max_correspondence_distance_m=0.15,
                             max_iterations=40, tolerance_m=1e-5,
                             trim_fraction=0.85, min_correspondences=20):
    """Trimmed point-to-point ICP. Requires overlap and a reasonable initial pose."""
    src = np.asarray(source_points, dtype=np.float64)
    tgt = np.asarray(target_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or tgt.ndim != 2 or tgt.shape[1] != 3:
        raise ValueError("ICP inputs must have shape (N,3)")
    if not np.isfinite(src).all() or not np.isfinite(tgt).all():
        raise ValueError("ICP inputs must be finite")
    if len(src) < min_correspondences or len(tgt) < min_correspondences:
        raise ValueError("not enough points for ICP")
    if max_correspondence_distance_m <= 0 or max_iterations < 1 or tolerance_m <= 0:
        raise ValueError("invalid ICP parameters")
    if not 0.1 <= trim_fraction <= 1.0:
        raise ValueError("trim_fraction must be in [0.1,1]")
    T = np.eye(4) if initial_T_target_source is None else np.asarray(
        initial_T_target_source, dtype=np.float64).copy()
    validate_se3(T)
    converged, rmse, used = False, float("inf"), 0
    for iteration in range(1, max_iterations + 1):
        moved = transform_points(src, T)
        distances, indices = _nearest(moved, tgt)
        valid = distances <= max_correspondence_distance_m
        if valid.sum() < min_correspondences:
            raise RuntimeError("ICP correspondence count too small; check overlap and initialization")
        a, b, d = moved[valid], tgt[indices[valid]], distances[valid]
        keep_n = max(min_correspondences, int(np.ceil(len(d) * trim_fraction)))
        keep = np.argsort(d)[:keep_n]
        a, b = a[keep], b[keep]
        delta = _fit_rigid(a, b)
        T = delta @ T
        residual = transform_points(a, delta) - b
        rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        used = len(a)
        if (np.linalg.norm(delta[:3, 3]) < tolerance_m and
                np.linalg.norm(delta[:3, :3] - np.eye(3), ord="fro") < tolerance_m):
            converged = True
            break
    return ICPResult(T, converged, iteration, rmse, used)


class TemporalPointCloudMapper:
    """Incremental local map; first frame defines the world coordinate system."""
    def __init__(self, map_voxel_size_m=0.03, registration_voxel_size_m=0.04,
                 max_correspondence_distance_m=0.15,
                 dynamic_residual_threshold_m=0.10, max_map_points=250000):
        values = [map_voxel_size_m, registration_voxel_size_m,
                  max_correspondence_distance_m, dynamic_residual_threshold_m]
        if min(values) <= 0 or max_map_points < 100:
            raise ValueError("invalid mapper parameters")
        self.map_voxel_size_m = map_voxel_size_m
        self.registration_voxel_size_m = registration_voxel_size_m
        self.max_correspondence_distance_m = max_correspondence_distance_m
        self.dynamic_residual_threshold_m = dynamic_residual_threshold_m
        self.max_map_points = max_map_points
        self.reset()

    def reset(self):
        self.previous_points = None
        self.T_world_previous = np.eye(4)
        self.map_points_world = np.empty((0, 3))
        self.frame_count = 0

    def integrate(self, current_points_camera):
        current = np.asarray(current_points_camera, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] != 3 or not np.isfinite(current).all():
            raise ValueError("current cloud must be finite Nx3")
        current = voxel_downsample_points(current, self.registration_voxel_size_m)
        if len(current) < 20:
            raise ValueError("at least 20 points required after downsampling")
        if self.previous_points is None:
            self.previous_points = current.copy()
            self.map_points_world = voxel_downsample_points(current, self.map_voxel_size_m)
            self.frame_count = 1
            identity = ICPResult(np.eye(4), True, 0, 0.0, len(current))
            return FrameIntegrationResult(np.eye(4), identity, current.copy(),
                                          np.empty((0, 3)), self.map_points_world.copy())
        icp = iterative_closest_point(
            current, self.previous_points,
            max_correspondence_distance_m=self.max_correspondence_distance_m,
            min_correspondences=20)
        T_previous_current = icp.T_target_source
        T_world_current = compose_se3(self.T_world_previous, T_previous_current)
        aligned_previous = transform_points(current, T_previous_current)
        distances, _ = _nearest(aligned_previous, self.previous_points)
        dynamic_mask = distances > self.dynamic_residual_threshold_m
        static_world = transform_points(current[~dynamic_mask], T_world_current)
        dynamic_current = current[dynamic_mask]
        if len(static_world):
            self.map_points_world = voxel_downsample_points(
                np.vstack([self.map_points_world, static_world]), self.map_voxel_size_m)
            if len(self.map_points_world) > self.max_map_points:
                ids = np.linspace(0, len(self.map_points_world)-1,
                                  self.max_map_points, dtype=np.int64)
                self.map_points_world = self.map_points_world[ids]
        self.previous_points = current.copy()
        self.T_world_previous = T_world_current
        self.frame_count += 1
        return FrameIntegrationResult(T_world_current.copy(), icp, static_world,
                                      dynamic_current, self.map_points_world.copy())
