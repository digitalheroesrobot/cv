#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric RGB-D to 3D scene primitives: cloud, plane, clusters, and boxes.

This is a transparent classical geometry baseline, not a neural detector.
Depth is optical-axis Z in metres and must be aligned to the color image.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from common import CameraCalibration, CameraPose, DepthMap, PointCloud, transform_points


@dataclass
class PlaneModel:
    normal: np.ndarray
    offset: float
    inlier_mask: np.ndarray
    rmse_m: float


@dataclass
class OrientedBoundingBox:
    center: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray
    corners: np.ndarray


@dataclass
class SceneResult:
    camera_cloud: PointCloud
    world_cloud: PointCloud
    ground_plane: Optional[PlaneModel]
    clusters: List[np.ndarray]
    boxes: List[OrientedBoundingBox]


def depth_to_point_cloud(depth: DepthMap, calibration: CameraCalibration,
                         color_image: Optional[np.ndarray] = None,
                         stride: int = 2, min_depth_m: float = 0.15,
                         max_depth_m: float = 8.0) -> PointCloud:
    """Back-project registered depth with pinhole intrinsics.

    Lens distortion must be removed/rectified before using this function.
    stride samples every Nth pixel to control computational load.
    """
    if stride < 1 or min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError("invalid stride or depth range")
    z = depth.in_metres()
    h, w = z.shape
    if (w, h) != (calibration.width, calibration.height):
        raise ValueError("depth resolution does not match calibration")
    if color_image is not None and color_image.shape[:2] != (h, w):
        raise ValueError("color image must be aligned and match depth")
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    zz = z[::stride, ::stride]
    valid = depth.valid_mask[::stride, ::stride] & np.isfinite(zz)
    valid &= (zz >= min_depth_m) & (zz <= max_depth_m)
    u, v, zv = uu[valid].astype(np.float64), vv[valid].astype(np.float64), zz[valid]
    x = (u - calibration.cx) * zv / calibration.fx
    y = (v - calibration.cy) * zv / calibration.fy
    points = np.column_stack((x, y, zv))
    colors = None
    if color_image is not None:
        colors = color_image[::stride, ::stride][valid, :3].astype(np.float64)
        if colors.size and colors.max() > 1.0:
            colors /= 255.0
    confidence = None
    if depth.uncertainty is not None:
        sigma = depth.uncertainty[::stride, ::stride][valid]
        confidence = np.clip(1.0 / (1.0 + sigma / np.maximum(zv, 1e-6)), 0, 1)
    return PointCloud(points, depth.timestamp, depth.camera_id, colors,
                      confidence=confidence)


def voxel_downsample(cloud: PointCloud, voxel_size: float) -> PointCloud:
    """Average points and attributes in each occupied voxel."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if len(cloud.points) == 0:
        return PointCloud(np.empty((0, 3)), cloud.timestamp, cloud.frame_id)
    keys = np.floor(cloud.points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inv)
    def average(values):
        if values is None:
            return None
        return np.column_stack([
            np.bincount(inv, weights=values[:, i]) / counts
            for i in range(values.shape[1])
        ])
    colors, normals = average(cloud.colors), average(cloud.normals)
    if normals is not None:
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    confidence = (np.bincount(inv, weights=cloud.confidence) / counts
                  if cloud.confidence is not None else None)
    return PointCloud(average(cloud.points), cloud.timestamp, cloud.frame_id,
                      colors, normals, confidence)


def statistical_outlier_removal(cloud: PointCloud, k: int = 12,
                                std_ratio: float = 2.0) -> PointCloud:
    """Remove points with unusually large mean k-neighbour distance."""
    n = len(cloud.points)
    if k < 2 or std_ratio <= 0:
        raise ValueError("k must be >=2 and std_ratio positive")
    if n <= k:
        return cloud
    try:
        from scipy.spatial import cKDTree
        distances, _ = cKDTree(cloud.points).query(cloud.points, k=k + 1)
        mean_distance = distances[:, 1:].mean(axis=1)
    except ImportError:
        mean_distance = np.empty(n, dtype=np.float64)
        chunk = max(1, min(512, 2_000_000 // n))
        for start in range(0, n, chunk):
            delta = cloud.points[start:start + chunk, None, :] - cloud.points[None, :, :]
            dist = np.sqrt(np.sum(delta * delta, axis=2))
            mean_distance[start:start + chunk] = np.partition(
                dist, kth=k, axis=1)[:, 1:k + 1].mean(axis=1)
    threshold = mean_distance.mean() + std_ratio * mean_distance.std()
    keep = mean_distance <= threshold
    return PointCloud(
        cloud.points[keep], cloud.timestamp, cloud.frame_id,
        cloud.colors[keep] if cloud.colors is not None else None,
        cloud.normals[keep] if cloud.normals is not None else None,
        cloud.confidence[keep] if cloud.confidence is not None else None)


def fit_ground_plane_ransac(points: np.ndarray, distance_threshold_m: float = 0.025,
                            iterations: int = 300, seed: int = 7,
                            up_axis: Optional[np.ndarray] = None,
                            max_tilt_deg: float = 25.0) -> Optional[PlaneModel]:
    """RANSAC plane n dot p + d = 0, optionally constrained by up direction."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if len(p) < 3 or distance_threshold_m <= 0 or iterations < 1:
        return None
    up = None
    if up_axis is not None:
        up = np.asarray(up_axis, dtype=np.float64)
        if up.shape != (3,) or np.linalg.norm(up) < 1e-9:
            raise ValueError("up_axis must be a non-zero 3-vector")
        up = up / np.linalg.norm(up)
        min_cos = np.cos(np.deg2rad(max_tilt_deg))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if up is not None and abs(float(normal @ up)) < min_cos:
            continue
        offset = -float(normal @ a)
        residual = np.abs(p @ normal + offset)
        inliers = residual <= distance_threshold_m
        score = int(inliers.sum())
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0] < 3:
        return None
    inliers = best[1]
    center = p[inliers].mean(axis=0)
    _, _, vh = np.linalg.svd(p[inliers] - center, full_matrices=False)
    normal = vh[-1]
    if up is not None and normal @ up < 0:
        normal = -normal
    offset = -float(normal @ center)
    residual = np.abs(p @ normal + offset)
    inliers = residual <= distance_threshold_m
    rmse = float(np.sqrt(np.mean(residual[inliers] ** 2))) if inliers.any() else float("inf")
    return PlaneModel(normal, offset, inliers, rmse)


def euclidean_clusters(points: np.ndarray, tolerance_m: float = 0.08,
                       min_points: int = 30, max_points: int = 100000) -> List[np.ndarray]:
    """Connected components under a radius-neighbour relation."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if tolerance_m <= 0 or min_points < 1 or max_points < min_points:
        raise ValueError("invalid clustering parameters")
    n = len(p)
    if n == 0:
        return []
    try:
        from scipy.spatial import cKDTree
        neighbours = cKDTree(p).query_ball_point(p, tolerance_m)
    except ImportError:
        if n > 5000:
            raise RuntimeError("Install scipy for clustering clouds larger than 5000 points")
        neighbours = []
        for i in range(n):
            d2 = np.sum((p - p[i]) ** 2, axis=1)
            neighbours.append(np.flatnonzero(d2 <= tolerance_m ** 2).tolist())
    visited = np.zeros(n, dtype=bool)
    clusters = []
    for seed_idx in range(n):
        if visited[seed_idx]:
            continue
        visited[seed_idx] = True
        stack, component = [seed_idx], []
        while stack:
            idx = stack.pop()
            component.append(idx)
            for nb in neighbours[idx]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if min_points <= len(component) <= max_points:
            clusters.append(np.asarray(component, dtype=np.int64))
    return clusters


def estimate_oriented_bbox(points: np.ndarray) -> OrientedBoundingBox:
    """Estimate a PCA-aligned box; not a semantic/class-conditioned detector."""
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 4:
        raise ValueError("at least four 3D points are required")
    mean = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - mean, full_matrices=False)
    axes = vh.T
    local = (p - mean) @ axes
    lo, hi = local.min(axis=0), local.max(axis=0)
    order = np.argsort(hi - lo)[::-1]
    axes = axes[:, order]
    dimensions = (hi - lo)[order]
    local_center = ((lo + hi) / 2)[order]
    center = mean + axes @ local_center
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    half = dimensions / 2
    signs = np.array([[x, y, z] for x in (-1, 1)
                      for y in (-1, 1) for z in (-1, 1)], dtype=np.float64)
    corners = center + (signs * half) @ axes.T
    return OrientedBoundingBox(center, dimensions, axes, corners)


class SceneReconstructionPipeline:
    """One-frame RGB-D pipeline with optional camera-to-world pose."""
    def __init__(self, voxel_size_m=0.02, stride=2, min_depth_m=0.15,
                 max_depth_m=8.0, ground_threshold_m=0.025,
                 cluster_tolerance_m=0.08, min_cluster_points=30):
        self.voxel_size_m = voxel_size_m
        self.stride = stride
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.ground_threshold_m = ground_threshold_m
        self.cluster_tolerance_m = cluster_tolerance_m
        self.min_cluster_points = min_cluster_points

    def process(self, depth: DepthMap, calibration: CameraCalibration,
                color_image=None, pose: Optional[CameraPose] = None) -> SceneResult:
        cloud = depth_to_point_cloud(depth, calibration, color_image, self.stride,
                                     self.min_depth_m, self.max_depth_m)
        cloud = voxel_downsample(cloud, self.voxel_size_m)
        cloud = statistical_outlier_removal(cloud)
        ground = fit_ground_plane_ransac(
            cloud.points, self.ground_threshold_m, up_axis=np.array([0., 1., 0.]),
            max_tilt_deg=25.0) if len(cloud.points) >= 3 else None
        object_mask = ~ground.inlier_mask if ground is not None else np.ones(len(cloud.points), bool)
        object_points = cloud.points[object_mask]
        clusters = euclidean_clusters(object_points, self.cluster_tolerance_m,
                                      self.min_cluster_points)
        boxes = [estimate_oriented_bbox(object_points[idx]) for idx in clusters if len(idx) >= 4]
        if pose is None:
            world = cloud
        else:
            world = PointCloud(transform_points(cloud.points, pose.T_world_camera),
                               cloud.timestamp, pose.frame_id, cloud.colors,
                               cloud.normals, cloud.confidence)
        return SceneResult(cloud, world, ground, clusters, boxes)
