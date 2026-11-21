#!/usr/bin/env python3
"""Temporal 4D scene analysis for ego-motion-compensated world point clouds.

The caller must first transform each frame into a stable world frame using
visual odometry/SLAM and camera-rig calibration. Point displacement is then
estimated with reciprocal nearest-neighbour correspondences. This transparent
baseline is not a learned scene-flow network and is sensitive to sampling,
occlusion, pose drift, and correspondence ambiguity.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree
from common import ObjectTrack


@dataclass
class SceneFlowResult:
    """Per-current-point displacement from its matched previous-frame point."""
    flow: np.ndarray                 # Nx3 metres per frame
    speed_mps: np.ndarray            # N, displacement / dt
    previous_indices: np.ndarray     # N, -1 means no valid correspondence
    correspondence_distance: np.ndarray
    labels: np.ndarray               # -1 unknown, 0 static, 1 dynamic
    confidence: np.ndarray           # N in [0,1]


@dataclass
class DynamicSceneResult:
    timestamp: float
    static_points: np.ndarray
    dynamic_points: np.ndarray
    unknown_points: np.ndarray
    static_map: np.ndarray
    scene_flow: SceneFlowResult
    dynamic_cluster_labels: np.ndarray
    frame_index: int


def _validate_points(points, name):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N,3)")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must contain finite values")
    return p


def estimate_scene_flow(previous_points, current_points, dt,
                        max_correspondence_distance_m=0.12,
                        static_speed_threshold_mps=0.08):
    """Estimate sparse scene flow using reciprocal nearest-neighbour matches.

    Coordinates must be in the same ego-motion-compensated world frame.
    A point is classified only when it has a reciprocal match within the
    distance gate; unmatched points remain unknown rather than being assumed
    dynamic (they may simply be newly visible/disoccluded background).
    """
    prev = _validate_points(previous_points, "previous_points")
    cur = _validate_points(current_points, "current_points")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if max_correspondence_distance_m <= 0 or static_speed_threshold_mps < 0:
        raise ValueError("invalid scene-flow thresholds")
    n = len(cur)
    flow = np.zeros((n, 3), dtype=np.float64)
    speed = np.zeros(n, dtype=np.float64)
    indices = np.full(n, -1, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float64)
    labels = np.full(n, -1, dtype=np.int8)
    confidence = np.zeros(n, dtype=np.float32)
    if len(prev) == 0 or n == 0:
        return SceneFlowResult(flow, speed, indices, distances, labels, confidence)
    prev_tree, cur_tree = cKDTree(prev), cKDTree(cur)
    d_cp, i_cp = prev_tree.query(cur, k=1)
    _, i_pc = cur_tree.query(prev, k=1)
    for i, (distance, j) in enumerate(zip(d_cp, i_cp)):
        if distance > max_correspondence_distance_m or i_pc[j] != i:
            continue
        indices[i] = int(j)
        distances[i] = float(distance)
        flow[i] = cur[i] - prev[j]
        speed[i] = np.linalg.norm(flow[i]) / dt
        labels[i] = 0 if speed[i] <= static_speed_threshold_mps else 1
        # Confidence falls smoothly toward the correspondence distance gate.
        confidence[i] = np.float32(max(0.0, 1.0-distance/max_correspondence_distance_m))
    return SceneFlowResult(flow, speed, indices, distances, labels, confidence)


def _voxel_centroids(points, voxel_size):
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    sums = np.zeros((inverse.max()+1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(sums, inverse, points)
    return sums / counts[:, None]


def _cluster_dynamic_points(points, voxel_size, min_cluster_points):
    """Connected components in occupied voxels using 26-neighbour adjacency."""
    n = len(points)
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    keys = np.floor(points / voxel_size).astype(np.int64)
    buckets = {}
    for i, key in enumerate(map(tuple, keys)):
        buckets.setdefault(key, []).append(i)
    visited = set()
    cluster_id = 0
    for key in buckets:
        if key in visited:
            continue
        stack = [key]
        visited.add(key)
        members = []
        while stack:
            cell = stack.pop()
            members.extend(buckets[cell])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        neighbour = (cell[0]+dx, cell[1]+dy, cell[2]+dz)
                        if neighbour in buckets and neighbour not in visited:
                            visited.add(neighbour)
                            stack.append(neighbour)
        if len(members) >= min_cluster_points:
            labels[members] = cluster_id
            cluster_id += 1
    return labels


def points_inside_track_boxes(points, tracks, margin_m=0.05):
    """Conservative axis-aligned 3D box mask for tracked objects.

    Track orientation is intentionally ignored here; dimensions are treated as
    world-axis-aligned extents. Use oriented-box geometry for rotated objects.
    """
    p = _validate_points(points, "points")
    mask = np.zeros(len(p), dtype=bool)
    for track in tracks:
        half = np.asarray(track.dimensions, dtype=np.float64) * 0.5 + margin_m
        mask |= np.all(np.abs(p-np.asarray(track.position)) <= half, axis=1)
    return mask


class DynamicSceneModel:
    """Separate stable background, moving observations, and unknown points.

    Only points with low measured scene-flow speed are admitted to the static
    voxel map. Dynamic points never contaminate that map. Track boxes may
    supplement motion evidence when a detector/tracker is available.
    """
    def __init__(self, voxel_size_m=0.04, max_static_points=300000,
                 max_correspondence_distance_m=0.12,
                 static_speed_threshold_mps=0.08,
                 dynamic_cluster_voxel_m=0.10,
                 min_cluster_points=4, track_box_margin_m=0.05):
        if voxel_size_m <= 0 or max_static_points < 1:
            raise ValueError("invalid static map settings")
        if dynamic_cluster_voxel_m <= 0 or min_cluster_points < 1:
            raise ValueError("invalid dynamic clustering settings")
        self.voxel_size_m = float(voxel_size_m)
        self.max_static_points = int(max_static_points)
        self.max_correspondence_distance_m = float(max_correspondence_distance_m)
        self.static_speed_threshold_mps = float(static_speed_threshold_mps)
        self.dynamic_cluster_voxel_m = float(dynamic_cluster_voxel_m)
        self.min_cluster_points = int(min_cluster_points)
        self.track_box_margin_m = float(track_box_margin_m)
        self.previous_points = np.empty((0, 3), dtype=np.float64)
        self.static_map = np.empty((0, 3), dtype=np.float64)
        self.last_timestamp = None
        self.frame_index = 0

    def reset(self):
        self.previous_points = np.empty((0, 3), dtype=np.float64)
        self.static_map = np.empty((0, 3), dtype=np.float64)
        self.last_timestamp = None
        self.frame_index = 0

    def update(self, points_world, timestamp, tracks=()):
        points = _validate_points(points_world, "points_world")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and non-negative")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        if self.last_timestamp is None:
            flow = SceneFlowResult(
                np.zeros_like(points), np.zeros(len(points)),
                np.full(len(points), -1, dtype=np.int64),
                np.full(len(points), np.inf), np.full(len(points), -1, dtype=np.int8),
                np.zeros(len(points), dtype=np.float32))
        else:
            dt = timestamp - self.last_timestamp
            flow = estimate_scene_flow(
                self.previous_points, points, dt,
                self.max_correspondence_distance_m,
                self.static_speed_threshold_mps)
        track_mask = points_inside_track_boxes(
            points, tracks, self.track_box_margin_m) if tracks else np.zeros(len(points), bool)
        dynamic_mask = (flow.labels == 1) | track_mask
        static_mask = (flow.labels == 0) & ~track_mask
        unknown_mask = ~(dynamic_mask | static_mask)
        static = points[static_mask]
        dynamic = points[dynamic_mask]
        # A bounded static map. This simple voxel refresh is not a probabilistic
        # occupancy map and does not perform ray-based free-space clearing.
        if len(static):
            merged = np.concatenate((self.static_map, static), axis=0)
            self.static_map = _voxel_centroids(merged, self.voxel_size_m)
            if len(self.static_map) > self.max_static_points:
                # Deterministic spatial thinning by evenly spaced array indices.
                idx = np.linspace(0, len(self.static_map)-1,
                                  self.max_static_points, dtype=np.int64)
                self.static_map = self.static_map[idx]
        clusters = _cluster_dynamic_points(
            dynamic, self.dynamic_cluster_voxel_m, self.min_cluster_points)
        result = DynamicSceneResult(
            timestamp, static, dynamic, points[unknown_mask],
            self.static_map.copy(), flow, clusters, self.frame_index)
        self.previous_points = points.copy()
        self.last_timestamp = timestamp
        self.frame_index += 1
        return result
