#!/usr/bin/env python3
"""Local scene-flow refinement, rigid/non-rigid motion, and prediction.

All geometry is metric and expressed in one ego-motion-compensated world frame.
The algorithms are classical baselines intended for transparent offline study;
they are not learned dense optical flow, articulated tracking, or a safety-rated
collision detector.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.spatial import cKDTree


def _points(value, name):
    p = np.asarray(value, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.all(np.isfinite(p)):
        raise ValueError(f"{name} must be a finite (N,3) array")
    return p


@dataclass
class LocalFlowResult:
    flow: np.ndarray
    speed_mps: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray


@dataclass
class MotionDecomposition:
    rigid_mask: np.ndarray
    nonrigid_mask: np.ndarray
    uncertain_mask: np.ndarray
    local_consistency: np.ndarray


@dataclass
class RegistrationResult:
    T_target_source: np.ndarray
    converged: bool
    iterations: int
    rmse_m: float
    correspondences: int


@dataclass
class MotionPrediction:
    track_id: int
    timestamp: float
    horizon_s: float
    position: np.ndarray
    velocity: np.ndarray
    covariance: Optional[np.ndarray]


@dataclass
class CollisionPrediction:
    track_id_a: int
    track_id_b: int
    time_to_closest_s: float
    closest_distance_m: float
    combined_radius_m: float
    collision_predicted: bool


def estimate_local_scene_flow(previous_points, current_points, dt,
                              max_distance_m=0.15, neighbors=6,
                              static_threshold_mps=0.05):
    """Dense-ish flow by inverse-distance interpolation of reciprocal matches.

    A reciprocal nearest-neighbour correspondence is a sparse anchor. Nearby
    current points receive a distance-weighted average of anchor displacements.
    This fills local gaps but can smear motion across object boundaries.
    """
    prev, cur = _points(previous_points, "previous_points"), _points(current_points, "current_points")
    if not np.isfinite(dt) or dt <= 0 or max_distance_m <= 0 or neighbors < 1:
        raise ValueError("invalid flow parameters")
    n = len(cur)
    flow = np.zeros((n, 3))
    confidence = np.zeros(n)
    valid = np.zeros(n, dtype=bool)
    if len(prev) == 0 or n == 0:
        return LocalFlowResult(flow, np.zeros(n), valid, confidence)
    tree_prev, tree_cur = cKDTree(prev), cKDTree(cur)
    d, idx = tree_prev.query(cur, k=1)
    _, reverse = tree_cur.query(prev, k=1)
    anchors = (d <= max_distance_m) & (reverse[idx] == np.arange(n))
    anchor_points = cur[anchors]
    anchor_flow = cur[anchors] - prev[idx[anchors]]
    if len(anchor_points):
        tree_anchor = cKDTree(anchor_points)
        k = min(neighbors, len(anchor_points))
        distances, ids = tree_anchor.query(cur, k=k)
        if k == 1:
            distances, ids = distances[:, None], ids[:, None]
        weights = 1.0 / np.maximum(distances, 1e-4)
        weights[distances > max_distance_m] = 0.0
        weight_sum = weights.sum(axis=1)
        good = weight_sum > 0
        flow[good] = np.sum(anchor_flow[ids[good]] * weights[good, :, None], axis=1) / weight_sum[good, None]
        confidence[good] = np.clip(weight_sum[good] / (weight_sum[good] + 1.0), 0, 1)
        valid = good
    speed = np.linalg.norm(flow, axis=1) / dt
    # A static threshold is intentionally not baked into validity: callers
    # need the continuous velocity field for decomposition and prediction.
    return LocalFlowResult(flow, speed, valid, confidence)


def decompose_rigid_nonrigid(points, flow, valid=None,
                            neighborhood_radius_m=0.20,
                            rigid_residual_threshold_mps=0.12,
                            min_neighbors=4):
    """Classify local flow consistency, not physical rigidity proof.

    Locally coherent displacement vectors are marked rigid-like; inconsistent
    vectors are marked non-rigid-like. Points without enough neighbours remain
    uncertain. A translating rigid body is a simple case; rotation can create
    spatially varying but still rigid motion and may be misclassified.
    """
    p, f = _points(points, "points"), _points(flow, "flow")
    if len(p) != len(f):
        raise ValueError("points and flow must have equal length")
    if neighborhood_radius_m <= 0 or rigid_residual_threshold_mps < 0 or min_neighbors < 1:
        raise ValueError("invalid decomposition thresholds")
    valid_mask = np.ones(len(p), bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid_mask.shape != (len(p),):
        raise ValueError("valid mask must have shape (N,)")
    rigid = np.zeros(len(p), bool)
    nonrigid = np.zeros(len(p), bool)
    uncertain = ~valid_mask.copy()
    consistency = np.full(len(p), np.inf)
    if len(p) == 0:
        return MotionDecomposition(rigid, nonrigid, uncertain, consistency)
    tree = cKDTree(p)
    for i in np.flatnonzero(valid_mask):
        ids = tree.query_ball_point(p[i], neighborhood_radius_m)
        ids = [j for j in ids if valid_mask[j] and j != i]
        if len(ids) < min_neighbors:
            uncertain[i] = True
            continue
        residual = np.linalg.norm(f[ids] - f[i], axis=1)
        consistency[i] = float(np.median(residual))
        if consistency[i] <= rigid_residual_threshold_mps:
            rigid[i] = True
        else:
            nonrigid[i] = True
    return MotionDecomposition(rigid, nonrigid, uncertain, consistency)


def _fit_rigid(source, target):
    """Kabsch transform mapping source points to target points."""
    a, b = source.mean(axis=0), target.mean(axis=0)
    H = (source-a).T @ (target-b)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = b - R @ a
    return T


def register_dynamic_cloud(source_points, target_points, max_correspondence_m=0.15,
                           max_iterations=30, tolerance=1e-5,
                           min_correspondences=6):
    """Point-to-point ICP for a dynamic object cloud; transform source -> target."""
    src, dst = _points(source_points, "source_points"), _points(target_points, "target_points")
    if max_correspondence_m <= 0 or max_iterations < 1 or min_correspondences < 3:
        raise ValueError("invalid registration parameters")
    if len(src) < min_correspondences or len(dst) < min_correspondences:
        raise ValueError("not enough points for registration")
    T = np.eye(4)
    tree = cKDTree(dst)
    previous_rmse = np.inf
    converged = False
    count = 0
    for iteration in range(max_iterations):
        transformed = src @ T[:3, :3].T + T[:3, 3]
        distances, indices = tree.query(transformed, k=1)
        keep = distances <= max_correspondence_m
        if int(keep.sum()) < min_correspondences:
            break
        delta = _fit_rigid(transformed[keep], dst[indices[keep]])
        T = delta @ T
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        count = int(keep.sum())
        if abs(previous_rmse-rmse) < tolerance:
            converged = True
            break
        previous_rmse = rmse
    transformed = src @ T[:3, :3].T + T[:3, 3]
    distances, _ = tree.query(transformed, k=1)
    keep = distances <= max_correspondence_m
    count = int(keep.sum())
    rmse = float(np.sqrt(np.mean(distances[keep] ** 2))) if count else float("inf")
    return RegistrationResult(T, converged, iteration+1, rmse, count)


def predict_track(track, horizon_s, acceleration_covariance=None):
    """Constant-velocity prediction from a common ObjectTrack record."""
    if not np.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("horizon_s must be finite and non-negative")
    position = np.asarray(track.position, dtype=np.float64) + np.asarray(track.velocity, dtype=np.float64)*horizon_s
    covariance = None
    if track.covariance is not None:
        P = np.asarray(track.covariance, dtype=np.float64)
        if P.shape != (3, 3):
            raise ValueError("track position covariance must be 3x3")
        Q = np.zeros((3, 3)) if acceleration_covariance is None else np.asarray(acceleration_covariance, dtype=np.float64)
        if Q.shape != (3, 3):
            raise ValueError("acceleration_covariance must be 3x3")
        covariance = P + (horizon_s**2) * Q
    return MotionPrediction(track.track_id, track.timestamp+horizon_s,
                            horizon_s, position, np.asarray(track.velocity).copy(), covariance)


def predict_pair_collision(track_a, track_b, horizon_s=3.0,
                           safety_margin_m=0.0):
    """Closest-approach test using circumscribed spheres around 3D boxes.

    Conservative for box orientation but not a full continuous collision
    detector. Constant velocities are assumed over the prediction horizon.
    """
    if horizon_s < 0 or safety_margin_m < 0:
        raise ValueError("horizon and safety margin must be non-negative")
    pa, pb = np.asarray(track_a.position), np.asarray(track_b.position)
    va, vb = np.asarray(track_a.velocity), np.asarray(track_b.velocity)
    r = 0.5*np.linalg.norm(track_a.dimensions) + 0.5*np.linalg.norm(track_b.dimensions) + safety_margin_m
    relative_p, relative_v = pa-pb, va-vb
    vv = float(relative_v @ relative_v)
    t = 0.0 if vv < 1e-12 else float(np.clip(-(relative_p @ relative_v)/vv, 0.0, horizon_s))
    distance = float(np.linalg.norm(relative_p + relative_v*t))
    return CollisionPrediction(track_a.track_id, track_b.track_id, t,
                               distance, float(r), distance <= r)


def motion_vector_field(points, flow, valid=None, max_vectors=500):
    """Deterministically subsample valid point-flow vectors for visualization."""
    p, f = _points(points, "points"), _points(flow, "flow")
    if len(p) != len(f) or max_vectors < 1:
        raise ValueError("invalid vector field inputs")
    mask = np.ones(len(p), bool) if valid is None else np.asarray(valid, bool)
    ids = np.flatnonzero(mask)
    if len(ids) > max_vectors:
        ids = ids[np.linspace(0, len(ids)-1, max_vectors, dtype=int)]
    return p[ids], f[ids]
