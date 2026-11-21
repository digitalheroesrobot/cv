#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Numerically explicit camera geometry and SE(3) utilities.

Transform convention: T_target_source maps column vectors expressed in
source coordinates into target coordinates: p_target = T_target_source @ p_source.
Camera convention is OpenCV optical coordinates (right, down, forward).
Points passed to these helpers are row-major arrays with shape (N,3).
"""
from __future__ import annotations
import numpy as np


def validate_se3(T: np.ndarray, atol: float = 1e-7) -> None:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4): raise ValueError("SE(3) transform must be 4x4")
    if not np.all(np.isfinite(T)): raise ValueError("transform must be finite")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=atol):
        raise ValueError("homogeneous transform last row must be [0,0,0,1]")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=atol):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        raise ValueError("rotation determinant must be +1")


def make_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if R.shape != (3, 3) or t.shape != (3,): raise ValueError("R must be 3x3 and t must be 3-vector")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    validate_se3(T)
    return T


def invert_se3(T_target_source: np.ndarray) -> np.ndarray:
    validate_se3(T_target_source)
    R, t = T_target_source[:3, :3], T_target_source[:3, 3]
    return make_se3(R.T, -R.T @ t)


def compose_se3(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    """Compose T_a_b and T_b_c to obtain T_a_c."""
    validate_se3(T_a_b); validate_se3(T_b_c)
    result = T_a_b @ T_b_c
    validate_se3(result)
    return result


def transform_points(points: np.ndarray, T_target_source: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3: raise ValueError("points must be (N,3)")
    validate_se3(T_target_source)
    return points @ T_target_source[:3, :3].T + T_target_source[:3, 3]


def project_points(points_camera: np.ndarray, K: np.ndarray,
                   min_depth: float = 1e-8):
    """Project 3D camera points to pixels; return (uv, valid_mask).

    Invalid/behind-camera points receive NaN pixels and valid=False.
    """
    p = np.asarray(points_camera, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3: raise ValueError("points must be (N,3)")
    if K.shape != (3, 3): raise ValueError("K must be 3x3")
    z = p[:, 2]
    valid = np.isfinite(p).all(axis=1) & (z > min_depth)
    uv = np.full((len(p), 2), np.nan, dtype=np.float64)
    q = p[valid] @ K.T
    uv[valid] = q[:, :2] / q[:, 2:3]
    return uv, valid


def backproject_pixels(uv: np.ndarray, depth_z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Back-project pixels using optical-axis Z depth, returning camera XYZ.

    This is Z-depth, not Euclidean ray range. Invalid/non-positive depths
    return NaN rows, preserving correspondence with input pixels.
    """
    uv = np.asarray(uv, dtype=np.float64)
    d = np.asarray(depth_z, dtype=np.float64).reshape(-1)
    K = np.asarray(K, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or len(uv) != len(d):
        raise ValueError("uv must be (N,2) and depth_z must have N values")
    if K.shape != (3, 3): raise ValueError("K must be 3x3")
    rays = np.column_stack([uv, np.ones(len(uv))]) @ np.linalg.inv(K).T
    points = rays * d[:, None]
    points[(~np.isfinite(d)) | (d <= 0)] = np.nan
    return points


def propagate_covariance(J: np.ndarray, covariance: np.ndarray,
                         process_covariance: np.ndarray | None = None) -> np.ndarray:
    """First-order covariance propagation: Sigma_y ~= J Sigma_x J^T + Q.

    This linearization is local. For large rotations or strongly nonlinear
    projections, use Monte-Carlo sampling or an appropriate manifold method.
    """
    J = np.asarray(J, dtype=np.float64)
    S = np.asarray(covariance, dtype=np.float64)
    if J.ndim != 2 or S.shape != (J.shape[1], J.shape[1]):
        raise ValueError("incompatible Jacobian and covariance dimensions")
    out = J @ S @ J.T
    if process_covariance is not None:
        Q = np.asarray(process_covariance, dtype=np.float64)
        if Q.shape != out.shape: raise ValueError("process covariance has wrong shape")
        out = out + Q
    return 0.5 * (out + out.T)
