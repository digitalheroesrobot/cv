#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confidence-aware fusion of RGB-D visual odometry and point-cloud ICP.

Both motion estimates map CURRENT-camera points into PREVIOUS-camera
coordinates:
    T_previous_current

The visual front-end returns T_current_previous, so it must be inverted.
ICP directly estimates T_previous_current from current cloud to previous cloud.

This module uses a geodesic interpolation on SO(3) and linear translation
interpolation. It is a transparent teaching fusion rule, not a full
probabilistic factor-graph optimizer.
"""
from dataclasses import dataclass
import numpy as np

from common.geometry import (
    compose_se3, invert_se3, make_se3, transform_points, validate_se3
)
from temporal_mapping.odometry import iterative_closest_point, voxel_downsample_points


@dataclass
class PoseFusionResult:
    success: bool
    T_previous_current: np.ndarray
    T_world_current: np.ndarray
    T_visual: np.ndarray | None
    T_icp: np.ndarray | None
    visual_score: float
    icp_score: float
    visual_used: bool
    icp_used: bool
    disagreement_translation_m: float
    disagreement_rotation_deg: float
    icp_rmse_m: float
    icp_correspondences: int
    message: str


def _so3_exp(vector):
    w = np.asarray(vector, dtype=np.float64).reshape(3)
    theta = np.linalg.norm(w)
    W = np.array([[0., -w[2], w[1]],
                  [w[2], 0., -w[0]],
                  [-w[1], w[0], 0.]])
    if theta < 1e-10:
        return np.eye(3) + W + 0.5 * (W @ W)
    return np.eye(3) + (np.sin(theta) / theta) * W + (
        (1.0 - np.cos(theta)) / (theta * theta)) * (W @ W)


def _so3_log(R):
    R = np.asarray(R, dtype=np.float64)
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))
    vee = np.array([R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1]])
    if theta < 1e-8:
        return 0.5 * vee
    if np.pi - theta < 1e-5:
        # Stable axis extraction near pi, where sin(theta) is nearly zero.
        eigvals, eigvecs = np.linalg.eigh(0.5 * (R + np.eye(3)))
        axis = eigvecs[:, np.argmax(eigvals)]
        return theta * axis
    return (theta / (2.0 * np.sin(theta))) * vee


def rotation_distance_rad(R_a, R_b):
    return float(np.linalg.norm(_so3_log(np.asarray(R_a).T @ np.asarray(R_b))))


def fuse_relative_poses(T_visual, T_icp, visual_score, icp_score,
                        max_translation_disagreement_m=0.25,
                        max_rotation_disagreement_deg=12.0):
    """Fuse two T_previous_current estimates with quality and disagreement gates.

    A score is a non-negative relative confidence weight. If estimates disagree
    beyond configured gates, only the higher-scoring estimate is accepted.
    If both estimates are consistent, interpolate translation and SO(3)
    orientation using normalized quality weights.
    """
    if T_visual is None and T_icp is None:
        raise ValueError("at least one motion estimate is required")
    if T_visual is not None:
        validate_se3(T_visual)
    if T_icp is not None:
        validate_se3(T_icp)
    vs, ins = max(0.0, float(visual_score)), max(0.0, float(icp_score))
    if T_visual is None:
        return np.asarray(T_icp).copy(), False, True, float("inf"), float("inf"), "ICP only"
    if T_icp is None:
        return np.asarray(T_visual).copy(), True, False, float("inf"), float("inf"), "visual odometry only"

    tv, ti = np.asarray(T_visual), np.asarray(T_icp)
    translation_gap = float(np.linalg.norm(tv[:3, 3] - ti[:3, 3]))
    rotation_gap = np.rad2deg(rotation_distance_rad(tv[:3, :3], ti[:3, :3]))
    conflict = (translation_gap > max_translation_disagreement_m or
                rotation_gap > max_rotation_disagreement_deg)
    if conflict:
        if vs >= ins:
            return tv.copy(), True, False, translation_gap, rotation_gap, "disagreement: visual estimate selected"
        return ti.copy(), False, True, translation_gap, rotation_gap, "disagreement: ICP estimate selected"

    total = vs + ins
    if total <= 1e-12:
        alpha = 0.5
    else:
        alpha = ins / total
    # Geodesic interpolation on SO(3): R = Rv Exp(alpha Log(Rv^T Ri)).
    relative_rotation = tv[:3, :3].T @ ti[:3, :3]
    R = tv[:3, :3] @ _so3_exp(alpha * _so3_log(relative_rotation))
    # Both translations are expressed in the same previous-camera frame.
    t = (1.0 - alpha) * tv[:3, 3] + alpha * ti[:3, 3]
    return make_se3(R, t), True, True, translation_gap, rotation_gap, "fused"


class RGBDVisualICPFusion:
    """Sequential local pose fusion and map accumulation.

    Call initialize() with the first frame's camera-frame cloud, then integrate()
    each next frame with its VisualOdometryResult and camera-frame point cloud.
    """
    def __init__(self, registration_voxel_m=0.04, map_voxel_m=0.03,
                 max_correspondence_m=0.18, max_icp_iterations=35,
                 min_correspondences=30, icp_good_rmse_m=0.025,
                 max_translation_disagreement_m=0.25,
                 max_rotation_disagreement_deg=12.0,
                 max_map_points=250000):
        if min(registration_voxel_m, map_voxel_m, max_correspondence_m,
               icp_good_rmse_m) <= 0:
            raise ValueError("voxel sizes, correspondence distance and RMSE scale must be positive")
        self.registration_voxel_m = registration_voxel_m
        self.map_voxel_m = map_voxel_m
        self.max_correspondence_m = max_correspondence_m
        self.max_icp_iterations = max_icp_iterations
        self.min_correspondences = min_correspondences
        self.icp_good_rmse_m = icp_good_rmse_m
        self.max_translation_disagreement_m = max_translation_disagreement_m
        self.max_rotation_disagreement_deg = max_rotation_disagreement_deg
        self.max_map_points = max_map_points
        self.reset()

    def reset(self):
        self.previous_cloud = None
        self.T_world_previous = np.eye(4)
        self.map_points_world = np.empty((0, 3), dtype=np.float64)
        self.frame_count = 0

    def initialize(self, points_camera):
        points = np.asarray(points_camera, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("point cloud must be finite Nx3")
        points = voxel_downsample_points(points, self.registration_voxel_m)
        if len(points) < self.min_correspondences:
            raise ValueError("not enough points after registration downsampling")
        self.previous_cloud = points.copy()
        self.map_points_world = voxel_downsample_points(points, self.map_voxel_m)
        self.T_world_previous = np.eye(4)
        self.frame_count = 1
        return self.map_points_world.copy()

    @staticmethod
    def _visual_quality(result):
        if result is None or not getattr(result, "success", False):
            return None, 0.0
        T_current_previous = getattr(result, "T_current_previous", None)
        if T_current_previous is None:
            return None, 0.0
        T_previous_current = invert_se3(T_current_previous)
        matches = max(1, int(getattr(result, "feature_matches", 0)))
        inliers = max(0, int(getattr(result, "pnp_inliers", 0)))
        rmse = float(getattr(result, "reprojection_rmse_px", np.inf))
        if not np.isfinite(rmse):
            return None, 0.0
        inlier_ratio = np.clip(inliers / matches, 0.0, 1.0)
        score = float(inlier_ratio * np.exp(-rmse / 4.0))
        return T_previous_current, score

    def _icp_quality(self, result):
        correspondence_ratio = min(
            1.0, result.correspondences / max(self.min_correspondences * 4, 1))
        convergence_factor = 1.0 if result.converged else 0.55
        score = float(correspondence_ratio * convergence_factor *
                      np.exp(-result.rmse_m / self.icp_good_rmse_m))
        return score

    def integrate(self, current_points_camera, visual_result=None):
        current = np.asarray(current_points_camera, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] != 3 or not np.isfinite(current).all():
            raise ValueError("current cloud must be finite Nx3")
        current = voxel_downsample_points(current, self.registration_voxel_m)
        if len(current) < self.min_correspondences:
            raise ValueError("not enough current points after downsampling")
        if self.previous_cloud is None:
            self.initialize(current)
            return PoseFusionResult(
                True, np.eye(4), np.eye(4), None, None, 0., 0.,
                False, False, 0., 0., 0., 0, "initialized")

        T_visual, visual_score = self._visual_quality(visual_result)
        icp_result = None
        try:
            initial = T_visual if T_visual is not None else np.eye(4)
            icp_result = iterative_closest_point(
                current, self.previous_cloud,
                initial_T_target_source=initial,
                max_correspondence_distance_m=self.max_correspondence_m,
                max_iterations=self.max_icp_iterations,
                trim_fraction=0.85,
                min_correspondences=self.min_correspondences)
            T_icp = icp_result.T_target_source
            icp_score = self._icp_quality(icp_result)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            T_icp, icp_score = None, 0.0

        if T_visual is None and T_icp is None:
            return PoseFusionResult(
                False, np.eye(4), self.T_world_previous.copy(), None, None,
                0., 0., False, False, float("inf"), float("inf"),
                float("inf"), 0, "both visual odometry and ICP failed; frame rejected")

        T_previous_current, visual_used, icp_used, t_gap, r_gap, message = fuse_relative_poses(
            T_visual, T_icp, visual_score, icp_score,
            self.max_translation_disagreement_m,
            self.max_rotation_disagreement_deg)
        T_world_current = compose_se3(self.T_world_previous, T_previous_current)

        # Integrate the accepted pose. The local map is a bounded voxel cloud;
        # this is not a TSDF, surfel map, loop-closed map, or dense mesh.
        aligned_world = transform_points(current, T_world_current)
        self.map_points_world = voxel_downsample_points(
            np.vstack([self.map_points_world, aligned_world]), self.map_voxel_m)
        if len(self.map_points_world) > self.max_map_points:
            indices = np.linspace(0, len(self.map_points_world)-1,
                                  self.max_map_points, dtype=np.int64)
            self.map_points_world = self.map_points_world[indices]
        self.previous_cloud = current.copy()
        self.T_world_previous = T_world_current
        self.frame_count += 1

        return PoseFusionResult(
            True, T_previous_current, T_world_current.copy(),
            T_visual, T_icp, visual_score, icp_score,
            visual_used, icp_used, t_gap, r_gap,
            icp_result.rmse_m if icp_result is not None else float("inf"),
            icp_result.correspondences if icp_result is not None else 0,
            message)
