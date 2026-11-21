#!/usr/bin/env python3
"""RGB-D geometric loop verification and globally corrected point-map rebuild."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from common.geometry import invert_se3, transform_points
from temporal_mapping.odometry import iterative_closest_point, voxel_downsample_points


@dataclass
class LoopVerificationResult:
    accepted: bool
    T_candidate_current: Optional[np.ndarray]
    pnp_available: bool
    icp_converged: bool
    icp_rmse_m: float
    icp_correspondences: int
    translation_disagreement_m: float
    rotation_disagreement_rad: float
    reason: str


def _rotation_gap(A, B):
    from scipy.spatial.transform import Rotation
    return float(Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())


class RGBDLoopClosureVerifier:
    """Verify a candidate using PnP initialization, ICP refinement, and consistency.

    PnP convention is T_current_candidate. ICP takes current points as source
    and candidate points as target, returning T_candidate_current.
    """
    def __init__(self, voxel_size_m=.025, max_correspondence_m=.10,
                 min_correspondences=30, max_rmse_m=.035,
                 max_translation_gap_m=.12, max_rotation_gap_rad=.12):
        values = [voxel_size_m, max_correspondence_m, max_rmse_m,
                  max_translation_gap_m, max_rotation_gap_rad]
        if min(values) <= 0 or min_correspondences < 3:
            raise ValueError("Invalid loop verifier thresholds")
        self.voxel_size_m = float(voxel_size_m)
        self.max_correspondence_m = float(max_correspondence_m)
        self.min_correspondences = int(min_correspondences)
        self.max_rmse_m = float(max_rmse_m)
        self.max_translation_gap_m = float(max_translation_gap_m)
        self.max_rotation_gap_rad = float(max_rotation_gap_rad)

    def verify_cloud_pair(self, candidate_cloud, current_cloud,
                          T_current_candidate):
        candidate = np.asarray(candidate_cloud, dtype=np.float64)
        current = np.asarray(current_cloud, dtype=np.float64)
        if (candidate.ndim != 2 or current.ndim != 2 or
                candidate.shape[1:] != (3,) or current.shape[1:] != (3,) or
                not np.isfinite(candidate).all() or not np.isfinite(current).all()):
            raise ValueError("Point clouds must be finite Nx3 arrays")
        if T_current_candidate is None:
            return LoopVerificationResult(False, None, False, False, float("inf"),
                                          0, float("inf"), float("inf"),
                                          "RGB-D PnP initialization is required")
        T_pnp_candidate_current = invert_se3(np.asarray(T_current_candidate, dtype=float))
        try:
            source = voxel_downsample_points(current, self.voxel_size_m)
            target = voxel_downsample_points(candidate, self.voxel_size_m)
            icp = iterative_closest_point(
                source, target,
                initial_T_target_source=T_pnp_candidate_current,
                max_correspondence_distance_m=self.max_correspondence_m,
                max_iterations=60, tolerance_m=1e-6,
                trim_fraction=.8, min_correspondences=self.min_correspondences)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            return LoopVerificationResult(False, None, True, False, float("inf"),
                                          0, float("inf"), float("inf"),
                                          "ICP failed: " + str(exc))
        dt = float(np.linalg.norm(
            icp.T_target_source[:3, 3] - T_pnp_candidate_current[:3, 3]))
        dr = _rotation_gap(T_pnp_candidate_current, icp.T_target_source)
        accepted = (icp.converged and icp.correspondences >= self.min_correspondences
                    and icp.rmse_m <= self.max_rmse_m
                    and dt <= self.max_translation_gap_m
                    and dr <= self.max_rotation_gap_rad)
        if not icp.converged:
            reason = "ICP did not converge"
        elif icp.rmse_m > self.max_rmse_m:
            reason = "ICP RMSE exceeds threshold"
        elif dt > self.max_translation_gap_m or dr > self.max_rotation_gap_rad:
            reason = "PnP and ICP estimates disagree"
        else:
            reason = "geometrically verified"
        return LoopVerificationResult(
            accepted, icp.T_target_source.copy() if accepted else None,
            True, icp.converged, float(icp.rmse_m), int(icp.correspondences),
            dt, dr, reason)


def rebuild_global_map(keyframes, optimized_poses, voxel_size_m=.03,
                       max_points=500000):
    """Reproject stored keyframe-local clouds using optimized world poses.

    This is a point-cloud reintegration baseline. It does not perform TSDF
    fusion, visibility reasoning, color blending, or dynamic-object filtering.
    """
    poses = list(optimized_poses)
    frames = list(keyframes)
    if len(frames) != len(poses):
        raise ValueError("Each keyframe must have exactly one optimized pose")
    clouds = []
    for frame, T_world_camera in zip(frames, poses):
        if getattr(frame, "point_cloud", None) is None:
            continue
        cloud = np.asarray(frame.point_cloud, dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError("Keyframe point_cloud must be finite Nx3")
        clouds.append(transform_points(cloud, T_world_camera))
    if not clouds:
        return np.empty((0, 3), dtype=np.float64)
    merged = voxel_downsample_points(np.vstack(clouds), voxel_size_m)
    if len(merged) > max_points:
        ids = np.linspace(0, len(merged)-1, max_points, dtype=np.int64)
        merged = merged[ids]
    return merged
