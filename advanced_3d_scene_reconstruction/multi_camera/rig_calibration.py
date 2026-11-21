#!/usr/bin/env python3
"""Multi-camera rig calibration from pairwise metric camera transforms.

Convention:
    T_rig_camera[i] maps camera-i coordinates into the rig frame.
    RigEdge.T_camera_j_camera_i maps camera-i coordinates into camera-j.

For an edge i -> j, the predicted measurement is:
    inv(T_rig_camera[j]) @ T_rig_camera[i]

The first camera is fixed as the gauge anchor. This optimizer estimates
extrinsics only; intrinsic calibration and target detection are separate.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from common import make_se3, invert_se3, validate_se3
from .calibration import StereoCalibrationResult


@dataclass
class RigEdge:
    camera_i: str
    camera_j: str
    T_camera_j_camera_i: np.ndarray
    translation_sigma_m: float = 0.005
    rotation_sigma_rad: float = np.deg2rad(0.5)
    source: str = "stereo_calibration"

    def __post_init__(self):
        if self.camera_i == self.camera_j:
            raise ValueError("Rig edge must connect two different cameras")
        self.T_camera_j_camera_i = np.asarray(
            self.T_camera_j_camera_i, dtype=np.float64).copy()
        validate_se3(self.T_camera_j_camera_i)
        if (not np.isfinite(self.translation_sigma_m) or
                not np.isfinite(self.rotation_sigma_rad) or
                self.translation_sigma_m <= 0 or self.rotation_sigma_rad <= 0):
            raise ValueError("Edge uncertainty must be finite and positive")


@dataclass
class RigCalibrationResult:
    T_rig_camera: Dict[str, np.ndarray]
    success: bool
    message: str
    iterations: int
    initial_cost: float
    final_cost: float
    edge_translation_errors_m: Dict[str, float]
    edge_rotation_errors_rad: Dict[str, float]
    covariance: Optional[np.ndarray]
    camera_order: Tuple[str, ...]


def _pose_to_vector(T):
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def _vector_to_pose(x):
    return make_se3(Rotation.from_rotvec(x[:3]).as_matrix(), x[3:6])


def _se3_error(T_measured, T_predicted):
    E = invert_se3(T_measured) @ T_predicted
    return np.r_[Rotation.from_matrix(E[:3, :3]).as_rotvec(), E[:3, 3]]


class RigPoseGraph:
    """Robust least-squares optimization over pairwise camera extrinsics."""
    def __init__(self, camera_ids: Sequence[str], edges: Sequence[RigEdge],
                 anchor_camera: Optional[str] = None):
        ids = tuple(camera_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("At least two unique camera IDs are required")
        self.camera_ids = ids
        self.anchor = anchor_camera or ids[0]
        if self.anchor not in ids:
            raise ValueError("Anchor camera is not in camera_ids")
        self.edges = list(edges)
        if not self.edges:
            raise ValueError("At least one pairwise calibration edge is required")
        for edge in self.edges:
            if edge.camera_i not in ids or edge.camera_j not in ids:
                raise ValueError("Edge references an unknown camera")
        self._validate_connected()

    def _validate_connected(self):
        adjacency = {c: set() for c in self.camera_ids}
        for e in self.edges:
            adjacency[e.camera_i].add(e.camera_j)
            adjacency[e.camera_j].add(e.camera_i)
        visited, stack = set(), [self.anchor]
        while stack:
            c = stack.pop()
            if c in visited:
                continue
            visited.add(c)
            stack.extend(adjacency[c] - visited)
        if len(visited) != len(self.camera_ids):
            raise ValueError("Camera calibration graph is disconnected")

    def _initial_poses(self):
        # Propagate measured transforms from the anchor. If edge i->j maps
        # camera-i points into camera-j, then T_rig_camera_j =
        # T_rig_camera_i @ inverse(T_camera_j_camera_i).
        poses = {self.anchor: np.eye(4)}
        pending = list(self.edges)
        changed = True
        while pending and changed:
            changed = False
            rest = []
            for e in pending:
                if e.camera_i in poses and e.camera_j not in poses:
                    poses[e.camera_j] = poses[e.camera_i] @ invert_se3(e.T_camera_j_camera_i)
                    changed = True
                elif e.camera_j in poses and e.camera_i not in poses:
                    poses[e.camera_i] = poses[e.camera_j] @ e.T_camera_j_camera_i
                    changed = True
                else:
                    rest.append(e)
            pending = rest
        if len(poses) != len(self.camera_ids):
            raise RuntimeError("Unable to initialize all camera poses")
        return poses

    def optimize(self, initial_poses: Optional[Mapping[str, np.ndarray]] = None,
                 max_nfev: int = 300, robust_loss: str = "soft_l1",
                 f_scale: float = 1.0) -> RigCalibrationResult:
        init = self._initial_poses() if initial_poses is None else {
            k: np.asarray(v, dtype=np.float64).copy()
            for k, v in initial_poses.items()}
        if set(init) != set(self.camera_ids):
            raise ValueError("Initial poses must contain every camera")
        for T in init.values():
            validate_se3(T)
        variables = [c for c in self.camera_ids if c != self.anchor]
        x0 = np.concatenate([_pose_to_vector(init[c]) for c in variables])
        def unpack(x):
            poses = {self.anchor: np.eye(4)}
            for k, c in enumerate(variables):
                poses[c] = _vector_to_pose(x[6*k:6*k+6])
            return poses
        def residual(x):
            poses = unpack(x)
            blocks = []
            for e in self.edges:
                predicted = invert_se3(poses[e.camera_j]) @ poses[e.camera_i]
                r = _se3_error(e.T_camera_j_camera_i, predicted)
                r[:3] /= e.rotation_sigma_rad
                r[3:] /= e.translation_sigma_m
                blocks.append(r)
            return np.concatenate(blocks)
        r0 = residual(x0)
        initial_cost = float(0.5 * r0 @ r0)
        result = least_squares(
            residual, x0, method="trf", loss=robust_loss,
            f_scale=float(f_scale), max_nfev=int(max_nfev),
            x_scale="jac")
        poses = unpack(result.x)
        trans_errors, rot_errors = {}, {}
        for index, e in enumerate(self.edges):
            predicted = invert_se3(poses[e.camera_j]) @ poses[e.camera_i]
            error = _se3_error(e.T_camera_j_camera_i, predicted)
            key = f"{e.camera_i}->{e.camera_j}#{index}"
            rot_errors[key] = float(np.linalg.norm(error[:3]))
            trans_errors[key] = float(np.linalg.norm(error[3:]))
        covariance = None
        if result.jac.size and result.jac.shape[0] >= result.jac.shape[1]:
            try:
                information = result.jac.T @ result.jac
                covariance = np.linalg.pinv(information, rcond=1e-10)
                covariance = 0.5 * (covariance + covariance.T)
            except np.linalg.LinAlgError:
                covariance = None
        return RigCalibrationResult(
            T_rig_camera=poses, success=bool(result.success),
            message=str(result.message), iterations=int(result.nfev),
            initial_cost=initial_cost,
            final_cost=float(0.5 * result.fun @ result.fun),
            edge_translation_errors_m=trans_errors,
            edge_rotation_errors_rad=rot_errors,
            covariance=covariance, camera_order=tuple(variables))


def stereo_result_to_rig_edge(camera_i: str, camera_j: str,
                              result: StereoCalibrationResult,
                              translation_sigma_m=0.005,
                              rotation_sigma_rad=np.deg2rad(0.5)) -> RigEdge:
    """Convert stereoCalibrate output into the graph's explicit edge convention."""
    return RigEdge(
        camera_i, camera_j, result.T_right_left,
        translation_sigma_m, rotation_sigma_rad, "opencv_stereo_calibrate")


def calibrate_camera_rig(camera_ids: Sequence[str], edges: Sequence[RigEdge],
                         anchor_camera: Optional[str] = None,
                         initial_poses: Optional[Mapping[str, np.ndarray]] = None):
    """Convenience wrapper returning the optimized rig result."""
    return RigPoseGraph(camera_ids, edges, anchor_camera).optimize(initial_poses)
