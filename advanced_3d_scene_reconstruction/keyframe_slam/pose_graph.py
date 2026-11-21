#!/usr/bin/env python3
"""A compact SE(3) pose graph with a fixed gauge anchor and robust least squares."""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _pose_to_vector(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError("Pose must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous bottom row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1., atol=1e-5):
        raise ValueError("Pose rotation must be in SO(3)")
    return np.r_[Rotation.from_matrix(R).as_rotvec(), T[:3, 3]]


def _vector_to_pose(x):
    x = np.asarray(x, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


def _invert(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


@dataclass
class PoseGraphEdge:
    """Measurement T_i_j maps coordinates from camera j into camera i."""
    i: int
    j: int
    T_i_j: np.ndarray
    translation_information: float = 1.0
    rotation_information: float = 1.0
    kind: str = "odometry"

    def __post_init__(self):
        self.i, self.j = int(self.i), int(self.j)
        self.T_i_j = np.asarray(self.T_i_j, dtype=float).copy()
        _pose_to_vector(self.T_i_j)
        if self.i == self.j:
            raise ValueError("A pose graph edge must connect different nodes")
        if self.translation_information <= 0 or self.rotation_information <= 0:
            raise ValueError("Edge information weights must be positive")


@dataclass
class PoseGraphResult:
    poses: List[np.ndarray]
    initial_cost: float
    final_cost: float
    success: bool
    message: str
    evaluations: int


class PoseGraph:
    """Optimize world-from-camera poses using relative SE(3) constraints.

    Node pose: T_world_camera_i.
    Edge measurement: T_i_j = inv(T_world_camera_i) @ T_world_camera_j.
    The first node is fixed to remove the six-degree-of-freedom gauge freedom.
    Residuals are rotation-vector and translation components in the edge frame.
    """
    def __init__(self):
        self.poses: List[np.ndarray] = []
        self.edges: List[PoseGraphEdge] = []

    def add_node(self, T_world_camera):
        pose = np.asarray(T_world_camera, dtype=float).copy()
        _pose_to_vector(pose)
        self.poses.append(pose)
        return len(self.poses) - 1

    def add_edge(self, edge: PoseGraphEdge):
        if max(edge.i, edge.j) >= len(self.poses) or min(edge.i, edge.j) < 0:
            raise IndexError("Edge references a node that does not exist")
        self.edges.append(edge)

    @staticmethod
    def _edge_residual(T_i_j_measured, T_world_i, T_world_j):
        predicted = _invert(T_world_i) @ T_world_j
        error = _invert(T_i_j_measured) @ predicted
        rot = Rotation.from_matrix(error[:3, :3]).as_rotvec()
        # Translation residual is expressed in the measured edge's i frame.
        return np.r_[rot, error[:3, 3]]

    def residual_vector(self, poses: Optional[List[np.ndarray]] = None):
        poses = self.poses if poses is None else poses
        chunks = []
        for edge in self.edges:
            r = self._edge_residual(edge.T_i_j, poses[edge.i], poses[edge.j])
            r[:3] *= np.sqrt(edge.rotation_information)
            r[3:] *= np.sqrt(edge.translation_information)
            chunks.append(r)
        return np.concatenate(chunks) if chunks else np.zeros(0)

    def optimize(self, max_nfev=200, loss="huber", f_scale=1.0):
        if len(self.poses) < 2 or not self.edges:
            raise ValueError("Pose graph needs at least two nodes and one edge")
        if max_nfev < 1 or f_scale <= 0:
            raise ValueError("max_nfev and f_scale must be positive")
        initial = [p.copy() for p in self.poses]
        initial_residual = self.residual_vector(initial)
        initial_cost = float(initial_residual @ initial_residual)
        # Fix node zero: optimizing it as well would leave global gauge undefined.
        x0 = np.concatenate([_pose_to_vector(p) for p in initial[1:]])

        def unpack(x):
            poses = [initial[0].copy()]
            for k in range(len(initial) - 1):
                poses.append(_vector_to_pose(x[k * 6:(k + 1) * 6]))
            return poses

        def fun(x):
            return self.residual_vector(unpack(x))

        result = least_squares(fun, x0, loss=loss, f_scale=f_scale,
                               max_nfev=max_nfev, method="trf")
        optimized = unpack(result.x)
        final_residual = self.residual_vector(optimized)
        final_cost = float(final_residual @ final_residual)
        self.poses = [p.copy() for p in optimized]
        return PoseGraphResult(
            poses=self.poses, initial_cost=initial_cost, final_cost=final_cost,
            success=bool(result.success), message=str(result.message),
            evaluations=int(result.nfev),
        )
