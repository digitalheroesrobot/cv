#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 52 — MULTI-CAMERA EXTRINSIC CALIBRATION AND RIG OPTIMIZATION

LEARNING OBJECTIVES
-------------------
1. Understand T_rig_camera and pairwise T_camera_j_camera_i conventions.
2. Build a connected camera calibration graph from pairwise stereo results.
3. Fix one camera as the gauge anchor and optimize all other camera poses.
4. Weight pairwise measurements by translation/rotation uncertainty.
5. Use robust nonlinear least squares to reduce the effect of outlier edges.
6. Inspect per-edge translation and rotation residuals and approximate covariance.

The default demo is deterministic and offline. Real checkerboard calibration
can reuse multi_camera.calibration.calibrate_stereo_from_chessboards and then
convert each StereoCalibrationResult with stereo_result_to_rig_edge.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common import make_se3, invert_se3
from multi_camera import RigEdge, RigPoseGraph


def pose_xyz_ypr(x, y, z, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    R = Rotation.from_euler("zyx", [yaw_deg, pitch_deg, roll_deg],
                            degrees=True).as_matrix()
    return make_se3(R, np.array([x, y, z], dtype=float))


def relative_measurement(T_rig_i, T_rig_j):
    """Return T_camera_j_camera_i: camera-i coordinates -> camera-j."""
    return invert_se3(T_rig_j) @ T_rig_i


def make_synthetic_graph(seed=52):
    rng = np.random.default_rng(seed)
    truth = {
        "cam0": np.eye(4),
        "cam1": pose_xyz_ypr(0.24, 0.0, 0.015, 1.0, 0.2, -0.3),
        "cam2": pose_xyz_ypr(0.0, 0.18, 0.025, -0.5, 0.8, 0.1),
        "cam3": pose_xyz_ypr(0.22, 0.17, 0.04, 0.7, -0.4, 0.5),
    }
    pairs = [
        ("cam0", "cam1"), ("cam0", "cam2"),
        ("cam1", "cam3"), ("cam2", "cam3"),
        ("cam0", "cam3"),
    ]
    edges = []
    for i, j in pairs:
        measured = relative_measurement(truth[i], truth[j])
        # Small realistic measurement noise in rotation and translation.
        noisy = measured.copy()
        noisy[:3, :3] = measured[:3, :3] @ Rotation.from_rotvec(
            rng.normal(0, np.deg2rad(0.04), 3)).as_matrix()
        noisy[:3, 3] += rng.normal(0, 0.0004, 3)
        edges.append(RigEdge(i, j, noisy,
                             translation_sigma_m=0.002,
                             rotation_sigma_rad=np.deg2rad(0.15)))
    # A deliberately biased loop measurement tests robust loss.
    outlier = relative_measurement(truth["cam0"], truth["cam3"]).copy()
    outlier[:3, 3] += np.array([0.018, -0.012, 0.006])
    edges.append(RigEdge("cam0", "cam3", outlier,
                         translation_sigma_m=0.003,
                         rotation_sigma_rad=np.deg2rad(0.2),
                         source="deliberate_outlier"))
    # Perturb initial poses so optimization has a meaningful starting error.
    initial = {"cam0": np.eye(4)}
    for camera_id in ("cam1", "cam2", "cam3"):
        T = truth[camera_id].copy()
        T[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(
            rng.normal(0, np.deg2rad(1.5), 3)).as_matrix()
        T[:3, 3] += rng.normal(0, 0.012, 3)
        initial[camera_id] = T
    return truth, edges, initial


def pose_errors(estimate, truth):
    dT = invert_se3(truth) @ estimate
    angle = np.linalg.norm(Rotation.from_matrix(dT[:3, :3]).as_rotvec())
    translation = np.linalg.norm(dT[:3, 3])
    return float(translation), float(angle)


def run_demo(seed=52, visualize=False):
    truth, edges, initial = make_synthetic_graph(seed)
    graph = RigPoseGraph(list(truth), edges, anchor_camera="cam0")
    result = graph.optimize(initial_poses=initial, robust_loss="soft_l1")
    errors = {}
    for camera_id in truth:
        dt, dr = pose_errors(result.T_rig_camera[camera_id], truth[camera_id])
        errors[camera_id] = {
            "translation_error_m": dt,
            "rotation_error_deg": float(np.rad2deg(dr))
        }
    report = {
        "lesson": 52,
        "offline": True,
        "success": result.success,
        "message": result.message,
        "iterations": result.iterations,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "anchor_camera": "cam0",
        "camera_pose_errors": errors,
        "edge_translation_residuals_m": result.edge_translation_errors_m,
        "edge_rotation_residuals_rad": result.edge_rotation_errors_rad,
        "T_rig_camera": {k: v.tolist() for k, v in result.T_rig_camera.items()},
        "covariance_shape": list(result.covariance.shape) if result.covariance is not None else None,
        "limitations": [
            "Synthetic experiment validates graph conventions and optimizer, not physical calibration accuracy.",
            "Real pairwise edges must come from synchronized target observations and calibrated intrinsics.",
            "Approximate covariance is a local inverse-Hessian estimate and depends on residual scaling and robust loss.",
            "A connected graph is required; weak geometry or poor target coverage can still make calibration ill-conditioned."
        ]
    }
    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    (out / "lesson52_camera_rig_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if visualize:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for camera_id, T in truth.items():
            p = T[:3, 3]
            ax.scatter(*p, marker="o", s=75, label=f"{camera_id} truth")
            q = result.T_rig_camera[camera_id][:3, 3]
            ax.scatter(*q, marker="x", s=75, label=f"{camera_id} estimate")
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], linewidth=1)
        ax.set_xlabel("Rig X (m)")
        ax.set_ylabel("Rig Y (m)")
        ax.set_zlabel("Rig Z (m)")
        ax.set_title("Lesson 52 — Camera Rig Extrinsic Optimization")
        ax.legend(fontsize=7)
        plt.tight_layout()
        plt.show()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    report = run_demo(args.seed, args.visualize)
    print(json.dumps({
        "success": report["success"],
        "initial_cost": report["initial_cost"],
        "final_cost": report["final_cost"],
        "camera_pose_errors": report["camera_pose_errors"],
        "report": "outputs/lesson52_camera_rig_report.json"
    }, indent=2))


if __name__ == "__main__":
    main()
