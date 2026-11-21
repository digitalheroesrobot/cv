#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 51 — MULTI-CAMERA RGB-D WORLD RECONSTRUCTION

LEARNING OBJECTIVES
-------------------
1. Synchronize camera observations using an explicit common clock domain.
2. Rectify each registered RGB-D stream using its own intrinsics/distortion.
3. Back-project each depth image into its own optical camera coordinates.
4. Transform clouds with calibrated T_world_camera extrinsics.
5. Fuse overlapping observations and reduce duplicate voxels.
6. Quantify synchronization skew and per-camera point contributions.

The default synthetic experiment is deterministic and fully offline.
Real-camera integration is exposed through the reusable MultiCameraReconstructor
API; users must provide measured extrinsics and a trustworthy common timestamp
domain. The lesson never guesses camera baselines or extrinsic poses.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common import CameraCalibration, CameraFrame, DepthMap
from camera_adapters import CameraSample
from multi_camera import MultiCameraReconstructor


def make_pose(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=float)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=float)
    return T


def make_synthetic_samples(seed=51, width=320, height=240):
    """Render the same fronto-parallel world plane from two virtual cameras."""
    rng = np.random.default_rng(seed)
    fx = fy = 260.0
    cx, cy = width / 2.0, height / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    calibration = CameraCalibration(K, width, height, np.zeros(5))
    # Camera B is 20 cm to the right in world coordinates.
    poses = {"cam0": np.eye(4), "cam1": make_pose(t=[0.20, 0.0, 0.0])}
    samples = {}
    timestamp = 10.0
    for camera_id, T_world_camera in poses.items():
        # Plane lies at world Z=2m. Transform its points into camera coordinates.
        yy, xx = np.mgrid[0:height:2, 0:width:2]
        z = np.full(xx.shape, 2.0, dtype=np.float32)
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        world = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        T_camera_world = np.linalg.inv(T_world_camera)
        camera = world @ T_camera_world[:3, :3].T + T_camera_world[:3, 3]
        u = np.rint(fx * camera[:, 0] / camera[:, 2] + cx).astype(int)
        v = np.rint(fy * camera[:, 1] / camera[:, 2] + cy).astype(int)
        depth = np.zeros((height, width), dtype=np.float32)
        valid = ((u >= 0) & (u < width) & (v >= 0) & (v < height) &
                 (camera[:, 2] > 0))
        depth[v[valid], u[valid]] = camera[valid, 2]
        # Fill holes with the exact plane depth where rays intersect the plane.
        depth[depth == 0] = 2.0
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = (xx.repeat(2, axis=0)[:height, :width] if False else 80)
        image[..., 1] = 120
        image[..., 2] = 180
        # Add local texture so the two inputs are not visually identical constants.
        image[..., 0] = rng.integers(50, 180, size=(height, width), dtype=np.uint8)
        frame = CameraFrame(image, timestamp + (0.003 if camera_id == "cam1" else 0),
                            camera_id, 0, "BGR")
        depth_map = DepthMap(depth, frame.timestamp, camera_id, unit="m")
        samples[camera_id] = CameraSample(frame, calibration, depth_map)
    return samples, poses


def save_ply(path, points, colors=None):
    points = np.asarray(points, dtype=float)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if colors is None:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            rgb = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(int)
            for p, c in zip(points, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[2]} {c[1]} {c[0]}\n")


def run_demo(seed=51, visualize=False):
    samples, poses = make_synthetic_samples(seed)
    engine = MultiCameraReconstructor(
        poses, max_timestamp_skew_s=0.02, stride=2,
        min_depth_m=0.2, max_depth_m=4.0, voxel_size_m=0.015)
    result = engine.process(samples)
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    save_ply(output / "lesson51_multi_camera_world.ply",
             result.fused_cloud.points, result.fused_cloud.colors)
    report = {
        "lesson": 51,
        "offline": True,
        "camera_ids": sorted(samples),
        "timestamp_s": result.timestamp,
        "max_timestamp_skew_s": result.max_skew_s,
        "per_camera_points": result.per_camera_points,
        "fused_world_points": len(result.fused_cloud.points),
        "world_frame": result.fused_cloud.frame_id,
        "T_world_camera": {k: v.tolist() for k, v in poses.items()},
        "artifacts": ["outputs/lesson51_multi_camera_world.ply"],
        "limitations": [
            "Synthetic scene uses known extrinsics; it does not estimate camera rig calibration.",
            "Software timestamp pairing requires a shared clock domain and does not provide hardware synchronization.",
            "The point-cloud fusion stage assumes calibrated extrinsics and does not estimate them.",
            "Only pinhole distortion rectification is supported in this lesson implementation.",
            "A flat synthetic plane validates transform direction and fusion plumbing, not real-world reconstruction quality."
        ]
    }
    (output / "lesson51_multi_camera_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if visualize:
        import matplotlib.pyplot as plt
        p = result.fused_cloud.points
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        if len(p) > 30000:
            idx = np.linspace(0, len(p) - 1, 30000, dtype=int)
            p = p[idx]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1)
        for camera_id, T in poses.items():
            ax.scatter(*T[:3, 3], s=70, marker="^", label=camera_id)
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_zlabel("World Z (m)")
        ax.set_title("Lesson 51 — Multi-Camera World Point Cloud")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    report = run_demo(args.seed, args.visualize)
    print(json.dumps({
        "camera_ids": report["camera_ids"],
        "max_timestamp_skew_s": report["max_timestamp_skew_s"],
        "per_camera_points": report["per_camera_points"],
        "fused_world_points": report["fused_world_points"],
        "artifacts": report["artifacts"]
    }, indent=2))


if __name__ == "__main__":
    main()
