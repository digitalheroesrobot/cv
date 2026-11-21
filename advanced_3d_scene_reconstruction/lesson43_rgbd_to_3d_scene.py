#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 43 — RGB-D TO 3D SCENE RECONSTRUCTION

PURPOSE
-------
Convert an aligned metric RGB-D frame into a 3D point cloud, remove noise,
detect a dominant ground plane, cluster objects, and estimate 3D boxes.

LEARNING OBJECTIVES
-------------------
1. Understand optical-axis Z depth and pinhole back-projection.
2. Build metric point clouds in metres.
3. Separate a dominant ground plane using RANSAC.
4. Cluster spatially connected points and estimate PCA-oriented boxes.
5. Transform camera geometry into a fixed world frame.

RUN
---
Synthetic demo:
    python lesson43_rgbd_to_3d_scene.py --visualize

RealSense D435:
    python lesson43_rgbd_to_3d_scene.py --realsense

The RealSense path requires pyrealsense2. This is a classical geometry
baseline, not semantic object recognition or a neural reconstruction model.
"""
from __future__ import annotations
import argparse
import time
import numpy as np
from common import CameraCalibration, DepthMap
from scene_reconstruction import SceneReconstructionPipeline


def make_synthetic_rgbd(width=640, height=480):
    """Create a synthetic floor and two depth patches for pipeline testing."""
    fx = fy = 600.0
    cx, cy = width / 2, height / 2
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    yy, xx = np.mgrid[0:height, 0:width]
    z = np.full((height, width), 2.8, dtype=np.float32)
    # Plane in camera coordinates: y - 0.35*z = 1.2.
    denom = (yy - cy) / fy - 0.35
    valid_floor = denom > 0.03
    z_floor = np.zeros_like(z)
    z_floor[valid_floor] = 1.2 / denom[valid_floor]
    z[valid_floor] = z_floor[valid_floor]
    z[210:300, 180:260] = 2.15
    z[190:285, 360:455] = 2.35
    z[(z < 0.3) | (z > 7.5)] = 0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:] = [100, 110, 120]
    rgb[210:300, 180:260] = [220, 80, 50]
    rgb[190:285, 360:455] = [50, 180, 80]
    calibration = CameraCalibration(K, width, height, calibration_id="synthetic")
    depth = DepthMap(z, time.monotonic(), "synthetic_camera")
    return rgb, depth, calibration


def visualize(result, title="RGB-D Scene Reconstruction"):
    """Show a sampled 3D cloud and PCA bounding boxes (Matplotlib optional)."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise RuntimeError("Install matplotlib: python -m pip install matplotlib") from exc
    points = result.camera_cloud.points
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(points):
        sample = points[::max(1, len(points) // 12000)]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1, alpha=0.35)
    for box in result.boxes:
        c = box.corners
        faces = [[c[0], c[1], c[3], c[2]], [c[4], c[5], c[7], c[6]],
                 [c[0], c[1], c[5], c[4]], [c[2], c[3], c[7], c[6]],
                 [c[0], c[2], c[6], c[4]], [c[1], c[3], c[7], c[5]]]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.12, linewidths=1))
        ax.scatter(*box.center, s=35)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def run_realsense(args):
    from camera_adapters.realsense_d435 import RealSenseD435Adapter
    engine = SceneReconstructionPipeline(
        voxel_size_m=args.voxel, stride=args.stride,
        min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        cluster_tolerance_m=args.cluster_tolerance,
        min_cluster_points=args.min_cluster_points)
    with RealSenseD435Adapter(width=args.width, height=args.height, fps=args.fps) as camera:
        print("RealSense D435 active. Press Ctrl+C to stop.")
        try:
            while True:
                sample = camera.read()
                if sample is None or sample.depth is None or sample.calibration is None:
                    continue
                result = engine.process(sample.depth, sample.calibration, sample.frame.image)
                print(f"points={len(result.camera_cloud.points):6d} "
                      f"clusters={len(result.clusters):2d} boxes={len(result.boxes):2d}")
                if args.visualize:
                    visualize(result, "D435 RGB-D frame")
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--voxel", type=float, default=0.025)
    parser.add_argument("--min-depth", type=float, default=0.2)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--cluster-tolerance", type=float, default=0.10)
    parser.add_argument("--min-cluster-points", type=int, default=12)
    args = parser.parse_args()
    if args.realsense:
        run_realsense(args)
        return
    rgb, depth, calibration = make_synthetic_rgbd(args.width, args.height)
    engine = SceneReconstructionPipeline(
        voxel_size_m=args.voxel, stride=args.stride,
        min_depth_m=args.min_depth, max_depth_m=args.max_depth,
        cluster_tolerance_m=args.cluster_tolerance,
        min_cluster_points=args.min_cluster_points)
    result = engine.process(depth, calibration, rgb)
    print("RGB-D scene reconstruction")
    print(f"Camera points: {len(result.camera_cloud.points)}")
    print(f"Ground plane detected: {result.ground_plane is not None}")
    if result.ground_plane is not None:
        print(f"Ground RMSE: {result.ground_plane.rmse_m:.5f} m")
    print(f"Object clusters: {len(result.clusters)}")
    for i, box in enumerate(result.boxes):
        print(f"Box {i}: center={np.round(box.center, 3)} m, "
              f"dimensions={np.round(box.dimensions, 3)} m")
    if args.visualize:
        visualize(result)


if __name__ == "__main__":
    main()
