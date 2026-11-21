#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 56 — DYNAMIC 4D SCENE UNDERSTANDING

LEARNING OBJECTIVES
-------------------
1. Estimate point-level scene flow between ego-motion-compensated world clouds.
2. Separate stable background, moving points, and unobserved/ambiguous points.
3. Keep moving observations out of the persistent static map.
4. Cluster dynamic points into spatial components.
5. Use 3D tracked-object boxes as additional dynamic masks.
6. Visualize static map, dynamic observations, and point motion vectors.
7. Export per-frame 4D scene diagnostics.

Input point clouds must already be transformed into a stable world frame by
camera calibration and SLAM. Synthetic mode is deterministic and offline.
This is a classical nearest-neighbour scene-flow baseline, not a learned
scene-flow network or a complete semantic dynamic-scene SLAM system.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from multi_camera.dynamic_scene import DynamicSceneModel


def make_sequence(frames=45, seed=56):
    rng = np.random.default_rng(seed)
    # Static wall/floor samples in world coordinates.
    wall_x, wall_y = np.meshgrid(
        np.linspace(-1.2, 1.2, 45), np.linspace(-0.2, 1.8, 38))
    wall = np.column_stack((wall_x.ravel(), wall_y.ravel(),
                            np.full(wall_x.size, 3.0)))
    floor_x, floor_z = np.meshgrid(
        np.linspace(-1.2, 1.2, 35), np.linspace(1.0, 3.0, 22))
    floor = np.column_stack((floor_x.ravel(), np.full(floor_x.size, -0.35),
                             floor_z.ravel()))
    static = np.concatenate((wall, floor), axis=0)
    # Moving object: a compact 3D point sample translating along world X.
    local = rng.uniform([-0.18, -0.15, -0.12], [0.18, 0.15, 0.12], (240, 3))
    sequence = []
    for frame in range(frames):
        t = frame / 15.0
        moving = local + np.array([-0.8 + 0.035*frame, 0.25, 2.35])
        # Small independent sensor sampling noise; static world coordinates
        # remain stable because ego motion has already been compensated.
        observed_static = static + rng.normal(0, .002, static.shape)
        observed_moving = moving + rng.normal(0, .003, moving.shape)
        points = np.concatenate((observed_static, observed_moving), axis=0)
        # Deterministic thinning mimics depth-camera visibility and sampling.
        keep = rng.random(len(points)) > .06
        sequence.append((t, points[keep]))
    return sequence


def run_demo(frames=45, visualize=False, output_dir="outputs"):
    model = DynamicSceneModel(
        voxel_size_m=.035, max_static_points=80000,
        max_correspondence_distance_m=.045,
        static_speed_threshold_mps=.12,
        dynamic_cluster_voxel_m=.10, min_cluster_points=6)
    history = []
    final = None
    started = time.perf_counter()
    for timestamp, points in make_sequence(frames):
        final = model.update(points, timestamp)
        history.append({
            "timestamp": timestamp,
            "input_points": len(points),
            "static_points": len(final.static_points),
            "dynamic_points": len(final.dynamic_points),
            "unknown_points": len(final.unknown_points),
            "static_map_points": len(final.static_map),
            "dynamic_clusters": int(final.dynamic_cluster_labels.max()+1),
            "mean_dynamic_speed_mps": (
                float(final.scene_flow.speed_mps[final.scene_flow.labels == 1].mean())
                if np.any(final.scene_flow.labels == 1) else 0.0),
        })
    elapsed = time.perf_counter() - started
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "lesson": 56, "frames": frames, "elapsed_s": elapsed,
        "processing_rate_hz": frames/elapsed if elapsed > 0 else 0.0,
        "final_frame": history[-1],
        "frame_history": history,
        "outputs": {"report": str(out / "lesson56_dynamic_scene_report.json")},
        "limitations": [
            "Input clouds must be ego-motion compensated and expressed in a stable world frame.",
            "Nearest-neighbour flow is sensitive to occlusion, sampling changes, and pose drift.",
            "Unmatched points are labelled unknown, not automatically dynamic.",
            "Track boxes are axis-aligned approximations; object orientation is not applied.",
            "Static map is bounded voxel aggregation without ray-based free-space clearing.",
            "No learned scene flow, semantic segmentation, or dynamic SLAM optimization."
        ]
    }
    (out / "lesson56_dynamic_scene_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if visualize and final is not None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        if len(final.static_map):
            ax.scatter(final.static_map[:, 0], final.static_map[:, 1],
                       final.static_map[:, 2], s=1, label="Static world map")
        if len(final.dynamic_points):
            ax.scatter(final.dynamic_points[:, 0], final.dynamic_points[:, 1],
                       final.dynamic_points[:, 2], s=8, label="Dynamic observations")
        moving = np.flatnonzero(final.scene_flow.labels == 1)
        if len(moving):
            sample = moving[::max(1, len(moving)//80)]
            ax.quiver(final.dynamic_points[:0, 0], final.dynamic_points[:0, 1],
                      final.dynamic_points[:0, 2], [], [], [])
            # Draw only a small, readable subset of valid flow vectors.
            cur = make_sequence(frames)[-1][1]
            if len(cur) == len(final.scene_flow.flow):
                p = cur[sample]
                v = final.scene_flow.flow[sample]
                ax.quiver(p[:, 0], p[:, 1], p[:, 2], v[:, 0], v[:, 1],
                          v[:, 2], length=1.0, normalize=False)
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_zlabel("World Z (m)")
        ax.set_title("Lesson 56 — Static and Dynamic 4D Scene")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=45)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    report = run_demo(args.frames, args.visualize)
    print(json.dumps({
        "processing_rate_hz": report["processing_rate_hz"],
        "final_frame": report["final_frame"],
        "outputs": report["outputs"]
    }, indent=2))


if __name__ == "__main__":
    main()
