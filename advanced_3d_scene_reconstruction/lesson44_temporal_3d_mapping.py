#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 44 — TEMPORAL 3D MAPPING WITH LOCAL ICP ODOMETRY

PURPOSE
-------
Register sequential point clouds, estimate relative camera motion, and build
a static 3D map in a fixed world frame. All computation is local and offline.

THEORY
------
ICP estimates T_target_source. Sequential poses compose as:
T_world_current = T_world_previous @ T_previous_current.
Frame-to-frame errors accumulate; this lesson has no loop closure or pose graph.

RUN
---
python lesson44_temporal_3d_mapping.py --visualize
python lesson44_temporal_3d_mapping.py --frames 8
"""
import argparse
import numpy as np
from common.geometry import make_se3, transform_points, invert_se3
from temporal_mapping import TemporalPointCloudMapper


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def make_static_scene(seed=13, count=1800):
    rng = np.random.default_rng(seed)
    n = count // 3
    floor = np.column_stack([rng.uniform(-3, 3, n),
                             rng.normal(0, .008, n), rng.uniform(2, 8, n)])
    wall = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-1.5, 1.5, n),
                            np.full(n, 8.) + rng.normal(0, .008, n)])
    blocks = []
    for center, scale in [([-1., .5, 4.], [.6, .8, .5]),
                          ([1.2, .4, 5.5], [.9, .7, .8]),
                          ([.2, 1.1, 6.5], [.5, .5, .5])]:
        blocks.append(rng.uniform(-1, 1, (count // 9, 3)) * scale + center)
    return np.vstack([floor, wall, *blocks])


def visualize_map(points):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib locally to visualize") from exc
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    p = points[::max(1, len(points)//15000)]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1, alpha=.5)
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_zlabel("World Z (m)")
    ax.set_title("Offline Temporal 3D Map")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    scene = make_static_scene(args.seed)
    mapper = TemporalPointCloudMapper(
        map_voxel_size_m=.045, registration_voxel_size_m=.035,
        max_correspondence_distance_m=.22, dynamic_residual_threshold_m=.12)
    print("Lesson 44: local-only temporal 3D mapping; no network access.")
    for i in range(args.frames):
        T_world_camera = make_se3(rotation_y(i*.008), np.array([i*.035, 0., 0.]))
        observed = transform_points(scene, invert_se3(T_world_camera))
        # A moving local cluster illustrates residual-based dynamic candidates.
        moving_world = np.random.default_rng(args.seed+i).normal(
            [.8-i*.06, .7, 3.5], .035, (90, 3))
        current = np.vstack([observed, transform_points(
            moving_world, invert_se3(T_world_camera))])
        result = mapper.integrate(current)
        print(f"frame={i:02d} map={len(result.map_points_world):5d} "
              f"icp_rmse={result.registration.rmse_m:.5f}m "
              f"dynamic_candidates={len(result.dynamic_points_current):4d} "
              f"world_x={result.T_world_current[0,3]:+.3f}m")
    if args.visualize:
        visualize_map(mapper.map_points_world)


if __name__ == "__main__":
    main()
