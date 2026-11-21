#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 47 — KEYFRAME RGB-D SLAM AND POSE GRAPH OPTIMIZATION

PURPOSE
-------
1. Explain why frame-to-frame odometry accumulates drift.
2. Select keyframes using translation, rotation, frame gap, and tracking quality.
3. Represent camera poses as graph nodes and relative motion as graph edges.
4. Add a geometrically verified loop-closure constraint.
5. Optimize the complete trajectory while fixing the first pose (gauge anchor).

IMPORTANT
---------
The synthetic experiment uses known relative-motion constraints with deliberately
drifted initial poses. It demonstrates pose-graph mathematics, not image-based
loop-closure performance. The ORB module only proposes visual candidates; a
real system must geometrically verify each candidate with RGB-D PnP/ICP before
creating a loop edge.

RUN (offline)
-------------
python lesson47_keyframe_rgbd_slam.py --synthetic --visualize
python -m unittest -v tests.test_keyframe_slam

Dependencies: numpy, scipy; matplotlib only for visualization; OpenCV only for
the optional ORB candidate detector. No network access or model downloads.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from keyframe_slam import PoseGraph, PoseGraphEdge, KeyframeManager


def make_pose(x, y, yaw=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, 0.]
    return T


def relative(T_world_i, T_world_j):
    return np.linalg.inv(T_world_i) @ T_world_j


def make_square_trajectory():
    return [
        make_pose(0, 0, 0),
        make_pose(1, 0, 0),
        make_pose(1, 1, np.pi / 2),
        make_pose(0, 1, np.pi),
        make_pose(0, 0, -np.pi / 2),
    ]


def run_synthetic(args):
    truth = make_square_trajectory()
    # Simulate odometry drift in the initial estimates. The first pose is fixed.
    initial = [
        make_pose(0, 0),
        make_pose(1.08, .02, .025),
        make_pose(1.13, 1.10, 1.60),
        make_pose(.04, 1.18, 3.18),
        make_pose(.18, .16, -1.53),
    ]
    manager = KeyframeManager(translation_threshold_m=.35,
                              rotation_threshold_rad=.25, max_frame_gap=10)
    graph = PoseGraph()
    for index, T in enumerate(initial):
        if manager.should_insert(index * 10, T, tracking_quality=1.0):
            kf = manager.add(index * 10, index / 10., T)
            graph.add_node(kf.T_world_camera)

    # Consecutive keyframe odometry constraints (simulated measurements).
    for i in range(len(truth) - 1):
        graph.add_edge(PoseGraphEdge(
            i, i + 1, relative(truth[i], truth[i + 1]),
            translation_information=1.0, rotation_information=2.0,
            kind="odometry"))

    # This loop edge represents a candidate that has already passed geometric
    # verification. Appearance similarity alone is not enough to add this edge.
    graph.add_edge(PoseGraphEdge(
        0, len(truth) - 1, np.eye(4),
        translation_information=args.loop_weight,
        rotation_information=args.loop_weight,
        kind="verified_loop"))

    before = graph.residual_vector()
    print("Initial trajectory:")
    for i, T in enumerate(graph.poses):
        print(f"  keyframe={i} position={T[:3, 3].round(3)}")
    result = graph.optimize(max_nfev=args.max_nfev, loss="huber",
                            f_scale=args.robust_scale)
    print("\nPose graph optimization:")
    print(f"  success={result.success} evaluations={result.evaluations}")
    print(f"  initial_cost={result.initial_cost:.8f}")
    print(f"  final_cost={result.final_cost:.8f}")
    print(f"  optimizer={result.message}")
    for i, T in enumerate(result.poses):
        error = np.linalg.norm(T[:3, 3] - truth[i][:3, 3])
        print(f"  keyframe={i} optimized={T[:3, 3].round(3)} "
              f"ground_truth={truth[i][:3, 3].round(3)} "
              f"translation_error={error:.4f}m")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "lesson": 47,
        "mode": "synthetic_pose_graph",
        "offline": True,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "success": result.success,
        "poses_world_camera": [T.tolist() for T in result.poses],
        "edge_types": [edge.kind for edge in graph.edges],
        "limitations": [
            "Synthetic relative constraints; no real visual loop closure is evaluated.",
            "ORB matching only proposes candidates and is not geometric verification.",
            "No IMU, bundle adjustment, marginalization, or dense global map correction.",
        ],
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report: {output}")

    if args.visualize:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        for poses, label, marker in [
            (initial, "Drifted initial", "x"),
            (truth, "Ground truth", "o"),
            (result.poses, "Optimized", "^"),
        ]:
            xy = np.array([T[:2, 3] for T in poses])
            ax.plot(xy[:, 0], xy[:, 1], marker=marker, label=label)
            for i, p in enumerate(xy):
                ax.annotate(str(i), p)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_title("Lesson 47 — Keyframe Pose Graph and Loop Closure")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Run deterministic offline pose-graph experiment")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--loop-weight", type=float, default=20.0)
    parser.add_argument("--robust-scale", type=float, default=.05)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--output", default="outputs/lesson47_pose_graph.json")
    args = parser.parse_args()
    if args.loop_weight <= 0 or args.robust_scale <= 0 or args.max_nfev < 1:
        parser.error("loop weight, robust scale, and max evaluations must be positive")
    run_synthetic(args)


if __name__ == "__main__":
    main()
