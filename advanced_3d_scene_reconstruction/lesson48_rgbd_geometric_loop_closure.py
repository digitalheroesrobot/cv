#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 48 — RGB-D GEOMETRIC LOOP CLOSURE AND GLOBAL MAP CORRECTION

PURPOSE
-------
1. Generate appearance-based loop candidates (Lesson 47 ORB module).
2. Estimate a metric candidate-to-current transform from RGB-D PnP.
3. Refine it with point-to-point ICP.
4. Reject loop closures when PnP and ICP disagree or geometry is weak.
5. Add only verified constraints to the pose graph.
6. Rebuild the global point map using optimized keyframe poses.

THIS LESSON
-----------
The executable demo is fully synthetic and tests the geometric verification
stage with a known metric PnP initialization. The reusable verifier accepts
that PnP result; real-image ORB matching and PnP are connected through the
existing visual_odometry.estimate_rgbd_pnp function. Appearance matching alone
must never create a loop edge.

RUN (offline)
-------------
python lesson48_rgbd_geometric_loop_closure.py --synthetic --visualize
python -m unittest -v tests.test_geometric_loop_closure

Dependencies: numpy, scipy; matplotlib only for visualization. OpenCV is
required for the real-image ORB/PnP front-end. No network calls or downloads.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common.geometry import invert_se3, transform_points
from keyframe_slam import (
    PoseGraph, PoseGraphEdge, Keyframe, RGBDLoopClosureVerifier,
    rebuild_global_map,
)


def make_pose(x=0., y=0., yaw=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, 0.]
    return T


def run_synthetic(args):
    rng = np.random.default_rng(args.seed)
    candidate_cloud = rng.uniform(
        [-1., -.7, 1.5], [1., .8, 4.], (args.points, 3))
    T_candidate_current = make_pose(.12, -.04, .025)
    current_cloud = transform_points(
        candidate_cloud, invert_se3(T_candidate_current))
    # PnP gives T_current_candidate; the verifier converts it to the graph
    # edge convention T_candidate_current.
    T_current_candidate_pnp = invert_se3(T_candidate_current)
    verifier = RGBDLoopClosureVerifier(
        voxel_size_m=args.voxel,
        max_correspondence_m=args.max_correspondence,
        min_correspondences=args.min_correspondences,
        max_rmse_m=args.max_rmse,
        max_translation_gap_m=args.max_translation_gap,
        max_rotation_gap_rad=np.deg2rad(args.max_rotation_gap_deg))
    result = verifier.verify_cloud_pair(
        candidate_cloud, current_cloud, T_current_candidate_pnp)
    print("Geometric loop closure verification:")
    print(f"  accepted={result.accepted} reason={result.reason}")
    print(f"  ICP converged={result.icp_converged} "
          f"correspondences={result.icp_correspondences}")
    print(f"  ICP RMSE={result.icp_rmse_m:.6f}m")
    print(f"  PnP/ICP translation disagreement={result.translation_disagreement_m:.6f}m")
    print(f"  PnP/ICP rotation disagreement={np.rad2deg(result.rotation_disagreement_rad):.5f}deg")

    # Demonstrate the required gate: an accepted closure becomes an edge;
    # a failed verification must not modify the graph.
    T0, T1_initial = make_pose(), make_pose(.28, .09, .05)
    graph = PoseGraph()
    graph.add_node(T0)
    graph.add_node(T1_initial)
    graph.add_edge(PoseGraphEdge(0, 1, make_pose(.12, -.04, .025),
                                 kind="odometry"))
    if result.accepted:
        graph.add_edge(PoseGraphEdge(
            0, 1, result.T_candidate_current,
            translation_information=args.loop_information,
            rotation_information=args.loop_information,
            kind="verified_loop"))
    before = graph.residual_vector()
    optimized = graph.optimize(max_nfev=200, loss="huber", f_scale=.05)
    print(f"  graph edges={len(graph.edges)} initial_cost={optimized.initial_cost:.7f} "
          f"final_cost={optimized.final_cost:.7f}")

    # Each keyframe stores its local cloud. Rebuild map from optimized poses.
    keyframes = [
        Keyframe(0, 0., T0, point_cloud=candidate_cloud),
        Keyframe(1, 1., T1_initial, point_cloud=current_cloud),
    ]
    global_map = rebuild_global_map(
        keyframes, optimized.poses, voxel_size_m=args.map_voxel)
    print(f"  corrected global map points={len(global_map)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "lesson": 48, "offline": True, "mode": "synthetic_geometry",
        "loop_accepted": result.accepted, "loop_reason": result.reason,
        "icp_rmse_m": result.icp_rmse_m,
        "icp_correspondences": result.icp_correspondences,
        "translation_disagreement_m": result.translation_disagreement_m,
        "rotation_disagreement_rad": result.rotation_disagreement_rad,
        "pose_graph_edge_count": len(graph.edges),
        "pose_graph_initial_cost": optimized.initial_cost,
        "pose_graph_final_cost": optimized.final_cost,
        "global_map_point_count": len(global_map),
        "limitations": [
            "Synthetic PnP initialization; no real image feature matching is evaluated.",
            "Point-to-point ICP assumes sufficient overlap and a reasonable initial pose.",
            "Map rebuild is voxel point fusion, not TSDF, mesh, or dynamic-object fusion.",
        ],
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved report: {out}")

    if args.visualize:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        if len(global_map):
            sample = global_map[::max(1, len(global_map)//12000)]
            ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
        for i, T in enumerate(optimized.poses):
            p = T[:3, 3]
            ax.scatter([p[0]], [p[1]], [p[2]], s=70)
            ax.text(p[0], p[1], p[2], str(i))
        ax.set_title("Lesson 48 — Verified Loop Closure and Corrected Map")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_zlabel("World Z (m)")
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--points", type=int, default=3000)
    parser.add_argument("--voxel", type=float, default=.012)
    parser.add_argument("--max-correspondence", type=float, default=.20)
    parser.add_argument("--min-correspondences", type=int, default=80)
    parser.add_argument("--max-rmse", type=float, default=.02)
    parser.add_argument("--max-translation-gap", type=float, default=.04)
    parser.add_argument("--max-rotation-gap-deg", type=float, default=2.5)
    parser.add_argument("--loop-information", type=float, default=20.)
    parser.add_argument("--map-voxel", type=float, default=.025)
    parser.add_argument("--output", default="outputs/lesson48_loop_closure.json")
    args = parser.parse_args()
    if (args.points < 100 or args.voxel <= 0 or args.max_correspondence <= 0 or
            args.min_correspondences < 3 or args.max_rmse <= 0 or
            args.max_translation_gap <= 0 or args.max_rotation_gap_deg <= 0 or
            args.loop_information <= 0 or args.map_voxel <= 0):
        parser.error("All geometric thresholds must be positive; points >= 100")
    run_synthetic(args)


if __name__ == "__main__":
    main()
