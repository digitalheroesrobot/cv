#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 49 — REAL-TIME RGB-D SLAM INTEGRATION (INTEL REALSENSE D435)

PURPOSE
-------
1. Capture aligned RGB-D frames from a locally connected D435.
2. Rectify images, estimate ORB RGB-D visual odometry, and register depth clouds.
3. Fuse visual and ICP motion estimates.
4. Select keyframes and construct a sparse pose graph.
5. Search old keyframes and verify loop candidates with RGB-D PnP + ICP.
6. Optimize the graph and rebuild the global point map after accepted loops.
7. Save a local PLY point cloud and JSON trajectory report.

RUN (LOCAL / OFFLINE)
---------------------
python lesson49_realtime_rgbd_slam.py --realsense --visualize
python lesson49_realtime_rgbd_slam.py --synthetic --visualize
python -m unittest -v tests.test_realtime_slam

The synthetic mode tests the mapping/graph backend with deterministic poses.
It does not emulate real image feature tracking or certify camera performance.
No internet requests, cloud inference, or automatic model downloads are used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common.geometry import invert_se3, transform_points
from keyframe_slam import PoseGraph, PoseGraphEdge, Keyframe, rebuild_global_map


def pose(x, y, yaw=0.):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, 0.]
    return T


def save_ply(path, points):
    points = np.asarray(points, dtype=np.float64)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("end_header\n")
        np.savetxt(stream, points, fmt="%.6f")


def run_synthetic(args):
    rng = np.random.default_rng(args.seed)
    world = rng.uniform([-2., -2., 1.0], [2., 2., 4.5], (args.points, 3))
    truth = [pose(0, 0), pose(.7, 0), pose(1.0, .6, .5),
             pose(.7, 1.2, 1.5), pose(0, 1.0, 2.4), pose(0, 0, 3.1)]
    noisy = [truth[0], pose(.76, .03), pose(1.10, .68, .56),
             pose(.78, 1.30, 1.58), pose(.08, 1.08, 2.47),
             pose(.13, .12, 3.15)]
    graph = PoseGraph()
    keyframes = []
    for i, (T_true, T_noisy) in enumerate(zip(truth, noisy)):
        graph.add_node(T_noisy)
        local = transform_points(world, invert_se3(T_true))
        keyframes.append(Keyframe(i, float(i), T_noisy, point_cloud=local))
        if i > 0:
            measured = invert_se3(truth[i-1]) @ truth[i]
            graph.add_edge(PoseGraphEdge(i-1, i, measured,
                                         kind="keyframe_odometry"))
    # Synthetic loop edge is treated as externally geometrically verified.
    graph.add_edge(PoseGraphEdge(0, len(truth)-1,
                                 invert_se3(truth[0]) @ truth[-1],
                                 translation_information=18.,
                                 rotation_information=18.,
                                 kind="synthetic_verified_loop"))
    result = graph.optimize(max_nfev=300, loss="huber", f_scale=.05)
    global_map = rebuild_global_map(keyframes, result.poses,
                                    voxel_size_m=args.map_voxel,
                                    max_points=args.max_map_points)
    save_ply(args.output_ply, global_map)
    report = {
        "lesson": 49, "mode": "synthetic_backend", "offline": True,
        "optimizer_success": result.success,
        "initial_cost": result.initial_cost, "final_cost": result.final_cost,
        "keyframes": len(keyframes), "graph_edges": len(graph.edges),
        "global_map_points": len(global_map),
        "optimized_poses": [T.tolist() for T in result.poses],
        "note": "Synthetic known loop constraint; not a real camera or visual loop-closure benchmark."
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Pose graph: {result.initial_cost:.6f} -> {result.final_cost:.6f}")
    print(f"Keyframes: {len(keyframes)} | map points: {len(global_map)}")
    print(f"Saved: {args.output_ply} and {args.output_json}")
    if args.visualize:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        sample = global_map[::max(1, len(global_map)//15000)]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
        for label, poses in [("Initial", noisy), ("Optimized", result.poses)]:
            xyz = np.asarray([T[:3, 3] for T in poses])
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker="o", label=label)
        ax.set_title("Lesson 49 — Integrated RGB-D SLAM Backend (Synthetic)")
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)"); ax.set_zlabel("World Z (m)")
        ax.legend(); plt.tight_layout(); plt.show()


def run_realsense(args):
    import cv2
    from camera_adapters.realsense_d435 import RealSenseD435Adapter
    from keyframe_slam.realtime import RealTimeRGBDSLAM

    print("Starting local Intel RealSense D435 RGB-D SLAM. Press q or Ctrl+C to stop.")
    slam = None
    trajectory = []
    try:
        with RealSenseD435Adapter(width=args.width, height=args.height,
                                  fps=args.fps, serial=args.serial) as camera:
            if camera.calibration is None:
                raise RuntimeError("D435 calibration is unavailable")
            slam = RealTimeRGBDSLAM(
                camera.calibration.K, min_depth_m=args.min_depth,
                max_depth_m=args.max_depth, cloud_stride=args.stride,
                keyframe_translation_m=args.keyframe_translation,
                keyframe_rotation_rad=np.deg2rad(args.keyframe_rotation_deg),
                keyframe_frame_gap=args.keyframe_gap,
                loop_min_id_gap=args.loop_min_gap,
                loop_min_matches=args.loop_min_matches,
                loop_min_inliers=args.loop_min_inliers,
                loop_max_rmse_m=args.loop_max_rmse,
                loop_max_translation_gap_m=args.loop_max_translation_gap,
                loop_max_rotation_gap_deg=args.loop_max_rotation_gap_deg,
                registration_voxel_m=args.registration_voxel,
                map_voxel_m=args.map_voxel,
                max_map_points=args.max_map_points)
            while True:
                sample = camera.read()
                if sample is None or sample.depth is None:
                    continue
                slam.set_image_shape(sample.frame.image.shape)
                depth_m = sample.depth.in_metres()
                result = slam.process(sample.frame.image, depth_m,
                                       sample.frame.timestamp)
                p = result.T_world_camera[:3, 3]
                trajectory.append(p.copy())
                view = sample.frame.image.copy()
                status = (f"frame {result.frame_index} KF={len(slam.keyframes.keyframes)} "
                          f"loops={len(slam._added_loop_pairs)} map={result.map_points}")
                cv2.putText(view, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            .58, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(view, f"pose[m] {p.round(2)} | {result.message[:90]}",
                            (12, 55), cv2.FONT_HERSHEY_SIMPLEX,
                            .45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Lesson 49 - Real-Time RGB-D SLAM", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if args.max_frames > 0 and result.frame_index + 1 >= args.max_frames:
                    break
    except KeyboardInterrupt:
        print("Capture interrupted by user.")
    finally:
        cv2.destroyAllWindows()

    if slam is not None:
        points = slam.global_map if len(slam.global_map) else slam.fusion.map_points_world
        save_ply(args.output_ply, points)
        report = {
            "lesson": 49, "mode": "realsense_d435", "offline": True,
            "frames_processed": slam.frame_index,
            "keyframes": len(slam.keyframes.keyframes),
            "pose_graph_edges": len(slam.graph.edges),
            "verified_loop_closures": len(slam._added_loop_pairs),
            "global_map_points": len(points),
            "trajectory_xyz_m": [p.tolist() for p in trajectory],
            "keyframe_poses": [T.tolist() for T in slam.graph.poses],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved global map: {args.output_ply}")
        print(f"Saved trajectory report: {args.output_json}")
        if args.visualize and len(points):
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection="3d")
            cloud = points[::max(1, len(points)//20000)]
            ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], s=1)
            if trajectory:
                path = np.asarray(trajectory)
                ax.plot(path[:, 0], path[:, 1], path[:, 2], linewidth=1.5)
            ax.set_title("Lesson 49 — D435 Global Point Map and Trajectory")
            ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)"); ax.set_zlabel("World Z (m)")
            plt.tight_layout(); plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--serial", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0,
                        help="0 means run until q/Ctrl+C")
    parser.add_argument("--min-depth", type=float, default=.20)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--keyframe-translation", type=float, default=.20)
    parser.add_argument("--keyframe-rotation-deg", type=float, default=8.0)
    parser.add_argument("--keyframe-gap", type=int, default=15)
    parser.add_argument("--loop-min-gap", type=int, default=10)
    parser.add_argument("--loop-min-matches", type=int, default=25)
    parser.add_argument("--loop-min-inliers", type=int, default=15)
    parser.add_argument("--loop-max-rmse", type=float, default=.035)
    parser.add_argument("--loop-max-translation-gap", type=float, default=.12)
    parser.add_argument("--loop-max-rotation-gap-deg", type=float, default=7.0)
    parser.add_argument("--registration-voxel", type=float, default=.04)
    parser.add_argument("--map-voxel", type=float, default=.035)
    parser.add_argument("--max-map-points", type=int, default=250000)
    parser.add_argument("--points", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=49)
    parser.add_argument("--output-ply", default="outputs/lesson49_global_map.ply")
    parser.add_argument("--output-json", default="outputs/lesson49_slam_report.json")
    args = parser.parse_args()
    if args.realsense:
        run_realsense(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
