#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 46 — RGB-D VISUAL ODOMETRY + ICP SENSOR FUSION

PURPOSE
-------
Fuse image-feature motion (ORB + RGB-D PnP) with 3D point-cloud registration
(ICP). Reject inconsistent estimates, compose a world camera pose, and build
a bounded local 3D point map.

FRAME CONVENTION
----------------
Visual PnP returns T_current_previous.
ICP(current_cloud, previous_cloud) returns T_previous_current.
Both are converted to the same T_previous_current convention before fusion.

RUN
---
python lesson46_visual_icp_fusion.py --synthetic --visualize
python lesson46_visual_icp_fusion.py --realsense
python -m unittest -v tests.test_sensor_fusion

All runtime processing is local/offline. No pretrained weights or network APIs.
"""
import argparse
import numpy as np

from common.geometry import make_se3, transform_points
from common import DepthMap, CameraCalibration
from scene_reconstruction.geometry_pipeline import depth_to_point_cloud
from visual_odometry import ORBRGBDVisualOdometry, rectify_rgbd
from sensor_fusion import RGBDVisualICPFusion


def make_scene(seed=46, count=3500):
    rng = np.random.default_rng(seed)
    # Multiple non-coplanar surfaces provide geometric constraints for ICP.
    floor = np.column_stack([rng.uniform(-2, 2, count//3),
                             rng.uniform(1.2, 2.0, count//3),
                             rng.uniform(1.0, 5.0, count//3)])
    wall = np.column_stack([rng.uniform(-2, 2, count//3),
                            rng.uniform(-1.5, 1.5, count//3),
                            rng.uniform(4.0, 5.0, count//3)])
    side = np.column_stack([rng.uniform(-2, -1.7, count-count//3*2),
                            rng.uniform(-1.5, 2.0, count-count//3*2),
                            rng.uniform(1.0, 5.0, count-count//3*2)])
    return np.vstack([floor, wall, side])


def make_depth_cloud(K, width, height, depth_value=3.0, stride=3):
    depth = np.full((height, width), depth_value, dtype=np.float32)
    # A deterministic depth gradient avoids a single perfectly planar cloud.
    depth += (np.arange(height, dtype=np.float32)[:, None] / height) * 0.3
    dmap = DepthMap(depth, 0.0, "synthetic", unit="m")
    calib = CameraCalibration(K, width, height)
    return depth_to_point_cloud(dmap, calib, stride=stride,
                                min_depth_m=.2, max_depth_m=8.).points


def run_synthetic(args):
    import cv2
    width, height = args.width, args.height
    fx = fy = .9 * width
    K = np.array([[fx, 0., width/2], [0., fy, height/2], [0., 0., 1.]])
    image = np.random.default_rng(args.seed).integers(
        0, 256, (height, width), dtype=np.uint8)
    depth = np.full((height, width), 3.0, dtype=np.float32)
    # Create a reproducible sequence with small pixel shifts and small camera motion.
    vo = ORBRGBDVisualOdometry(K, min_depth_m=.2, max_depth_m=6.,
                               max_features=1800, min_matches=12,
                               min_pnp_inliers=8)
    fusion = RGBDVisualICPFusion(
        registration_voxel_m=.025, map_voxel_m=.035,
        max_correspondence_m=.22, min_correspondences=30)
    rng = np.random.default_rng(args.seed)
    points_world = make_scene(args.seed, 4500)
    true_pose_world_camera = np.eye(4)
    results = []
    for frame_index in range(args.frames):
        # Move the virtual camera through a static world. This ground-truth
        # trajectory is independent of the estimated/fused trajectory.
        if frame_index > 0:
            true_step = make_se3(np.eye(3), np.array([.018, 0.0, 0.0]))
            true_pose_world_camera = true_pose_world_camera @ true_step
        points_camera = transform_points(
            points_world, np.linalg.inv(true_pose_world_camera))
        if frame_index == 0:
            frame_image = image.copy()
        else:
            # Image translation is a deterministic visual-motion stimulus.
            dx = -2 * frame_index
            A = np.array([[1., 0., dx], [0., 1., 0.]], dtype=np.float32)
            frame_image = cv2.warpAffine(image, A, (width, height))
        vo_result = vo.process(frame_image, depth)
        result = fusion.integrate(points_camera, vo_result)
        results.append(result)
        translation_error = float(np.linalg.norm(
            result.T_world_current[:3, 3] - true_pose_world_camera[:3, 3]))
        print(f"frame={frame_index:03d} success={result.success} "
              f"visual={result.visual_used} icp={result.icp_used} "
              f"v_score={result.visual_score:.3f} i_score={result.icp_score:.3f} "
              f"ICP_RMSE={result.icp_rmse_m:.4f}m "
              f"pose_error={translation_error:.4f}m "
              f"map_points={len(fusion.map_points_world)} {result.message}")
    print("Final fused camera pose:")
    print(np.array2string(fusion.T_world_previous, precision=5, suppress_small=True))
    if args.visualize and len(fusion.map_points_world):
        import matplotlib.pyplot as plt
        cloud = fusion.map_points_world
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        sample = cloud[::max(1, len(cloud)//12000)]
        ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1)
        p = fusion.T_world_previous[:3, 3]
        ax.scatter([p[0]], [p[1]], [p[2]], s=70, marker="^")
        ax.set_box_aspect((1, 1, 1))
        ax.set_title("Lesson 46 — Fused local 3D map and camera pose")
        ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)"); ax.set_zlabel("World Z (m)")
        plt.tight_layout(); plt.show()


def run_realsense(args):
    from camera_adapters.realsense_d435 import RealSenseD435Adapter
    print("D435 RGB-D VO + ICP fusion. Ctrl+C to stop.")
    try:
        with RealSenseD435Adapter(width=args.width, height=args.height,
                                  fps=args.fps) as camera:
            sample = camera.read()
            if sample is None or sample.depth is None or sample.calibration is None:
                raise RuntimeError("D435 did not return an RGB-D sample")
            vo = ORBRGBDVisualOdometry(
                sample.calibration.K, min_depth_m=args.min_depth,
                max_depth_m=args.max_depth, max_features=args.features)
            fusion = RGBDVisualICPFusion(
                registration_voxel_m=args.registration_voxel,
                map_voxel_m=args.map_voxel,
                max_correspondence_m=args.max_correspondence,
                min_correspondences=args.min_correspondences)
            count = 0
            while sample is not None and count < args.max_frames:
                rectified_image, rectified_depth = rectify_rgbd(
                    sample.frame.image, sample.depth.in_metres(),
                    sample.calibration.K, sample.calibration.distortion)
                vo_result = vo.process(rectified_image, rectified_depth)
                rectified_map = DepthMap(
                    rectified_depth, sample.depth.timestamp,
                    sample.depth.camera_id, unit="m")
                cloud = depth_to_point_cloud(
                    rectified_map, sample.calibration, color_image=None,
                    stride=args.stride, min_depth_m=args.min_depth,
                    max_depth_m=args.max_depth)
                result = fusion.integrate(cloud.points, vo_result)
                count += 1
                print(f"frame={count:05d} ok={result.success} "
                      f"visual={result.visual_used} ICP={result.icp_used} "
                      f"V={result.visual_score:.3f} I={result.icp_score:.3f} "
                      f"ICP_RMSE={result.icp_rmse_m:.4f}m "
                      f"pose={result.T_world_current[:3,3].round(3)} "
                      f"{result.message}")
                sample = camera.read()
    except KeyboardInterrupt:
        print("Capture interrupted.")
    if 'fusion' in locals() and args.visualize and len(fusion.map_points_world):
        import matplotlib.pyplot as plt
        p = fusion.map_points_world
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(p[::max(1, len(p)//12000), 0],
                   p[::max(1, len(p)//12000), 1],
                   p[::max(1, len(p)//12000), 2], s=1)
        ax.set_title("D435 fused local point map")
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--features", type=int, default=2200)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--min-depth", type=float, default=.2)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--registration-voxel", type=float, default=.04)
    parser.add_argument("--map-voxel", type=float, default=.035)
    parser.add_argument("--max-correspondence", type=float, default=.2)
    parser.add_argument("--min-correspondences", type=int, default=30)
    parser.add_argument("--seed", type=int, default=46)
    args = parser.parse_args()
    if args.realsense:
        run_realsense(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
