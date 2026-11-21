#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 45 — ORB VISUAL ODOMETRY WITH METRIC RGB-D PnP

PURPOSE
-------
Estimate camera motion from consecutive local RGB-D frames using ORB feature
matches, Essential Matrix geometry, and robust 3D-to-2D PnP.

THEORY
------
PnP maps 3D points from the previous camera frame into 2D pixels in the current
frame. OpenCV returns T_current_previous. To update the camera world pose:
T_world_current = T_world_previous @ inverse(T_current_previous).

The Essential Matrix estimates translation direction only (monocular scale
ambiguity). Metric depth is used by PnP to recover translation in metres.

RUN
---
python lesson45_rgbd_visual_odometry.py --synthetic --visualize
python lesson45_rgbd_visual_odometry.py --realsense --visualize

All processing is local. No model weights, network access, or cloud services.
"""
import argparse
import numpy as np
from visual_odometry import ORBRGBDVisualOdometry


def make_textured_scene(width, height, seed=5):
    """Create deterministic high-frequency texture and a metric depth scene."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, (height, width), dtype=np.uint8)
    # Add stable corners and shapes to provide repeatable ORB features.
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install opencv-python locally") from exc
    for y in range(35, height-20, 55):
        for x in range(35, width-20, 55):
            cv2.rectangle(image, (x, y), (x+18, y+18),
                          int(rng.integers(0, 255)), 2)
            cv2.circle(image, (x+9, y+9), 5, int(rng.integers(0, 255)), -1)
    depth = np.full((height, width), 2.8, dtype=np.float32)
    # Give the scene non-planar depth variation to constrain PnP.
    for y in range(height):
        depth[y] += 0.25 * np.sin(y / 37.0)
    return image, depth


def warp_pair(image, depth, K, T_current_previous):
    """Render an approximate second view using a local depth-based inverse warp.

    This renderer is a controlled teaching fixture. Real camera mode uses the
    actual sensor frames; it is not intended to model occlusion perfectly.
    """
    import cv2
    h, w = depth.shape
    yy, xx = np.mgrid[0:h, 0:w]
    z = depth
    X = (xx - K[0, 2]) * z / K[0, 0]
    Y = (yy - K[1, 2]) * z / K[1, 1]
    points = np.stack([X, Y, z], axis=-1).reshape(-1, 3)
    R = T_current_previous[:3, :3]
    t = T_current_previous[:3, 3]
    current = points @ R.T + t
    uvw = current @ K.T
    u = (uvw[:, 0] / uvw[:, 2]).reshape(h, w)
    v = (uvw[:, 1] / uvw[:, 2]).reshape(h, w)
    map_x, map_y = u.astype(np.float32), v.astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    # Approximate z-buffer by remapping depth; enough for controlled demo.
    warped_depth = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT)
    warped_depth[warped_depth <= 0] = np.nan
    return warped, warped_depth


def run_synthetic(args):
    width, height = args.width, args.height
    fx = fy = 0.9 * width
    K = np.array([[fx, 0., width/2], [0., fy, height/2], [0., 0., 1.]])
    image, depth = make_textured_scene(width, height, args.seed)
    vo = ORBRGBDVisualOdometry(K, min_depth_m=.2, max_depth_m=6.,
                               min_matches=12, min_pnp_inliers=8)
    first = vo.process(image, depth)
    # Small known camera motion; translation is in metres.
    angle = np.deg2rad(0.5)
    R = np.array([[np.cos(angle), 0., np.sin(angle)],
                  [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
    from common.geometry import make_se3
    T_current_previous = make_se3(R, np.array([-.035, 0., .005]))
    second_image, second_depth = warp_pair(image, depth, K, T_current_previous)
    second = vo.process(second_image, second_depth)
    print("Lesson 45 — synthetic local RGB-D visual odometry")
    print(f"first frame: {first.message}")
    print(f"second frame: success={second.success} matches={second.feature_matches} "
          f"essential_inliers={second.essential_inliers} "
          f"pnp_inliers={second.pnp_inliers} reprojection_RMSE={second.reprojection_rmse_px:.3f}px")
    print("Estimated camera world pose:")
    print(np.array2string(second.T_world_current, precision=5, suppress_small=True))
    if args.visualize:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image, cmap="gray"); axes[0].set_title("Previous frame")
        axes[1].imshow(second_image, cmap="gray"); axes[1].set_title("Current frame")
        for ax in axes: ax.axis("off")
        plt.tight_layout(); plt.show()


def run_realsense(args):
    """Run ORB RGB-D VO from a local D435 stream."""
    from camera_adapters.realsense_d435 import RealSenseD435Adapter
    print("D435 local RGB-D visual odometry. Ctrl+C to stop.")
    try:
        with RealSenseD435Adapter(width=args.width, height=args.height,
                                  fps=args.fps) as camera:
            sample0 = camera.read()
            if sample0 is None or sample0.depth is None or sample0.calibration is None:
                raise RuntimeError("No aligned RGB-D sample received from D435")
            vo = ORBRGBDVisualOdometry(
                sample0.calibration.K, min_depth_m=args.min_depth,
                max_depth_m=args.max_depth, max_features=args.features)
            sample = sample0
            count = 0
            while count < args.max_frames:
                if sample is not None and sample.depth is not None:
                    result = vo.process(sample.frame.image, sample.depth.in_metres())
                    count += 1
                    print(f"frame={count:05d} ok={result.success} "
                          f"matches={result.feature_matches:4d} "
                          f"PnP_inliers={result.pnp_inliers:4d} "
                          f"reproj={result.reprojection_rmse_px:.2f}px "
                          f"world_t={result.T_world_current[:3,3].round(3)} "
                          f"{result.message}")
                sample = camera.read()
    except KeyboardInterrupt:
        print("Capture interrupted.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--realsense", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--features", type=int, default=2500)
    parser.add_argument("--min-depth", type=float, default=.2)
    parser.add_argument("--max-depth", type=float, default=6.)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()
    if args.realsense:
        run_realsense(args)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
