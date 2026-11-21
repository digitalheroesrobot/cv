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
from visual_odometry import ORBRGBDVisualOdometry, rectify_rgbd


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
    """Create a small translated teaching pair using a local affine warp.

    The fixture approximates lateral motion at mean scene depth. It is not a
    physically complete renderer and does not model occlusion or parallax.
    """
    import cv2
    mean_depth = float(np.nanmedian(depth))
    # For small lateral motion, delta_u ~= fx * t_x / Z.
    dx = float(K[0, 0] * T_current_previous[0, 3] / mean_depth)
    dy = float(K[1, 1] * T_current_previous[1, 3] / mean_depth)
    A = np.array([[1., 0., dx], [0., 1., dy]], dtype=np.float32)
    h, w = depth.shape
    warped = cv2.warpAffine(image, A, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
    warped_depth = cv2.warpAffine(depth, A, (w, h), flags=cv2.INTER_NEAREST,
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
                    rectified_image, rectified_depth = rectify_rgbd(
                        sample.frame.image, sample.depth.in_metres(),
                        sample.calibration.K, sample.calibration.distortion)
                    result = vo.process(rectified_image, rectified_depth)
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
