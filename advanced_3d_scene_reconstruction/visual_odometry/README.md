# Visual Odometry — Lesson 45

Local ORB feature matching plus metric RGB-D PnP.

## Algorithm

1. Detect ORB keypoints and binary descriptors in each image.
2. Match descriptors with Hamming distance and Lowe ratio filtering.
3. Estimate an Essential Matrix with RANSAC as a two-view geometric diagnostic.
4. Sample metric depth around previous-frame keypoints.
5. Back-project those keypoints into 3D in the previous camera frame.
6. Use current-frame matched pixels with solvePnPRansac to estimate
   `T_current_previous` in metric units.
7. Invert that transform and compose the camera world pose.

The Essential Matrix alone cannot recover metric translation scale. The PnP
stage uses the previous RGB-D frame's depth to estimate translation in metres.

## Offline runtime

No remote inference, web API, telemetry, or model downloads. OpenCV and NumPy
execute locally. The optional RealSense mode reads the connected D435 directly.

Prepare dependencies before disconnecting:

    python -m pip install numpy opencv-python matplotlib pyrealsense2

Run:

    python lesson45_rgbd_visual_odometry.py --synthetic --visualize
    python lesson45_rgbd_visual_odometry.py --realsense
    python -m unittest -v tests.test_visual_odometry

## Input requirements

- Color and depth must be spatially aligned and have identical dimensions.
- Images must be rectified pinhole images; lens distortion must be removed.
- Depth must be optical-axis Z in metres, not millimetres or radial range.
- The previous-frame depth is used for 3D feature lifting.
- Enough static textured points and camera overlap are required.

## Limitations

This is a front-end odometry baseline, not complete SLAM. It has no loop
closure, global pose graph, bundle adjustment, relocalization, learned feature
matching, or robust long-term map maintenance. Repeated texture, motion blur,
low texture, dynamic objects, and weak depth coverage can cause failure.
The synthetic warp is a controlled educational fixture, not a physically
complete renderer with visibility-aware z-buffering.
