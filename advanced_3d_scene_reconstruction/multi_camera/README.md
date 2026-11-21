# Multi-Camera Synchronization, Calibration, and Fusion

## 1. Timestamp synchronization

`FrameSynchronizer` pairs frames from multiple camera queues using timestamps.
All input timestamps must share a clock domain. USB host timestamps from
independent machines or RealSense sensor timestamps must not be mixed without
clock conversion. Software pairing does not replace hardware triggering.

## 2. Intrinsic calibration

Use a printed chessboard with a known square size in metres. Capture at least
5 images (20–30 varied views is a more practical target), spanning image
corners, distances, and board orientations. The helper returns the camera
matrix, distortion coefficients, global RMS, and per-view pixel RMSE.

## 3. Stereo extrinsic calibration

Provide synchronized left/right chessboard image pairs and separately estimated
intrinsics. The result convention is:

`X_right = R_right_left * X_left + T_right_left`

The result includes both transform directions, essential/fundamental matrices,
global calibration RMS, and per-view reprojection diagnostics. The stereo
calibration helper assumes equal image dimensions and a common board geometry.

## 4. Point-cloud fusion

`fuse_point_clouds` transforms each camera-local cloud using a supplied
`T_world_camera` and concatenates them. Optional voxel filtering averages
attributes within each occupied voxel. It does not estimate extrinsics or
solve registration; camera poses must come from calibration, a rig definition,
SLAM, or another validated estimator.

## 5. Run tests

From `advanced_3d_scene_reconstruction/`:

```bash
python -m unittest -v tests.test_multi_camera
```

The unit tests do not require OpenCV, cameras, or GPU. Chessboard calibration
functions require `opencv-python` and physical calibration images.
