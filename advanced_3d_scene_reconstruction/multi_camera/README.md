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


## 6. Lesson 51 synchronized RGB-D world reconstruction

`reconstruction.py` provides `MultiCameraReconstructor`. It accepts one
`CameraSample` per calibrated camera, rejects timestamp skew beyond a
configured tolerance, rectifies registered color/depth, creates metric
camera-local clouds, applies supplied `T_world_camera` transforms, and fuses
the result in a common world frame.

Run the offline demonstration from the course root:

```bash
python lesson51_multi_camera_reconstruction.py --visualize
python -m unittest -v tests.test_multi_camera_reconstruction
```

Important assumptions:

- All camera timestamps must share a clock domain or have a validated clock
  conversion. Independent camera hardware clocks are not automatically
  synchronized by this software.
- Extrinsics are supplied, not estimated by the fusion engine.
- Depth must be registered to the color image.
- The current rectification helper supports pinhole distortion; fisheye needs
  a dedicated calibration/rectification implementation.
- Voxel fusion removes/reduces duplicate samples but is not a TSDF, mesh, or
  radiometric color-blending reconstruction.


## 7. Lesson 52 — camera rig extrinsic optimization

`rig_calibration.py` builds a connected graph from pairwise camera
measurements. Each edge stores `T_camera_j_camera_i`, while each optimized
node stores `T_rig_camera`. The optimizer fixes one camera as the rig origin,
weights rotation and translation residuals using supplied standard deviations,
and applies robust nonlinear least squares.

For a physical pair (camera_i, camera_j):

1. Calibrate each camera intrinsic matrix and lens distortion.
2. Capture at least five synchronized views of the same rigid chessboard.
3. Call `calibrate_stereo_from_chessboards(...)`.
4. Convert the result with `stereo_result_to_rig_edge(camera_i, camera_j, result)`.
5. Build a connected graph and call `RigPoseGraph(camera_ids, edges).optimize()`.
6. Check per-edge residuals and validate using held-out target observations.

Run the offline numerical lab:

```bash
python lesson52_camera_rig_calibration.py --visualize
python -m unittest -v tests.test_rig_calibration
```

The approximate covariance is derived from the local Jacobian information
matrix. It is conditional on the chosen residual parameterization, uncertainty
scales, and robust-loss behavior. Poor target coverage, inaccurate square
dimensions, unsynchronized captures, or weak graph connectivity can produce
misleadingly confident results.


## 8. Lesson 53 — clock-domain calibration and frame alignment

`time_sync.py` provides:

- An affine device-to-master clock model, including offset and drift (ppm).
- Robust least-squares fitting from paired synchronization events with MAD
  outlier rejection.
- A multi-camera frame aligner that converts timestamps before matching.
- Explicit counters for duplicate IDs, out-of-order frames, queue overflow,
  stale frames, and emitted synchronized sets.
- Camera-pose interpolation using linear translation and SO(3) SLERP.

Run the deterministic offline lab:

```bash
python lesson53_multicamera_time_synchronization.py --visualize
python -m unittest -v tests.test_time_sync
```

The clock fit requires pairs that represent the same physical synchronization
event, such as a shared hardware trigger or a precisely timestamped pulse.
Do not fit clock offset from unrelated host frame-arrival times: USB,
scheduler, buffering, and exposure latency contaminate those observations.
A fitted software clock model corrects timestamp coordinates; it does not
synchronize sensor exposure. For hardware synchronization, use supported
trigger wiring and validate exposure timing independently.

RealSense frames now retain the SDK timestamp-domain label and sensor frame
number in metadata. Devices with different timestamp domains require a
calibrated mapping into one master clock before frame pairing. The synthetic
lab does not claim that independently connected D435 devices are hardware
synchronized.


## 9. Lesson 54 — threaded real-time RGB-D capture

`capture.py` owns camera adapter lifecycle, launches one reader thread per
camera, converts frame and depth timestamps into a configured master clock,
maintains bounded per-camera queues, and emits complete RGB-D bundles whose
maximum timestamp skew is within a configured threshold. It records captured
frames, stale drops, queue overflow, out-of-order timestamps, reader errors,
and emitted/rejected bundles.

Run the local synthetic pipeline:

```bash
python lesson54_realtime_multicamera_world.py --cameras 3 --frames 30 --visualize
python -m unittest -v tests.test_multicamera_capture
```

Run connected D435 devices with measured calibration:

```bash
python lesson54_realtime_multicamera_world.py --realsense \
  --serials SERIAL_A,SERIAL_B \
  --poses-json configs/camera_rig.json \
  --clock-json configs/camera_clocks.json
```

The rig configuration must contain `T_world_camera` for each `camN`. Clock
configuration is optional only when timestamp domains are already known to be
compatible. Software matching cannot synchronize exposure. The latest fused
cloud is exported as a world-frame PLY snapshot; temporal object tracking and
persistent map integration are intentionally outside this lesson.


## 10. Lesson 55 — Multi-camera 3D detection fusion and tracking

`object_tracking.py` introduces:

- `Detection3D`: one camera's metric world-frame detection.
- `MultiCameraDetectionFusion`: cross-camera duplicate fusion, never merging
  two detections from the same camera.
- `ConstantVelocityKalman`: 6D position/velocity state with a 3D measurement.
- `MultiCamera3DTracker`: gated Hungarian assignment, class/size/optional
  embedding checks, track confirmation, occlusion tolerance, and expiration.

All detections must already use the same world coordinate frame and metres.
The fusion stage uses geometric proximity and optional supplied embeddings;
it does not estimate camera poses or run a neural detector.

Run the offline demo and tests:

```bash
python lesson55_multicamera_3d_tracking.py --frames 100 --visualize
python -m unittest -v tests.test_object_tracking
```

This is a tracking backend lab, not a complete multi-camera MOT system. A
production deployment still needs a validated detector, calibrated 3D
uncertainty, learned appearance/ReID, robust occlusion handling, and evaluation
on representative sequences.
