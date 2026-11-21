# Lesson-by-Lesson Study Guide

All lessons are runnable independently. Each lesson uses a deterministic synthetic road scene when no real video is supplied, so you can start without downloading a dataset. Real video can be passed with `--video path/to/video.mp4`.

## Execution
From this directory:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python lesson02_camera_timing.py
```

Every lesson writes a JSON report to `outputs/`. Geometry lessons also save point-cloud or scene artifacts where applicable.

## Study workflow
1. Run one lesson.
2. Inspect its visualization and JSON report.
3. Change one parameter and run it again.
4. Compare the result with the expected geometry.
5. Discuss the result and only then move to the next lesson.

## Important engineering rule
A camera image is a 2D projection. A useful 3D scene requires camera calibration and depth/scale information. Monocular depth networks often provide relative depth; do not interpret their output as metric distance without calibration or another scale source.

## Run the automated checks

From this directory:

```bash
python -m unittest -v test_course_core.py
```

The tests check the pinhole projection equations, expected inverse-depth scaling,
synthetic frame validity, report generation, and invalid lesson handling.

### Lesson 47 — Keyframe RGB-D SLAM and Pose Graph Optimization

- Keyframe selection from translation, rotation, frame gap, and tracking quality.
- ORB descriptor matching for appearance-based loop candidate ranking.
- Relative SE(3) pose graph with explicit transform direction and a fixed gauge anchor.
- Robust nonlinear least-squares optimization with weighted rotation/translation residuals.
- Synthetic square-loop experiment demonstrating global drift correction.

Run:
```bash
python lesson47_keyframe_rgbd_slam.py --synthetic --visualize
python -m unittest -v tests.test_keyframe_slam
```

A visual match is only a loop candidate. Add a loop edge only after geometric verification using RGB-D constraints such as PnP and ICP.



### Lesson 48 — RGB-D Geometric Loop Closure and Global Map Correction

- RGB-D PnP metric initialization for candidate-to-current alignment.
- ICP refinement and correspondence/RMSE quality checks.
- Reject loop constraints when PnP and ICP estimates disagree.
- Insert only geometrically verified loop edges into the pose graph.
- Rebuild the global point map from keyframe-local clouds and optimized poses.

Run:
```bash
python lesson48_rgbd_geometric_loop_closure.py --synthetic --visualize
python -m unittest -v tests.test_geometric_loop_closure
```

The current synthetic experiment tests geometry verification with known PnP
initialization. It does not evaluate a real image retrieval or RGB-D PnP pipeline.


### Lesson 49 — Real-Time RGB-D SLAM Integration

- Integrate RealSense D435 aligned color/depth capture and camera calibration.
- Rectify RGB-D using the active camera distortion coefficients.
- Fuse ORB RGB-D visual odometry and ICP frame-to-frame registration.
- Select and store keyframes with local RGB-D point clouds and ORB descriptors.
- Propose revisited locations and verify them using RGB-D PnP plus ICP.
- Add verified loop edges, optimize the sparse pose graph, and reintegrate the global point map.
- Save an offline PLY point map and JSON trajectory report.

Run with D435:
```bash
python lesson49_realtime_rgbd_slam.py --realsense --visualize
```

Run synthetic graph/map backend and tests:
```bash
python lesson49_realtime_rgbd_slam.py --synthetic --visualize
python -m unittest -v tests.test_realtime_slam
```

Tracking failure starts a fresh local submap so visual odometry and ICP do not
silently use mismatched previous-frame references. This baseline does not
include IMU, TSDF, semantic dynamic masking, or production-grade relocalization.


### Lesson 50 — Robust SLAM, Submap Management and Relocalization

- Tracking degradation/lost detection with hysteresis and controlled recovery.
- Bounded submap management and keyframe/point memory limits.
- Appearance-based keyframe relocalization candidate ranking.
- Duplicate loop suppression and PnP/ICP consistency checks.
- 6x6 covariance-to-information conversion and residual whitening.
- Rolling operational health metrics for frame failures and timing gaps.

Run locally:

```bash
python lesson50_robust_slam.py
python -m unittest -v tests.test_robust_slam
```

The lab uses deterministic inputs. Appearance retrieval is not a metric
relocalization result; a real RGB-D system must still validate candidate poses
geometrically. Memory estimates cover registered payload arrays rather than
the complete Python process or GPU memory.


### Lesson 51 — Multi-Camera RGB-D World Reconstruction

- Validate timestamp skew across multiple camera observations.
- Rectify each camera's registered RGB-D data using its own calibration.
- Back-project metric depth into camera-local point clouds.
- Transform each cloud using calibrated `T_world_camera` extrinsics.
- Fuse overlapping clouds in a common world coordinate frame.
- Reduce duplicate geometry using voxel aggregation and export PLY.

Run offline:

```bash
python lesson51_multi_camera_reconstruction.py --visualize
python -m unittest -v tests.test_multi_camera_reconstruction
```

The synthetic demo validates transform direction and fusion plumbing using
known extrinsics. It does not estimate camera extrinsics. Real cameras require
a measured rig calibration and a shared timestamp clock domain; software
synchronization alone is not hardware synchronization.


### Lesson 52 — Multi-Camera Extrinsic Calibration and Camera Rig Optimization

- Define explicit `T_rig_camera` and pairwise `T_camera_j_camera_i` conventions.
- Convert pairwise OpenCV stereo calibration results into graph constraints.
- Require a connected camera calibration graph and fix one gauge anchor.
- Optimize all camera extrinsics with uncertainty-weighted robust least squares.
- Inspect per-edge translation/rotation residuals and approximate covariance.
- Validate rig poses against synthetic ground truth and held-out physical captures.

Run offline:

```bash
python lesson52_camera_rig_calibration.py --visualize
python -m unittest -v tests.test_rig_calibration
```

For physical calibration, calibrate intrinsics first, collect synchronized
chessboard pairs for each graph edge, run stereo calibration, and optimize the
connected rig graph. The synthetic demo validates transform conventions and
optimization only; it does not replace real target capture or held-out
calibration validation.


### Lesson 53 — Multi-Camera Clock Synchronization and Temporal Alignment

- Distinguish sensor timestamps, host arrival timestamps, and master-clock time.
- Estimate affine clock offset and drift from paired synchronization events.
- Reject synchronization-event outliers with robust residual statistics.
- Convert independent camera clocks into one master time domain.
- Pair nearest frames under a configurable maximum skew.
- Account for duplicates, out-of-order frames, queue overflow, and stale frames.
- Interpolate camera poses using linear translation and SO(3) SLERP.

Run offline:

```bash
python lesson53_multicamera_time_synchronization.py --visualize
python -m unittest -v tests.test_time_sync
```

Clock calibration requires paired observations of the same event. Arbitrary
frame-arrival timestamps are contaminated by variable transport latency.
Software alignment does not synchronize sensor exposure; validate hardware
triggering and timestamp domains separately for a real multi-D435 rig.


### Lesson 54 — Real-Time Multi-Camera RGB-D World Reconstruction

- Run one camera-adapter reader thread per physical camera.
- Normalize RGB and depth timestamps into a configured master clock.
- Maintain bounded queues and pair complete multi-camera bundles by timestamp.
- Handle stale frames, out-of-order timestamps, queue overflow, and reader failures.
- Apply calibrated camera-rig extrinsics and fuse point clouds in world coordinates.
- Monitor bundle rate, fusion latency, timestamp skew, queue depth, and frame drops.
- Export the latest temporally coherent world-frame point-cloud snapshot.

Run offline:

```bash
python lesson54_realtime_multicamera_world.py --cameras 3 --frames 30 --visualize
python -m unittest -v tests.test_multicamera_capture
```

For live D435 capture, supply device serials and measured `T_world_camera`
extrinsics. Supply clock calibration when devices do not already share a
validated timestamp domain. Software timestamp pairing is not hardware
exposure synchronization. The PLY output is a latest-frame snapshot, not a
persistent dynamic SLAM map.


### Lesson 55 — Multi-Camera 3D Object Tracking and Data Association

- Represent per-camera detections as metric 3D observations in the common world frame.
- Fuse duplicate detections across cameras while preserving separate same-camera objects.
- Use Hungarian assignment for globally consistent frame-to-track matching.
- Gate associations by semantic class, 3D Mahalanobis distance, object dimensions, and optional appearance embeddings.
- Estimate position and velocity with a constant-velocity Kalman filter.
- Maintain track IDs through short observation gaps and expire stale tracks.
- Export trajectories, velocities, track diagnostics, and a 3D visualization.

Run offline:

```bash
python lesson55_multicamera_3d_tracking.py --frames 100 --visualize
python -m unittest -v tests.test_object_tracking
```

This lesson consumes detections that have already been transformed into a
shared metric world frame. The synthetic demo validates the tracking backend;
it is not a neural detector, learned ReID model, or production multi-camera MOT
benchmark.
