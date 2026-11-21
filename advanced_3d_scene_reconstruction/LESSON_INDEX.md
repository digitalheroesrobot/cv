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
