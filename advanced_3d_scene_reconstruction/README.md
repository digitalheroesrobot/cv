# Advanced Level: Camera-to-3D Scene Reconstruction

This folder contains an advanced, project-oriented course on reconstructing a spatially meaningful 3D representation of the real world from camera video.

The course is inspired by camera-based 3D scene visualization used in modern automotive perception systems. It progresses from video processing and camera geometry to depth, BEV, 3D occupancy, dynamic scene understanding, reconstruction, and simulation.

> **Important distinction:** A 3D-looking automotive display does not necessarily contain a complete photorealistic reconstruction. Depending on the system, it may render 3D bounding boxes, simplified object meshes, BEV features, occupancy voxels, or reconstructed surfaces. This course teaches these representations separately and explains what each can and cannot recover.

## Learning objectives

By the end of this level, you will be able to:

1. Process recorded or live camera video with Python.
2. Calibrate cameras and transform points between coordinate frames.
3. Estimate depth and create metric point clouds when sufficient sensor or calibration information is available.
4. Detect and segment objects in images.
5. Represent scenes in Bird's-Eye View (BEV) and 3D voxel occupancy grids.
6. Estimate 3D object position, dimensions, orientation, and motion.
7. Reconstruct static and dynamic scenes from multiple observations.
8. Transfer suitable geometry and scene state into a simulator.

## Course structure

### Phase 1 — Camera Video Processing
- Lesson 01: Read Video, Inspect Metadata, and Extract Frames
- Lesson 02: Camera Streams, Frame Timing, and Video Synchronization
- Lesson 03: Image Preprocessing and Dataset Organization

### Phase 2 — Camera Geometry and Calibration
- Lesson 04: Pinhole Camera Model and Intrinsic Parameters
- Lesson 05: Extrinsic Calibration and Coordinate Frames
- Lesson 06: Perspective Projection and 2D-to-3D Back-Projection

### Phase 3 — Detection and Segmentation
- Lesson 07: Object Detection with a Pretrained Detector
- Lesson 08: Semantic and Instance Segmentation
- Lesson 09: Multi-Object Detection Evaluation and Failure Analysis

### Phase 4 — Depth Estimation
- Lesson 10: Monocular Depth Estimation
- Lesson 11: Stereo Geometry and Disparity
- Lesson 12: RGB-D Depth Processing and Metric Point Clouds

### Phase 5 — 3D Object Detection
- Lesson 13: 3D Bounding-Box Parameterization
- Lesson 14: Object Dimensions, Position, and Orientation
- Lesson 15: Camera-to-World 3D Object Coordinates

### Phase 6 — Bird's-Eye View (BEV)
- Lesson 16: Image-to-BEV Geometry
- Lesson 17: Multi-Camera BEV Projection and Fusion
- Lesson 18: BEV Object Detection and Spatial Maps

### Phase 7 — 3D Occupancy
- Lesson 19: Voxel Grids and Occupancy States
- Lesson 20: Semantic 3D Occupancy
- Lesson 21: Temporal Occupancy and Unknown Space

### Phase 8 — 3D Scene Reconstruction
- Lesson 22: Feature Matching and Structure from Motion
- Lesson 23: Multi-View Stereo and Point-Cloud Fusion
- Lesson 24: Mesh Reconstruction and Neural Scene Representations

### Phase 9 — Dynamic Scene Understanding
- Lesson 25: Multi-Object Tracking
- Lesson 26: 3D Velocity and Trajectory Estimation
- Lesson 27: Temporal Scene Representation

### Phase 10 — Simulation and Digital Twins
- Lesson 28: Exporting Point Clouds, Meshes, and Scene Metadata
- Lesson 29: Building a Scene in CARLA, MuJoCo, or Isaac Sim
- Lesson 30: Camera-Driven 3D Digital Twin Capstone

## Recommended technology stack

| Area | Main tools |
|---|---|
| Video and image processing | OpenCV, NumPy |
| Deep learning | PyTorch |
| Object detection | Ultralytics YOLO |
| Segmentation | SAM / compatible segmentation models |
| Depth estimation | Stereo methods, RGB-D, Depth Anything |
| Point clouds | Open3D |
| 3D perception | MMDetection3D and selected research implementations |
| Multi-view reconstruction | COLMAP |
| Automotive simulation | CARLA |
| Robotics simulation | MuJoCo, NVIDIA Isaac Sim |
| Visualization | Matplotlib, Open3D, simulator viewers |

We will install dependencies phase by phase rather than installing the entire stack at once. Model repositories and checkpoints may have their own version and license requirements; each lesson must document the versions it actually uses.

## Recommended progression

1. Start with one recorded RGB camera and learn the complete image/video pipeline.
2. Add calibration and known-scale geometry.
3. Add an RGB-D or stereo source for metric depth.
4. Move to multi-camera BEV and occupancy representations.
5. Add tracking and temporal consistency.
6. Export the scene into a simulator.

A single monocular RGB video generally does not determine absolute metric scale or hidden surfaces by itself. Reliable metric reconstruction needs additional constraints such as calibration, known dimensions, stereo, depth sensors, or external localization.

## Repository conventions

- All course material, comments, explanations, and outputs are in English.
- Each lesson is a complete runnable Python script where practical.
- Scripts include purpose, learning objectives, theory, a real-world application, visualization, and numerical verification.
- Prefer deterministic tests and explicit assumptions.
- Do not silently treat estimated monocular depth as metric ground-truth depth.

## First lesson

Run:

```bash
python -m pip install numpy opencv-python matplotlib
python lesson01_video_processing.py
```

To process a video file:

```bash
python lesson01_video_processing.py --video /path/to/video.mp4
```

The first run without a video creates a small synthetic demo clip, so the pipeline can be tested immediately.

## Capstone

The capstone takes camera observations through perception and 3D scene representation, then exports object poses and geometry for a simulator. The target is a useful spatial digital twin, not merely a visually attractive 3D rendering.


## Advanced research extension (Lessons 31–42)

After completing the 30 core lessons, continue to the research-level track covering feed-forward geometry transformers (VGGT and related point-map models), dynamic 4D correspondences, metric depth uncertainty, SAM 3.1 open-vocabulary video perception, Gaussian Splatting, Gaussian-SLAM, 4D occupancy, and occupancy world models.

- [Advanced Research Track — technical syllabus and papers](advanced_research/README.md)
- [Advanced track — getting started and execution](advanced_research/GETTING_STARTED.md)
- [Advanced mathematical demo engine](advanced_research/advanced_core.py)

The advanced track distinguishes runnable mathematical reference labs from actual pretrained research models. Official checkpoints are optional and must be installed with their own verified dependency and license requirements.


## Research Extension (Lessons 43–66)

Continue beyond the first research track with 24 additional lessons covering transformer-based 3D perception, foundation depth models, metric scale and uncertainty, neural rendering, dynamic 4D reconstruction, multi-camera calibration and fusion, open-world 3D understanding, occupancy forecasting, risk-aware planning interfaces, GPU optimization, streaming reliability, and the final Camera-to-3D World Intelligence capstone.

- [Research Extensions — Lessons 43–66](research_extensions/README.md)


## Shared data contracts and geometry engine

All course tracks now share canonical NumPy data structures and coordinate conventions.
Start here before implementing Lessons 43–66 or connecting external foundation models.

- [Shared data contracts and geometry documentation](common/README.md)
- [Typed data structures](common/data_types.py)
- [SE(3), projection, back-projection, and uncertainty utilities](common/geometry.py)
- [Geometry engine tests](tests/test_common_geometry.py)

Run the contract and geometry tests from this directory:

```bash
python -m unittest -v tests.test_common_geometry
```

The internal length unit is metres; camera optical axes follow OpenCV (+X right,
+Y down, +Z forward); transforms are named `T_target_source`; occupancy is
tri-state (unknown=-1, free=0, occupied=1). These conventions are intentionally
explicit to prevent silent frame, scale, and unknown-space errors across modules.


## Camera input adapters

The shared camera interface now has adapters for USB/UVC, Intel RealSense D435,
and a two-camera stereo capture pair.

- [Camera adapter documentation](camera_adapters/README.md)
- [USB/UVC adapter](camera_adapters/usb_camera.py)
- [RealSense D435 adapter](camera_adapters/realsense_d435.py)
- [Stereo pair adapter](camera_adapters/stereo_camera.py)

Run hardware-free adapter contract tests:

```bash
python -m unittest -v tests.test_camera_adapters
```

The USB adapter requires OpenCV. The D435 adapter additionally requires a
compatible Intel RealSense SDK and `pyrealsense2`. Stereo capture records
timestamp skew but does not claim hardware synchronization; metric stereo
depth requires a separately measured/calibrated baseline and rectification.


## Multi-camera synchronization, calibration, and fusion

The course now includes a shared multi-camera layer:

- [Multi-camera documentation](multi_camera/README.md)
- [Timestamp frame synchronizer](multi_camera/synchronization.py)
- [Chessboard intrinsic/stereo calibration](multi_camera/calibration.py)
- [World-frame point-cloud fusion](multi_camera/fusion.py)

Run hardware-free tests:

```bash
python -m unittest -v tests.test_multi_camera
```

Calibration routines require OpenCV and real calibration images. Timestamp
pairing requires a common clock domain; it is not a substitute for hardware
trigger synchronization. Point-cloud fusion requires externally validated
camera-to-world poses and does not estimate them.


## Lesson 43 — RGB-D to 3D scene reconstruction

The first executable scene-geometry pipeline is available:

- [Lesson 43 runnable script](lesson43_rgbd_to_3d_scene.py)
- [Scene reconstruction engine](scene_reconstruction/geometry_pipeline.py)
- [Engine documentation](scene_reconstruction/README.md)
- [Numerical tests](tests/test_scene_reconstruction.py)

The pipeline performs depth back-projection, voxel downsampling, statistical
outlier removal, ground-plane RANSAC, Euclidean clustering, PCA-oriented
bounding-box estimation, and optional camera-to-world transformation.

Run a synthetic example:

```bash
python lesson43_rgbd_to_3d_scene.py --visualize
python -m unittest -v tests.test_scene_reconstruction
```

For Intel RealSense D435:

```bash
python lesson43_rgbd_to_3d_scene.py --realsense
```

This is a classical RGB-D geometry baseline. It is not a semantic detector,
monocular metric-depth model, SLAM system, or complete surface reconstruction.


## Offline execution policy

All lesson programs are designed to execute locally without internet access.
See [OFFLINE_POLICY.md](OFFLINE_POLICY.md) for dependency preparation, local
model checkpoint rules, and the runtime network prohibition.

## Lesson 44 — Temporal 3D mapping

- [Runnable Lesson 44](lesson44_temporal_3d_mapping.py)
- [Local ICP odometry and map engine](temporal_mapping/odometry.py)
- [Technical notes and limitations](temporal_mapping/README.md)
- [Numerical tests](tests/test_temporal_mapping.py)

Run the synthetic multi-frame mapping example:

```bash
python lesson44_temporal_3d_mapping.py --visualize
python -m unittest -v tests.test_temporal_mapping
```

Lesson 44 uses local point-cloud geometry and does not download models or call
remote inference services. It is a teaching baseline for sequential ICP mapping,
not a loop-closure or globally optimized SLAM implementation.


## Lesson 45 — ORB RGB-D Visual Odometry

- [Runnable Lesson 45](lesson45_rgbd_visual_odometry.py)
- [ORB + RGB-D PnP engine](visual_odometry/feature_vo.py)
- [Technical notes and limitations](visual_odometry/README.md)
- [Numerical tests](tests/test_visual_odometry.py)

Run the local synthetic demo:

```bash
python lesson45_rgbd_visual_odometry.py --synthetic --visualize
python -m unittest -v tests.test_visual_odometry
```

For a locally connected Intel RealSense D435:

```bash
python lesson45_rgbd_visual_odometry.py --realsense
```

Lesson 45 uses ORB descriptors, Essential Matrix RANSAC, and RGB-D PnP.
The Essential Matrix is scale-ambiguous; metric translation comes from
previous-frame depth lifted into 3D. This is a local visual-odometry front-end,
not a complete SLAM system.


## Lesson 46 — RGB-D Visual Odometry + ICP Fusion

- [Runnable Lesson 46](lesson46_visual_icp_fusion.py)
- [Confidence-aware pose fusion engine](sensor_fusion/pose_fusion.py)
- [Fusion design and limitations](sensor_fusion/README.md)
- [Numerical tests](tests/test_sensor_fusion.py)

Run locally:

```bash
python lesson46_visual_icp_fusion.py --synthetic --visualize
python -m unittest -v tests.test_sensor_fusion
```

For a connected Intel RealSense D435:

```bash
python lesson46_visual_icp_fusion.py --realsense --visualize
```

Lesson 46 converts ORB/PnP and ICP into the same transform convention,
scores their quality, gates large disagreements, and fuses consistent
relative poses. The accepted camera pose is accumulated into a bounded
voxel point map. This is a local fusion front-end, not a probabilistic
factor graph or globally optimized SLAM system.

## Lesson 47 — Keyframe RGB-D SLAM and pose graph optimization

Lesson 47 adds sparse keyframe selection, ORB appearance-based loop-closure
candidate ranking, relative-pose graph constraints, a fixed gauge anchor, and
robust nonlinear least-squares optimization.

- [Lesson 47 runnable synthetic experiment](lesson47_keyframe_rgbd_slam.py)
- [Keyframe SLAM engine](keyframe_slam/README.md)
- [Pose graph optimizer](keyframe_slam/pose_graph.py)
- [Keyframe manager and loop candidate detector](keyframe_slam/keyframe_manager.py)
- [Numerical tests](tests/test_keyframe_slam.py)

Run locally:

```bash
python lesson47_keyframe_rgbd_slam.py --synthetic --visualize
python -m unittest -v tests.test_keyframe_slam
```

The synthetic experiment demonstrates how a verified loop constraint reduces
trajectory drift. ORB matching only proposes loop candidates; geometric
verification is mandatory before adding a loop edge. This is a sparse SLAM
backend teaching implementation, not a complete production SLAM system.

