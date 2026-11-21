# Advanced Research Track — Foundation Models, 4D Reconstruction, and 3D World Models

**Position in the curriculum:** This is the research-level extension to Lessons 01–30. It assumes the learner can work with calibrated cameras, point clouds, depth maps, BEV, voxel occupancy, and tracking.

**Target outcome:** Build a camera-driven 3D world model that estimates geometry and camera motion, recognizes and tracks open-vocabulary objects, separates static and dynamic content, maintains a temporally consistent 3D map, predicts near-future occupancy, and renders the result in a real-time viewer.

> Research boundary: the scripts in this folder are runnable mathematical demonstrations and integration scaffolds. They do not claim to reproduce VGGT, SAM 3.1, 4D Gaussian Splatting, or an autonomous-driving foundation model. Real checkpoints, model licenses, CUDA versions, datasets, and hardware requirements must be verified before enabling each model adapter.

## Research map (literature snapshot: 21 November 2026)

The advanced track focuses on several important directions:

- **Feed-forward multi-view geometry transformers:** VGGT predicts camera parameters, depth maps, point maps, and point tracks from one or more views. The VGGT repository announced VGGT-Omega on 18 May 2026; treat this as a new research implementation to inspect, not as a guaranteed drop-in real-time automotive stack.
- **Dynamic point-map reconstruction:** D²USt3R (NeurIPS 2025) explicitly addresses dense correspondences in dynamic regions, where static-scene alignment assumptions break.
- **Promptable open-vocabulary perception:** SAM 3 unifies concept detection, segmentation, and tracking from short text or visual exemplars. SAM 3.1 (March 2026) adds object multiplexing for more efficient multi-object video tracking.
- **Explicit neural scene representations:** 3D Gaussian Splatting (3DGS) offers explicit anisotropic Gaussian primitives and fast differentiable rendering. Dynamic 4DGS extends the representation over time, but rendering speed must not be confused with mapping/update speed.
- **Gaussian-SLAM and neural mapping:** combine camera tracking, geometric map updates, loop closure, and view synthesis. Evaluate tracking accuracy, drift, map completeness, latency, and memory—not just FPS.
- **4D occupancy and world models:** represent what is occupied, free, or unknown over space and time; predict future occupancy conditioned on ego-motion or planned actions.

## Lessons 31–42

### Research Phase A — Foundation geometry and correspondence

**Lesson 31 — Feed-Forward Multi-View Geometry: DUSt3R, MASt3R, VGGT, and VGGT-Omega**
- Technical topics: image tokenization, cross-frame attention, camera tokens, dense point maps, depth heads, point tracks, reference-frame conventions, confidence.
- Math: pinhole projection; rigid transforms in SE(3); reprojection residual; scale/gauge ambiguity.
- Lab: run the mathematical point-map baseline, then connect the official VGGT inference API and export predictions to a common point-cloud schema.
- Critical check: do not assume a single monocular sequence provides metric scale. Validate camera poses and point maps against known calibration or metric sensor data.

**Lesson 32 — Dynamic Multi-View Geometry and D²USt3R**
- Technical topics: static-scene correspondence failure, independently moving objects, 4D point maps, motion-aware matching, robust alignment.
- Lab: compare a static-scene alignment baseline against a dynamic mask-aware correspondence pipeline.
- Metrics: correspondence inlier ratio, 3D alignment residual, moving-object leakage, temporal consistency.

**Lesson 33 — Metric Depth, Scale Recovery, and Uncertainty**
- Technical topics: relative vs metric depth, scale-and-shift alignment, calibrated depth sensors, uncertainty maps, confidence-aware fusion.
- Math: robust scale/shift fit; depth uncertainty propagation through back-projection.
- Lab: align a monocular depth prediction to sparse RealSense/stereo anchors and quantify residuals.
- Safety: never silently convert relative depth into meters.

### Research Phase B — Open-world perception and 4D tracking

**Lesson 34 — Open-Vocabulary Detection and Promptable Concept Segmentation with SAM 3 / SAM 3.1**
- Technical topics: concept prompts, exemplar prompts, detector/tracker coupling, masklets, identity persistence, prompt ambiguity.
- Lab: segment “vehicle”, “pedestrian”, “traffic cone”, and a user-selected rare object; propagate identities over a video.
- Compare: fixed-class detector + tracker vs promptable concept segmentation.
- Deployment: benchmark the exact model/checkpoint, image size, number of tracked objects, GPU, and end-to-end latency.

**Lesson 35 — Dense Point Tracking and 4D Scene Correspondence**
- Technical topics: point tracks, occlusion, disocclusion, track confidence, camera-induced vs object-induced motion.
- Lab: track image points through a clip and lift them into 3D using depth and camera poses.
- Output: time-indexed 3D trajectories and track confidence.

**Lesson 36 — Dynamic Scene Decomposition**
- Technical topics: static background vs independently moving objects, motion segmentation, rigid motion hypotheses, scene flow, temporal masks.
- Lab: estimate camera motion, subtract ego-motion, and identify residual object motion.
- Failure cases: moving camera plus moving objects, rolling shutter, low texture, motion blur, and occlusion.

### Research Phase C — Neural rendering and persistent mapping

**Lesson 37 — 3D Gaussian Splatting from First Principles**
- Technical topics: Gaussian means, covariance, opacity, spherical-harmonic appearance, projection to screen space, alpha compositing, densification and pruning.
- Math: covariance transformation under camera projection; front-to-back compositing.
- Lab: create and render a tiny Gaussian scene; compare point-cloud rendering and Gaussian splatting.
- Distinguish: a visually convincing novel view is not automatically a geometrically accurate or metrically scaled map.

**Lesson 38 — Dynamic 4D Gaussian Splatting**
- Technical topics: time-conditioned deformation, canonical Gaussians, motion fields, dynamic/static decomposition, selective deformation, temporal regularization.
- Lab: animate a small Gaussian scene and measure reconstruction quality vs update cost.
- Compare: per-frame independent maps vs a temporally linked 4D representation.

**Lesson 39 — Gaussian-SLAM and Online Neural Mapping**
- Technical topics: visual odometry, keyframes, local map optimization, loop closure, submaps, map compression, uncertainty.
- Lab: build an online mapping loop with a conventional pose estimator first; attach a Gaussian map renderer as a separate subsystem.
- Metrics: ATE/RPE, drift per meter, map completeness, update latency, GPU memory, relocalization success.

### Research Phase D — Automotive world models and deployment

**Lesson 40 — Camera-Only 3D/4D Occupancy Networks**
- Technical topics: multi-camera image encoders, view transformation, BEV feature aggregation, voxel decoder, semantic occupancy, flow, unknown space.
- Lab: construct a voxel target from depth/poses and compare a geometric baseline with a learned occupancy model.
- Metrics: IoU, class-wise IoU, occupancy precision/recall, calibration, temporal stability.

**Lesson 41 — Occupancy World Models and Action-Conditioned Forecasting**
- Technical topics: latent world state, temporal transformer, occupancy flow, future-state prediction, ego-motion/action conditioning, planning cost.
- Lab: predict future occupancy using a constant-velocity baseline, then define an adapter for a learned world model.
- Evaluate: horizon-wise occupancy IoU, flow error, collision recall, calibration, latency.
- Key distinction: reconstructing the present scene and predicting future scene states are different tasks.

**Lesson 42 — Real-Time Camera-to-3D Digital Twin: System Integration**
- Integrate camera ingestion, calibration, synchronization, perception, depth/geometry, tracking, static/dynamic map, occupancy, uncertainty, and renderer.
- Define interfaces and timestamps for every module. Keep camera, vehicle, world, and map coordinate frames explicit.
- Build a bounded-latency pipeline with asynchronous queues and stale-frame handling.
- Final evaluation: metric scale error, pose drift, 3D detection/segmentation quality, occupancy IoU, track continuity, end-to-end latency, memory, and recovery after tracking failure.

## Architecture target

```text
RGB / Stereo / RGB-D Camera(s)
            |
   Calibration + Time Sync
            |
   Image Foundation Models
   (detector / SAM / depth)
            |
   Multi-view Geometry Transformer
   (camera pose / depth / point maps)
            |
   Static-Dynamic Decomposition
            |
   3D Tracks + Metric Point Cloud
            |
   BEV + Semantic 4D Occupancy
            |
   Persistent Map / World Model
            |
   Gaussian Renderer + Simulator
            |
   3D Digital Twin + Future Prediction
```

## Engineering principles

1. Keep metric geometry separate from appearance rendering. Gaussian splats can look excellent while scale or geometry is wrong.
2. Every 3D point must declare its coordinate frame, units, timestamp, and confidence.
3. Use camera intrinsics/extrinsics and sensor synchronization as first-class data, not hidden constants.
4. Preserve an **unknown** occupancy state. No observation is not evidence of free space.
5. Separate ego-motion from object motion before estimating object trajectories.
6. Track uncertainty through depth, pose, projection, and temporal fusion.
7. Report end-to-end latency, not only neural-network inference time.
8. Test on real sequences with motion blur, low light, repeated textures, occlusion, reflective surfaces, and camera vibration.
9. Keep research model adapters optional so the base course remains runnable without large checkpoints.
10. Verify the current repository, checkpoint license, dependency versions, and hardware support before installation.

## Datasets and evaluation

- nuScenes: multi-camera driving scenes, 3D boxes, tracking, map context.
- KITTI: stereo, odometry, depth, object detection.
- Waymo Open Dataset: large-scale driving perception and motion.
- SemanticKITTI / Occ3D-nuScenes: semantic point-cloud and occupancy evaluation.
- TUM RGB-D / EuRoC: camera tracking and geometry evaluation.
- RealSense D435 recordings: metric depth and camera-to-world calibration exercises.

Use official dataset splits and protocols. Do not compare numbers across papers unless sensor setup, resolution, split, metric definition, and evaluation code are aligned.

## Primary references

- VGGT (CVPR 2025): https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html
- VGGT implementation and VGGT-Omega update: https://github.com/facebookresearch/vggt
- D²USt3R (NeurIPS 2025): https://cvlab-kaist.github.io/DDUSt3R/
- SAM 3 research paper: https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- SAM 3 / SAM 3.1 update: https://ai.meta.com/blog/segment-anything-model-3/
- Feed-forward point-cloud reconstruction survey (2026): https://www.sciencedirect.com/science/article/pii/S2096579626000203
- 3DGS-SLAM survey (2026): https://arxiv.org/abs/2602.04251
- 4D occupancy world model, Drive-OccWorld (AAAI 2025): https://drive-occworld.github.io/
- 2026 BEV occupancy survey: https://www.mdpi.com/2227-7390/14/15/2720
- Dynamic 4D Gaussian splatting research (2026): https://www.sciencedirect.com/science/article/pii/S1568494626016698

## Recommended order

Complete Lessons 01–30 to understand the geometry and data contracts. Then study Lessons 31–42. In practice, the strongest path to the capstone is 31 → 33 → 34 → 35 → 36 → 37 → 39 → 40 → 41 → 42. Lessons 32 and 38 are deeper research electives.
