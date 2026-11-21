# Research Extensions — Lessons 43–66

**Course:** Camera-to-3D World Intelligence  
**Language:** English  
**Prerequisites:** Core Lessons 01–30 and Research Lessons 31–42  
**Status:** Curriculum specification and implementation plan. Individual executable labs will be added and validated incrementally.

This extension deepens the existing course into a research-oriented engineering program. It does not replace Lessons 01–42. The target is a robust camera-to-3D system that can ingest supported cameras, estimate geometry and motion, maintain a temporally consistent world representation, and expose measurable outputs to robotics or autonomous systems.

> **Research integrity:** A lesson may contain a runnable mathematical baseline, an adapter to an official research implementation, or a full model reproduction. These are different deliverables and must be labeled accurately. A paper citation or placeholder adapter is not evidence that a pretrained model has been reproduced.

## Course outcomes

By the end of Lessons 43–66, the learner should be able to:
- Compare feed-forward geometry models, classical multi-view geometry, and learned depth models under controlled conditions.
- Build robust camera calibration, synchronization, pose, depth, and uncertainty interfaces.
- Fuse multiple camera observations into common metric coordinates without silently mixing coordinate frames.
- Implement neural scene representations and evaluate geometry separately from rendering quality.
- Track dynamic objects and estimate scene flow while compensating for ego-motion.
- Build and evaluate occupancy and future-state prediction pipelines.
- Profile end-to-end throughput, latency, memory, and failure recovery.
- Package the system as a reproducible research project with datasets, configs, tests, metrics, and deployment interfaces.

## Module 15 — Transformer-based 3D Perception

### Lesson 43 — Vision Transformers for Multi-View Geometry
- **Theory:** patch/token embeddings, positional encoding, cross-view attention, camera tokens, feature aggregation, dense prediction heads.
- **Math:** scaled dot-product attention, pinhole projection, rigid transforms in SE(3), gauge freedom.
- **Lab:** compare pairwise feature matching, a geometric multi-view baseline, and a feed-forward point-map interface on the same image set.
- **Tests:** camera-frame convention, reprojection residual, scale ambiguity, deterministic inference shape checks.
- **Deliverable:** common prediction schema for camera poses, depth, point maps, confidence, and source image IDs.

### Lesson 44 — Geometry Foundation Model Adapters
- **Theory:** model preprocessing, image resizing/cropping, normalization, checkpoint formats, inference precision, confidence outputs.
- **Lab:** create a versioned adapter boundary for an officially released geometry model such as VGGT; export results to NumPy/Open3D formats.
- **Tests:** input/output contract, finite values, coordinate convention, checkpoint hash, reproducibility metadata.
- **Deliverable:** adapter interface plus a baseline fallback that works without downloading a checkpoint.

### Lesson 45 — Benchmarking Learned Geometry Against SfM
- **Theory:** learned feed-forward geometry vs incremental SfM, matching, bundle adjustment, robust estimation, runtime/quality trade-offs.
- **Lab:** evaluate a fixed image sequence with COLMAP and one available feed-forward model.
- **Metrics:** rotation/translation error where ground truth exists, reprojection error, point-cloud alignment, completeness, runtime, peak memory.
- **Deliverable:** benchmark report with explicit dataset split and metric definitions.

## Module 16 — Foundation Depth and Metric Scale

### Lesson 46 — Monocular Depth Foundation Models
- **Theory:** relative depth, metric depth, scale/shift ambiguity, domain shift, depth priors, resolution effects.
- **Lab:** compare at least two depth predictors where compatible checkpoints are available; otherwise run a deterministic synthetic baseline and keep model adapters optional.
- **Metrics:** AbsRel, SqRel, RMSE, RMSE-log, delta thresholds, valid-pixel coverage.
- **Deliverable:** depth prediction interface that labels metric status and units.

### Lesson 47 — Metric Scale Recovery and Sensor Anchoring
- **Theory:** scale alignment, scale-and-shift fitting, stereo baseline, RGB-D calibration, known-size anchors, robust regression.
- **Lab:** align predicted depth to RealSense/stereo sparse depth anchors and evaluate held-out anchors.
- **Tests:** unit consistency, outlier robustness, no leakage between fit and evaluation points.
- **Deliverable:** calibrated metric-depth map and residual visualization.

### Lesson 48 — Depth Uncertainty and Probabilistic Fusion
- **Theory:** aleatoric/epistemic uncertainty, confidence calibration, uncertainty propagation through back-projection, robust fusion.
- **Math:** first-order covariance propagation using the Jacobian: covariance_out ≈ J covariance_in J-transpose.
- **Lab:** propagate pixel/depth uncertainty into 3D point covariance and compare weighted vs unweighted point fusion.
- **Metrics:** calibration error, negative log likelihood where applicable, 3D residual normalized by uncertainty.
- **Deliverable:** point cloud with per-point confidence/covariance.

## Module 17 — Neural Rendering and Scene Representations

### Lesson 49 — Neural Radiance Fields and Differentiable Rendering
- **Theory:** radiance fields, volume rendering, ray sampling, transmittance, positional encoding, photometric objectives.
- **Lab:** implement a tiny educational ray marcher; optionally train a small NeRF on a bounded sample dataset.
- **Metrics:** PSNR, SSIM, LPIPS when dependencies and reference images are available; also report training/inference time.
- **Deliverable:** novel-view rendering baseline with documented compute requirements.

### Lesson 50 — 3D Gaussian Splatting Optimization
- **Theory:** anisotropic covariance, projection Jacobian, opacity, spherical harmonics, alpha compositing, densification, pruning.
- **Lab:** inspect an official 3DGS implementation and run a small licensed sample scene; compare against a point-cloud renderer.
- **Metrics:** PSNR/SSIM/LPIPS plus geometry checks such as depth error and point-to-surface distance.
- **Deliverable:** Gaussian scene artifact and evaluation report.

### Lesson 51 — Hybrid Geometry: Points, Meshes, TSDF, and Gaussians
- **Theory:** representation trade-offs, surface extraction, truncation distance, topology, map updates, rendering vs collision geometry.
- **Lab:** convert a depth sequence into a TSDF/mesh and compare it with point-cloud and Gaussian representations.
- **Metrics:** completeness, surface distance, memory footprint, update latency, collision-query suitability.
- **Deliverable:** representation converter and decision matrix for downstream robotics use.

## Module 18 — Dynamic 4D Scene Understanding

### Lesson 52 — Scene Flow and 3D Motion Estimation
- **Theory:** optical flow, depth lifting, ego-motion compensation, rigid flow, non-rigid residual motion.
- **Lab:** estimate 2D flow, lift correspondences into 3D, and estimate per-point scene flow.
- **Metrics:** endpoint error, 3D flow error, static-scene residual after ego-motion compensation.
- **Deliverable:** time-indexed 3D motion field.

### Lesson 53 — Dynamic Gaussian Representations
- **Theory:** canonical space, time-conditioned deformation, motion fields, temporal regularization, dynamic/static decomposition.
- **Lab:** animate a compact synthetic Gaussian scene; connect an official implementation only after dependency and checkpoint verification.
- **Metrics:** temporal rendering quality, trajectory consistency, deformation smoothness, update cost.
- **Deliverable:** time-conditioned scene representation with documented motion convention.

### Lesson 54 — Persistent Object Identity and Long-Term Tracking
- **Theory:** detection-to-track association, appearance embeddings, 3D motion gating, occlusion handling, re-identification, track lifecycle.
- **Lab:** compare image-space tracking with 3D-aware association using camera pose and depth.
- **Metrics:** HOTA, IDF1, MOTA, identity switches, track fragmentation, 3D position error.
- **Deliverable:** persistent object tracks in world coordinates.

## Module 19 — Multi-Camera Calibration and Fusion

### Lesson 55 — Multi-Camera Intrinsic and Extrinsic Calibration
- **Theory:** pinhole/fisheye models, distortion, target-based calibration, hand-eye calibration, observability, calibration covariance.
- **Lab:** calibrate a camera rig and estimate transforms into a rig/world frame.
- **Tests:** reprojection error, transform inverse consistency, loop closure across frame transforms.
- **Deliverable:** versioned rig calibration file with units and coordinate-frame documentation.

### Lesson 56 — Time Synchronization and Rolling-Shutter Effects
- **Theory:** hardware/software timestamps, clock drift, trigger synchronization, rolling shutter, temporal interpolation.
- **Lab:** synchronize asynchronous video streams and quantify geometric error induced by timestamp offset.
- **Metrics:** timestamp skew, drift, reprojection change, moving-object position error.
- **Deliverable:** synchronized multi-camera packet format.

### Lesson 57 — Cross-Camera Feature, Object, and BEV Fusion
- **Theory:** early/late fusion, feature-level fusion, calibration-aware projection, visibility, occlusion, duplicate suppression.
- **Lab:** fuse detections or features from a small camera rig into a common BEV/world representation.
- **Metrics:** coverage, duplicate rate, localization error, cross-camera association accuracy, latency.
- **Deliverable:** multi-camera fusion module with per-camera provenance.

## Module 20 — Open-World 3D Understanding

### Lesson 58 — Open-Vocabulary Detection and Concept Segmentation
- **Theory:** text-conditioned visual representations, concept prompts, prompt ambiguity, class-agnostic masks, model/tracker coupling.
- **Lab:** evaluate a supported open-vocabulary detector and promptable segmentation model on selected common and rare object categories.
- **Metrics:** class-agnostic mask quality, prompt consistency, latency, false-positive rate, track continuity.
- **Deliverable:** open-world perception adapter with prompt and model provenance.

### Lesson 59 — Language-Grounded 3D Scene Graphs
- **Theory:** object-centric scene graphs, spatial relations, grounding, entity persistence, uncertainty-aware symbolic facts.
- **Lab:** convert 3D object tracks into relations such as left-of, behind, inside, near, and moving-toward.
- **Tests:** relation consistency under coordinate transforms; uncertainty and missing-observation handling.
- **Deliverable:** timestamped 3D scene graph with links to source observations.

### Lesson 60 — Unknown Objects, Novelty, and Out-of-Distribution Detection
- **Theory:** open-set recognition, embedding distance, density/energy scores, novelty thresholds, unknown-class handling.
- **Lab:** detect out-of-distribution examples and preserve unknown objects instead of forcing a known class.
- **Metrics:** AUROC/AUPR for OOD splits, false unknown rate, unknown-object recall.
- **Deliverable:** perception output that separates known classes from unknown candidates.

## Module 21 — World Models and Future Occupancy

### Lesson 61 — Temporal BEV and 4D Occupancy
- **Theory:** temporal feature aggregation, voxel occupancy, semantic occupancy, occupancy flow, unknown/free/occupied semantics.
- **Lab:** build a geometry-based temporal occupancy baseline from calibrated depth and poses.
- **Metrics:** voxel IoU, class-wise IoU, occupancy precision/recall, temporal consistency.
- **Deliverable:** timestamped 4D occupancy tensor with explicit coordinate metadata.

### Lesson 62 — Occupancy Forecasting and Action Conditioning
- **Theory:** temporal transformers, latent state, ego-motion conditioning, action-conditioned prediction, horizon-dependent uncertainty.
- **Lab:** compare constant-velocity occupancy forecasting with a learned-model adapter when a compatible public checkpoint is available.
- **Metrics:** horizon-wise IoU, flow error, collision recall, calibration, inference latency.
- **Deliverable:** future occupancy predictions with uncertainty and horizon labels.

### Lesson 63 — Planning Interfaces and Risk-Aware Spatial Costs
- **Theory:** collision checking, reachable sets, uncertainty-aware costs, dynamic obstacles, conservative unknown-space handling.
- **Lab:** create a simple BEV/voxel costmap and evaluate candidate trajectories against predicted occupancy.
- **Metrics:** collision detection recall, false-safe rate, planning time, minimum clearance.
- **Deliverable:** planner-facing spatial interface; educational integration, not a certified safety system.

## Module 22 — Real-Time Optimization and Deployment

### Lesson 64 — GPU Profiling and Model Runtime Optimization
- **Theory:** throughput vs latency, batching, mixed precision, memory bandwidth, CUDA streams, TensorRT, quantization.
- **Lab:** profile a selected model pipeline before and after one optimization; preserve numerical comparison.
- **Metrics:** p50/p95/p99 latency, FPS, peak GPU memory, power where measurable, output deviation.
- **Deliverable:** reproducible benchmark harness with hardware/software manifest.

### Lesson 65 — Streaming Architecture, Fault Tolerance, and Observability
- **Theory:** asynchronous stages, bounded queues, back-pressure, frame dropping, stale-data policy, watchdogs, structured logs.
- **Lab:** build a multi-stage streaming pipeline with synthetic load and injected failures.
- **Metrics:** end-to-end latency, dropped-frame ratio, queue depth, recovery time, stale-output rate.
- **Deliverable:** resilient pipeline skeleton with health and timing telemetry.

### Lesson 66 — Production-Style Camera-to-3D World Intelligence Capstone
- **System:** camera ingestion → calibration/time sync → perception → depth/geometry → pose/motion → static/dynamic map → occupancy/world model → viewer/export.
- **Inputs:** recorded RGB, live USB camera, RealSense RGB-D, stereo, and multiple cameras where hardware is available.
- **Outputs:** metric point cloud where scale is observable, camera/world poses, 3D object boxes, persistent tracks, BEV, semantic occupancy, scene graph, and digital-twin visualization.
- **Evaluation:** metric scale error, camera pose drift, 3D localization error, tracking continuity, occupancy IoU, end-to-end latency, peak memory, failure recovery.
- **Deliverable:** complete reproducible project, configuration files, tests, benchmark report, and engineering limitations document.

## Global engineering standards

### Coordinate frames and units
Every artifact must declare its coordinate frame, axis convention, handedness, length unit, timestamp clock, and calibration version. Use named transforms such as `T_world_camera` and document transform direction. Never mix camera, rig, vehicle, and world coordinates implicitly.

### Metric integrity
Relative monocular depth is not metric depth. Scale recovery must use an explicit anchor (stereo baseline, RGB-D, known dimensions, calibrated motion, or external localization) and must be evaluated on held-out measurements.

### Unknown space
Occupancy has at least three semantic states: free, occupied, and unknown. Unobserved space must not be silently labeled free.

### Dynamic scenes
Estimate ego-motion separately from independently moving objects. Preserve observation timestamps and track confidence. Handle occlusion, disocclusion, rolling shutter, and dropped frames explicitly.

### Evaluation and reproducibility
Each experiment records dataset/version/split, calibration, model/checkpoint hash, package versions, hardware, random seed, command line, metrics, and output artifacts. Report failures and uncertainty, not only successful visualizations.

### Deployment measurements
Measure end-to-end sensor-to-output latency, not only neural network forward-pass time. Report percentile latency and dropped frames. Rendering FPS alone is not a measure of perception quality.

## Suggested implementation order

1. Complete the curriculum specification and dependency matrix.
2. Add shared data contracts for frames, calibration, poses, depth, point clouds, tracks, occupancy, and scene graphs.
3. Implement deterministic CPU mathematical baselines and unit tests.
4. Add optional model adapters behind feature flags.
5. Validate official model repositories, licenses, checkpoints, CUDA/PyTorch compatibility, and hardware requirements.
6. Add real camera integration and synchronized multi-camera tests.
7. Complete the capstone only after individual modules pass their tests.

## Research references and official project entry points

These are starting points for literature review and implementation verification; check each project's current license, release, checkpoint availability, and environment instructions before use.

- [VGGT — Visual Geometry Grounded Transformer (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.html)
- [VGGT official repository](https://github.com/facebookresearch/vggt)
- [D²USt3R — Dynamic DUSt3R (NeurIPS 2025)](https://cvlab.kaist.ac.kr/DDUSt3R/)
- [Segment Anything Model 3 research](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- [Meta SAM 3.1 announcement](https://ai.meta.com/blog/segment-anything-model-3/)
- [3D Gaussian Splatting project](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [COLMAP documentation](https://colmap.github.io/)
- [Open3D documentation](https://www.open3d.org/docs/)
- [CARLA documentation](https://carla.readthedocs.io/)
- [ROS 2 documentation](https://docs.ros.org/)

The curriculum is intentionally broader than any single paper or model. Model-specific lessons must be updated when official implementations, checkpoints, or evaluation protocols change.
