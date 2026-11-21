# Keyframe RGB-D SLAM and Pose Graph

Lessons 47–48 implement a sparse keyframe SLAM backend and geometric loop
verification.

## Components

- `KeyframeManager`: selects frames using translation, rotation, elapsed-frame,
  and tracking-quality thresholds.
- `LoopCandidateDetector`: ranks old keyframes by ORB descriptor matches.
  This is an appearance-based proposal stage only.
- `PoseGraph`: stores camera poses and relative SE(3) constraints, fixes node
  zero to remove gauge freedom, and optimizes remaining poses with robust
  nonlinear least squares.
- `RGBDLoopClosureVerifier`: converts RGB-D PnP output to the graph edge
  convention, refines current-to-candidate cloud alignment with ICP, and
  rejects weak or inconsistent geometry.
- `rebuild_global_map`: transforms every stored keyframe-local cloud using
  optimized poses and voxel-fuses the result.

## Transform convention

Node (i) stores `T_world_camera_i`, mapping camera coordinates into world.
An edge stores `T_i_j = inverse(T_world_camera_i) @ T_world_camera_j`, mapping
points from camera (j) into camera (i).

RGB-D PnP returns `T_current_candidate`. ICP receives current points as source
and candidate points as target, returning `T_candidate_current`. Both
estimates must agree before a loop edge is accepted.

## Run

From `advanced_3d_scene_reconstruction`:

```bash
python lesson47_keyframe_rgbd_slam.py --synthetic --visualize
python lesson48_rgbd_geometric_loop_closure.py --synthetic --visualize
python -m unittest -v tests.test_keyframe_slam
python -m unittest -v tests.test_geometric_loop_closure
```

## Engineering limitations

ORB matching only proposes candidates. Real-image loop closure requires RGB-D
PnP and geometric verification before an edge is inserted. The synthetic
Lesson 48 experiment validates the PnP/ICP consistency gate with known metric
initialization; it does not evaluate real image matching. Point-map rebuilding
is voxel point fusion, not TSDF/mesh fusion. A production system also needs
calibrated uncertainty, robust data association, relocalization, submap
management, dynamic-object filtering, and global map correction at scale.
All runtime computation is local; no network calls or automatic model
downloads are used.


## Lesson 49 — Real-time RGB-D SLAM integration

`realtime.py` connects the existing local ORB RGB-D VO, visual/ICP fusion,
keyframe selection, ORB loop-candidate ranking, RGB-D PnP, ICP geometric
verification, pose-graph optimization, and keyframe point-cloud reintegration.

The D435 runner rectifies registered color/depth using the active camera
intrinsics and distortion coefficients. A loop edge is inserted only after
feature matching, metric PnP, and ICP consistency checks. When tracking is
rejected, the current implementation starts a fresh local submap to keep VO
and ICP temporal reference frames synchronized; it does not silently bridge
an untracked gap.

Run the hardware integration from the course root:

```bash
python lesson49_realtime_rgbd_slam.py --realsense --visualize
```

Press q to stop. The program writes an ASCII PLY global point map and a JSON
trajectory report under `outputs/`. The synthetic mode exercises the graph
and map-reintegration backend only; it is not a camera tracking benchmark.

The implementation is an educational sparse RGB-D SLAM integration, not a
production SLAM system. It does not include IMU fusion, loop-closure
relocalization across submaps, TSDF fusion, photometric bundle adjustment,
semantic dynamic-object masking, or a calibrated probabilistic estimator.
Real loop closure may not occur in every session: sufficient revisited visual
texture, valid depth, overlap, and geometric consistency are required.


## Lesson 50 — Robust SLAM supervision

The `robust_slam.py` module provides reusable components for long-running
SLAM sessions:

- `TrackingMonitor`: hysteretic INITIALIZING/TRACKING/DEGRADED/LOST/
  RELOCALIZING state transitions.
- `SubmapManager`: bounded submap/keyframe/point registry.
- `KeyframeRelocalizationIndex`: ORB descriptor candidate ranking.
- `LoopClosurePolicy`: duplicate suppression and independent PnP/ICP
  transform-consistency gates.
- `information_from_covariance` and `whiten_pose_residual`: 6D uncertainty
  weighting utilities.
- `MemoryBudget` and `SLAMHealthMonitor`: bounded payload accounting and
  rolling operational metrics.

Run from the course root:

```bash
python lesson50_robust_slam.py
python -m unittest -v tests.test_robust_slam
```

These components are integration primitives, not a claim of complete
multi-session SLAM. Candidate retrieval must be followed by metric geometric
verification. Submap geometry alignment and merging must be performed by a
pose-graph/registration backend; metadata merge alone is insufficient.
