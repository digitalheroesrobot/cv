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
