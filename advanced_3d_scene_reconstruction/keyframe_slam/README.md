# Keyframe RGB-D SLAM and Pose Graph

Lesson 47 introduces a sparse keyframe SLAM backend.

## Components

- `KeyframeManager`: selects frames using translation, rotation, elapsed-frame,
  and tracking-quality thresholds.
- `LoopCandidateDetector`: ranks old keyframes by ORB descriptor matches.
  This is an appearance-based proposal stage only.
- `PoseGraph`: stores camera poses and relative SE(3) constraints, fixes node
  zero to remove gauge freedom, and optimizes the remaining poses with SciPy
  robust nonlinear least squares.

## Transform convention

Node (i) stores `T_world_camera_i`, mapping camera coordinates into world.
An edge stores `T_i_j = inverse(T_world_camera_i) @ T_world_camera_j`, mapping
points from camera (j) into camera (i).

The residual is based on:

`Log(inverse(T_i_j_measurement) @ inverse(T_world_camera_i) @ T_world_camera_j)`.

Rotation residuals use an SO(3) rotation vector; translation residuals use
metres. Edge information weights scale the corresponding residual components.

## Run

From `advanced_3d_scene_reconstruction`:

```bash
python lesson47_keyframe_rgbd_slam.py --synthetic --visualize
python -m unittest -v tests.test_keyframe_slam
```

## Engineering limitations

The synthetic lesson validates graph optimization using deterministic relative
constraints. It does not claim a complete camera loop-closure system. A real
loop candidate must be verified geometrically (for example, RGB-D PnP followed
by ICP consistency checks) before an edge is inserted. A production SLAM system
also needs calibrated uncertainty, robust data association, relocalization,
submap management, IMU integration when available, and global map correction.
All lesson runtime computation is local; no network calls or automatic model
downloads are used.
