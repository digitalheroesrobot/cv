# Lesson 46 — RGB-D Visual Odometry + ICP Fusion

This lesson fuses two estimates of the same relative camera transform.

## Transform convention

- PnP returns `T_current_previous`: previous-camera points to current-camera coordinates.
- ICP is called with current cloud as source and previous cloud as target. It returns
  `T_previous_current`.
- The visual transform is inverted before fusion.
- World pose update:
  `T_world_current = T_world_previous @ T_previous_current`.

## Fusion policy

1. Score visual odometry from PnP inlier ratio and reprojection RMSE.
2. Initialize ICP with the visual estimate when available.
3. Score ICP from correspondence count, convergence, and residual RMSE.
4. Compare translation and rotation disagreement.
5. If estimates agree, interpolate translation and rotation (geodesic SO(3)).
6. If estimates conflict, select the estimate with the higher quality score.
7. If one estimator fails, use the other. If both fail, reject the frame.

This is an explicit heuristic confidence fusion baseline. It is not a Kalman
filter, factor graph, covariance-optimal estimator, or globally consistent SLAM.

## Run locally

    python lesson46_visual_icp_fusion.py --synthetic --visualize
    python lesson46_visual_icp_fusion.py --realsense --visualize
    python -m unittest -v tests.test_sensor_fusion

Dependencies (prepare before going offline):

    python -m pip install numpy scipy opencv-python matplotlib

For D435, install a compatible local `pyrealsense2` / librealsense build.

## Limitations

- ICP needs overlap and a reasonable initialization; it can converge to a local minimum.
- The visual quality score is heuristic and does not model a full pose covariance.
- The current fusion map is a bounded voxel point cloud, not a TSDF or mesh.
- No loop closure, pose graph optimization, bundle adjustment, relocalization,
  dynamic-object masking, or global drift correction is implemented here.
- The synthetic demo tests the point-cloud registration and map pipeline; the
  generated image sequence is a visual stimulus, not ground-truth camera rendering.
