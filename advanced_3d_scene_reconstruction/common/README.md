# Shared Data Contracts and Geometry Engine

This package is the stable interface for Lessons 01–66 and the capstone.
It is NumPy-first so that camera I/O, foundation models, Open3D, ROS 2,
Gaussian Splatting, and simulators can exchange data without importing one
another's heavyweight dependencies.

## Canonical types

| Type | Contract |
|---|---|
| `CameraFrame` | Image, timestamp, camera ID, frame index, color order |
| `CameraCalibration` | Intrinsic matrix, distortion, image dimensions |
| `CameraPose` | `T_world_camera`, timestamp, optional 6x6 covariance |
| `DepthMap` | Depth, validity mask, units, optional uncertainty |
| `PointCloud` | Nx3 metric points, optional color/normal/confidence |
| `ObjectTrack` | Position, velocity, dimensions, quaternion, confidence, covariance |
| `OccupancyGrid` | 3D tri-state voxels: unknown=-1, free=0, occupied=1 |

## Coordinate and unit policy

- Camera optical coordinates follow OpenCV: +X right, +Y down, +Z forward.
- A transform is named `T_target_source` and maps source coordinates into target coordinates.
- Point arrays are row-major `(N,3)`; transforms are applied to column-vector semantics.
- Translation, points, dimensions, and depth are metres internally.
- Timestamps are seconds; the producer must document whether its clock is monotonic, sensor, or dataset time.
- Quaternions use `(x,y,z,w)`.
- Occupancy array axis order is `(x,y,z)`; -1 is unknown, 0 free, 1 occupied.

## Geometry engine

`geometry.py` provides validated SE(3) construction, inversion and composition,
point transforms, pinhole projection, Z-depth back-projection, and first-order
covariance propagation.

For a camera point `(X,Y,Z)`, pinhole projection uses
`u = fx*X/Z + cx`, `v = fy*Y/Z + cy`. Back-projection uses the optical-axis
Z-depth, not Euclidean ray range. Lens distortion is stored in calibration,
but distortion/undistortion must be explicitly applied by the camera adapter.

For uncertainty, the first-order approximation is
`Sigma_y = J Sigma_x J^T + Q`. This local linearization is not a substitute
for Monte-Carlo analysis near discontinuities or for large rotations on SE(3).

## Run tests

From `advanced_3d_scene_reconstruction/`:

```bash
python -m unittest -v tests.test_common_geometry
```

No pretrained model, GPU, or camera is required for these tests.
