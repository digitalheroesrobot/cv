# Temporal 3D Mapping — Lesson 44

Local-only sequential point-cloud odometry and mapping baseline.

## Pipeline

1. Voxel-downsample the current camera-frame cloud.
2. Match current points to the previous frame.
3. Estimate source-to-target motion with trimmed point-to-point ICP.
4. Compose relative motion into the fixed world frame.
5. Merge aligned static geometry into a voxelized map.
6. Flag large temporal residuals as dynamic candidates.

## Offline policy

Runtime code uses local Python modules and local camera devices only. It does
not use HTTP clients, cloud inference, telemetry, or automatic checkpoint
downloads. Optional packages are imported only when their feature is used.
Matplotlib visualization runs locally. A NumPy nearest-neighbour fallback is
available for small clouds; SciPy is recommended for larger clouds.

Prepare dependencies before disconnecting the machine:

    python -m pip install numpy scipy matplotlib opencv-python

Then disconnect the network and run:

    python lesson44_temporal_3d_mapping.py --visualize
    python -m unittest -v tests.test_temporal_mapping

## Limitations

This is point-to-point frame-to-frame ICP, not production SLAM. It assumes
sufficient static overlap and a reasonable initial pose. Large motion, low
overlap, repeated geometry, dynamic-dominant scenes, and degenerate geometry
can cause failure. Drift accumulates because this lesson has no loop closure,
pose graph optimization, bundle adjustment, or relocalization. Residual-based
dynamic candidates are heuristic, not semantic motion segmentation.
