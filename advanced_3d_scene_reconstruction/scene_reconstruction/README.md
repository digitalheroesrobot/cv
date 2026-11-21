# RGB-D Scene Reconstruction Engine

This module implements a transparent classical geometry pipeline:

1. Aligned metric depth + camera intrinsics -> camera-frame point cloud.
2. Voxel downsampling.
3. Statistical outlier removal.
4. Ground-plane RANSAC.
5. Euclidean connected-component clustering.
6. PCA-oriented 3D bounding-box estimation.
7. Optional camera-to-world transformation.

## Assumptions and limitations

- Depth values represent optical-axis Z in metres after unit conversion.
- Color and depth are spatially aligned if color is supplied.
- The pinhole back-projector assumes distortion has been removed or rectified.
- Ground-plane fitting assumes the expected up direction is supplied correctly.
- Euclidean clusters are geometric components, not semantic object identities.
- PCA boxes can be unstable for symmetric objects and are not class-conditioned.
- This baseline does not estimate camera motion, hidden surfaces, or metric depth from monocular RGB.

## Run

From the advanced_3d_scene_reconstruction directory:

```bash
python lesson43_rgbd_to_3d_scene.py --visualize
python lesson43_rgbd_to_3d_scene.py --realsense
python -m unittest -v tests.test_scene_reconstruction
```

Optional dependencies:

```bash
python -m pip install numpy scipy matplotlib opencv-python
```

Core point-cloud and RANSAC operations use NumPy. SciPy accelerates neighbour
search; without SciPy the fallback is intended for small clouds.
