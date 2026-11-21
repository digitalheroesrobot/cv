# Advanced Research Track — Lessons 31–42

This is the research-level extension to the 30-lesson foundation course. It covers 2025–2026 directions in feed-forward multi-view geometry, dynamic point maps, open-vocabulary video perception, 3D/4D Gaussian Splatting, Gaussian-SLAM, camera-only occupancy, and occupancy world models.

## Run

From this directory:

```bash
python -m pip install -r ../requirements.txt
python lesson31_vggt_multiview_geometry.py
python lesson34_sam31_open_vocabulary.py
python lesson37_gaussian_splatting.py
python lesson40_camera_4d_occupancy.py
python lesson42_camera_3d_digital_twin.py
```

For a headless server, use:

```bash
python lesson31_vggt_multiview_geometry.py --no-display
```

Each lesson writes a JSON report to `outputs/`. Lessons 39 and 42 also export point-cloud or occupancy artifacts.

## What the code is—and is not

`advanced_core.py` is an executable reference lab with transparent geometry and visualization. It demonstrates point-map conventions, scale/shift depth alignment, optical-flow tracking, a small 2D Gaussian compositor, and voxel occupancy. It does **not** contain or claim to reproduce pretrained VGGT, SAM 3.1, D²USt3R, a full 3DGS/4DGS renderer, Gaussian-SLAM, or an autonomous-driving world model. Those require official model repositories/checkpoints and dedicated integration work.

The research syllabus explains the relevant architecture, equations, assumptions, failure cases, evaluation metrics, and official research references. Start with `README.md` before each lesson.

## Automated checks

```bash
python -m unittest -v test_advanced_core.py
```

The suite verifies camera back-projection, homogeneous transforms, metric depth
scale/shift fitting, voxel deduplication, and smoke-runs all twelve labs.
