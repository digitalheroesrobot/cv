# Lesson-by-Lesson Study Guide

All lessons are runnable independently. Each lesson uses a deterministic synthetic road scene when no real video is supplied, so you can start without downloading a dataset. Real video can be passed with `--video path/to/video.mp4`.

## Execution
From this directory:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python lesson02_camera_timing.py
```

Every lesson writes a JSON report to `outputs/`. Geometry lessons also save point-cloud or scene artifacts where applicable.

## Study workflow
1. Run one lesson.
2. Inspect its visualization and JSON report.
3. Change one parameter and run it again.
4. Compare the result with the expected geometry.
5. Discuss the result and only then move to the next lesson.

## Important engineering rule
A camera image is a 2D projection. A useful 3D scene requires camera calibration and depth/scale information. Monocular depth networks often provide relative depth; do not interpret their output as metric distance without calibration or another scale source.
