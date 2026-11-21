#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LESSON 17 — Multi-Camera BEV Projection and Fusion

LEARNING OBJECTIVE
------------------
Multi-Camera BEV Projection and Fusion. Run this lesson, inspect its visual output and JSON report, then
modify the parameters and discuss the result before moving on.

RUN
---
    pip install -r requirements.txt
    python lesson17_multi_camera_bev.py
    python lesson17_multi_camera_bev.py --video path/to/drive_video.mp4

The shared course engine provides a deterministic synthetic road scene so the
lesson can run without a dataset or pretrained weights. See LESSON_INDEX.md
for the study workflow and course_core.py for the complete implementation.
"""
from course_core import main

if __name__ == "__main__":
    main(17)
