"""Shared typed data contracts and geometry primitives for the course.

All modules are NumPy-first and deliberately independent of optional model stacks.
"""
from .data_types import (
    CameraCalibration, CameraFrame, CameraPose, DepthMap, ObjectTrack,
    OccupancyGrid, PointCloud,
)
from .geometry import (
    backproject_pixels, compose_se3, invert_se3, make_se3, project_points,
    propagate_covariance, transform_points, validate_se3,
)

__all__ = [
    "CameraCalibration", "CameraFrame", "CameraPose", "DepthMap",
    "ObjectTrack", "OccupancyGrid", "PointCloud",
    "backproject_pixels", "compose_se3", "invert_se3", "make_se3",
    "project_points", "propagate_covariance", "transform_points",
    "validate_se3",
]
