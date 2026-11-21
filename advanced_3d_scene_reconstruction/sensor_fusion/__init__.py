"""Confidence-aware local RGB-D visual odometry and ICP fusion."""
from .pose_fusion import (
    PoseFusionResult, RGBDVisualICPFusion,
    fuse_relative_poses, rotation_distance_rad,
)
__all__ = ["PoseFusionResult", "RGBDVisualICPFusion",
           "fuse_relative_poses", "rotation_distance_rad"]
