"""Multi-camera synchronization, calibration, and point-cloud fusion."""
from .synchronization import FrameSynchronizer, SynchronizedFrames
from .calibration import (
    StereoCalibrationResult, calibrate_stereo_from_chessboards,
    calibrate_intrinsics_from_chessboards, reprojection_rmse,
)
from .fusion import fuse_point_clouds

__all__ = [
    "FrameSynchronizer", "SynchronizedFrames", "StereoCalibrationResult",
    "calibrate_stereo_from_chessboards", "calibrate_intrinsics_from_chessboards",
    "reprojection_rmse", "fuse_point_clouds",
]
