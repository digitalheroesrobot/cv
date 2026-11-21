"""Multi-camera synchronization, calibration, and point-cloud fusion."""
from .synchronization import FrameSynchronizer, SynchronizedFrames
from .calibration import (
    StereoCalibrationResult,
    calibrate_stereo_from_chessboards,
    calibrate_intrinsics_from_chessboards,
    reprojection_rmse,
)
from .fusion import fuse_point_clouds
from .reconstruction import (
    MultiCameraFrameResult,
    MultiCameraReconstructor,
    rectify_registered_rgbd,
)
from .rig_calibration import (
    RigCalibrationResult,
    RigEdge,
    RigPoseGraph,
    calibrate_camera_rig,
    stereo_result_to_rig_edge,
)
from .capture import CaptureBundle, MultiCameraCaptureManager
from .object_tracking import (
    Detection3D, FusedDetection3D, MultiCameraDetectionFusion,
    ConstantVelocityKalman, MultiCamera3DTracker,
)
from .time_sync import (
    ClockFitResult, ClockModel, MultiCameraClockSynchronizer,
    fit_clock_model, interpolate_camera_pose,
)

__all__ = [
    "FrameSynchronizer", "SynchronizedFrames", "StereoCalibrationResult",
    "calibrate_stereo_from_chessboards", "calibrate_intrinsics_from_chessboards",
    "reprojection_rmse", "fuse_point_clouds", "MultiCameraFrameResult",
    "MultiCameraReconstructor", "rectify_registered_rgbd", "RigCalibrationResult",
    "RigEdge", "RigPoseGraph", "calibrate_camera_rig",
    "stereo_result_to_rig_edge", "ClockFitResult", "ClockModel",
    "MultiCameraClockSynchronizer", "fit_clock_model",
    "interpolate_camera_pose", "CaptureBundle",
    "MultiCameraCaptureManager", "Detection3D", "FusedDetection3D",
    "MultiCameraDetectionFusion", "ConstantVelocityKalman",
    "MultiCamera3DTracker",
]
