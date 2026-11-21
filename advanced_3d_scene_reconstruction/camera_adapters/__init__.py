"""Camera input adapters with a shared CameraFrame/DepthMap contract."""
from .base import CameraAdapter, CameraSample
from .usb_camera import USBCameraAdapter
from .stereo_camera import StereoCameraAdapter
from .realsense_d435 import RealSenseD435Adapter

__all__ = [
    "CameraAdapter", "CameraSample", "USBCameraAdapter",
    "StereoCameraAdapter", "RealSenseD435Adapter",
]
