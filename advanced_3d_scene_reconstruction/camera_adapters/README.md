# Camera Input Adapters

All camera sources normalize their output to the shared course contracts:
`CameraFrame`, optional `CameraCalibration`, and optional `DepthMap`.

## USB / UVC camera

```python
from camera_adapters import USBCameraAdapter

with USBCameraAdapter(device=0, width=1280, height=720, fps=30) as camera:
    sample = camera.read()
    if sample is not None:
        print(sample.frame.image.shape, sample.frame.timestamp)
```

Requires `opencv-python`. Requested resolution/FPS are driver requests, not
guarantees; query `actual_properties()` after opening.

## Intel RealSense D435

Install the Intel RealSense SDK / librealsense and a compatible `pyrealsense2`
binding for the operating system. The adapter aligns depth to the color stream,
reads factory color intrinsics, and converts raw depth using the active sensor's
reported depth scale. Do not assume the raw unit or a fixed depth scale.

```python
from camera_adapters import RealSenseD435Adapter

with RealSenseD435Adapter(width=640, height=480, fps=30) as camera:
    sample = camera.read()
    if sample is not None:
        print(sample.frame.image.shape)
        print(sample.depth.in_metres())
        print(sample.calibration.K)
```

## Stereo pair

`StereoCameraAdapter` reads left/right UVC streams and records host timestamp
skew. Sequential USB reads are not hardware synchronization. A pair outside the
configured skew tolerance is returned with `synchronization_valid=False`.
Metric depth additionally requires calibrated intrinsics, baseline, distortion
handling, and stereo rectification. This adapter deliberately does not invent
those values or claim to produce disparity/depth.

## Timestamp policy

USB timestamps are host monotonic timestamps captured after each read call.
RealSense timestamps use the color sensor timestamp when available. These clocks
must not be mixed without an explicit clock-domain conversion. For production
multi-camera fusion, prefer hardware trigger synchronization or a driver that
exposes exposure timestamps.

## Tests without cameras

From `advanced_3d_scene_reconstruction/`:

```bash
python -m unittest -v tests.test_camera_adapters
```
