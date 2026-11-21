#!/usr/bin/env python3
"""Clock-domain conversion and multi-camera timestamp alignment.

Clock model:
    t_master = scale * t_device + offset

The model is fitted from paired observations of the same synchronization event.
It does not create hardware synchronization. Pair observations must be generated
by a shared trigger, wired pulse, or another independently identifiable event.
Using unrelated frame arrival times as pairs folds transport latency into the
clock estimate and is not a valid calibration method.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from common import CameraFrame, CameraPose, make_se3, validate_se3


@dataclass(frozen=True)
class ClockModel:
    """Affine device-to-master clock conversion, time units are seconds."""
    scale: float = 1.0
    offset_s: float = 0.0
    reference_device_s: float = 0.0
    residual_rmse_s: float = 0.0
    samples: int = 0

    def __post_init__(self):
        values = (self.scale, self.offset_s, self.reference_device_s,
                  self.residual_rmse_s)
        if not np.all(np.isfinite(values)) or self.scale <= 0:
            raise ValueError("Clock model parameters must be finite; scale must be positive")
        if self.residual_rmse_s < 0 or self.samples < 0:
            raise ValueError("Clock diagnostics must be non-negative")

    def to_master(self, device_timestamp_s: float) -> float:
        if not np.isfinite(device_timestamp_s):
            raise ValueError("Timestamp must be finite")
        # Centering at reference_device_s improves numerical conditioning.
        return float(self.scale * (device_timestamp_s - self.reference_device_s)
                     + self.reference_device_s + self.offset_s)

    @property
    def drift_ppm(self) -> float:
        return float((self.scale - 1.0) * 1e6)


@dataclass
class ClockFitResult:
    model: ClockModel
    inlier_mask: np.ndarray
    residuals_s: np.ndarray


def fit_clock_model(device_times_s: Sequence[float],
                    master_times_s: Sequence[float],
                    outlier_threshold_s: float = 0.003,
                    max_iterations: int = 10) -> ClockFitResult:
    """Robustly fit an affine clock map using paired synchronization events.

    Iteratively rejects residual outliers using a MAD scale estimate. At least
    three pairs are required to estimate both offset and drift robustly.
    """
    x = np.asarray(device_times_s, dtype=np.float64).reshape(-1)
    y = np.asarray(master_times_s, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or len(x) < 3:
        raise ValueError("At least three paired timestamps are required")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Timestamp pairs must be finite")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Device timestamps must be strictly increasing")
    if outlier_threshold_s <= 0 or max_iterations < 1:
        raise ValueError("Invalid robust fitting configuration")
    x0 = float(np.mean(x))
    inliers = np.ones(len(x), dtype=bool)
    slope, intercept = 1.0, float(np.mean(y - x))
    for _ in range(max_iterations):
        if np.count_nonzero(inliers) < 3:
            raise ValueError("Too few clock synchronization inliers")
        A = np.column_stack([x[inliers] - x0, np.ones(np.count_nonzero(inliers))])
        slope, centered_offset = np.linalg.lstsq(A, y[inliers] - x0, rcond=None)[0]
        intercept = float(centered_offset)
        residuals = y - (slope * (x - x0) + x0 + intercept)
        median = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - median)))
        robust_sigma = 1.4826 * mad
        threshold = max(float(outlier_threshold_s), 3.5 * robust_sigma)
        new_inliers = np.abs(residuals - median) <= threshold
        if np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    if np.count_nonzero(inliers) < 3:
        raise ValueError("Clock fit rejected too many synchronization pairs")
    A = np.column_stack([x[inliers] - x0, np.ones(np.count_nonzero(inliers))])
    slope, centered_offset = np.linalg.lstsq(A, y[inliers] - x0, rcond=None)[0]
    residuals = y - (slope * (x - x0) + x0 + centered_offset)
    rmse = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    model = ClockModel(float(slope), float(centered_offset), x0, rmse,
                       int(np.count_nonzero(inliers)))
    return ClockFitResult(model, inliers, residuals)


class MultiCameraClockSynchronizer:
    """Convert camera clocks to one master clock and pair nearest frames.

    Frames are only paired after clock conversion. Each camera must have a
    fitted ClockModel or an identity model when it is already in the master
    clock domain. Queue overflow, duplicate IDs, and out-of-order frames are
    counted explicitly rather than silently hidden.
    """
    def __init__(self, camera_ids: Iterable[str],
                 clock_models: Mapping[str, ClockModel],
                 tolerance_s: float = 0.008, queue_size: int = 30):
        ids = tuple(camera_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("At least two unique camera IDs are required")
        if set(clock_models) != set(ids):
            raise ValueError("Provide exactly one clock model per camera")
        if tolerance_s < 0 or queue_size < 1:
            raise ValueError("Invalid synchronizer limits")
        self.camera_ids = ids
        self.clock_models = dict(clock_models)
        self.tolerance_s = float(tolerance_s)
        self.queues = {c: deque() for c in ids}
        self.queue_size = int(queue_size)
        self.last_timestamp = {c: -np.inf for c in ids}
        self.last_frame_id = {c: -1 for c in ids}
        self.stats = {c: {"accepted": 0, "out_of_order": 0,
                          "duplicate": 0, "overflow": 0,
                          "stale": 0} for c in ids}
        self.emitted = 0
        self.rejected_sets = 0

    def push(self, frame: CameraFrame) -> bool:
        c = frame.camera_id
        if c not in self.queues:
            raise KeyError(f"Unknown camera ID: {c}")
        if frame.frame_id <= self.last_frame_id[c]:
            self.stats[c]["duplicate"] += 1
            return False
        master_t = self.clock_models[c].to_master(frame.timestamp)
        if master_t <= self.last_timestamp[c]:
            self.stats[c]["out_of_order"] += 1
            return False
        mapped = CameraFrame(
            frame.image, master_t, c, frame.frame_id, frame.color_order,
            metadata={**frame.metadata, "device_timestamp_s": frame.timestamp,
                      "master_timestamp_s": master_t,
                      "clock_scale": self.clock_models[c].scale,
                      "clock_offset_s": self.clock_models[c].offset_s})
        q = self.queues[c]
        if len(q) >= self.queue_size:
            q.popleft()
            self.stats[c]["overflow"] += 1
        q.append(mapped)
        self.last_timestamp[c] = master_t
        self.last_frame_id[c] = frame.frame_id
        self.stats[c]["accepted"] += 1
        return True

    def pop(self):
        """Return the closest complete frame set; discard stale unmatched frames."""
        if any(not q for q in self.queues.values()):
            return None
        while all(self.queues.values()):
            heads = {c: q[0].timestamp for c, q in self.queues.items()}
            reference = max(heads.values())
            # Drop queue heads that cannot match the newest head within tolerance.
            dropped = False
            for c, q in self.queues.items():
                while q and reference - q[0].timestamp > self.tolerance_s:
                    q.popleft()
                    self.stats[c]["stale"] += 1
                    dropped = True
            if any(not q for q in self.queues.values()):
                return None
            # Choose each camera's closest queued frame to the latest head.
            chosen = {}
            for c, q in self.queues.items():
                candidate = min(q, key=lambda f: abs(f.timestamp - reference))
                chosen[c] = candidate
            stamps = [f.timestamp for f in chosen.values()]
            skew = max(stamps) - min(stamps)
            if skew <= self.tolerance_s:
                for c, frame in chosen.items():
                    q = self.queues[c]
                    while q:
                        item = q.popleft()
                        if item is frame:
                            break
                self.emitted += 1
                from .synchronization import SynchronizedFrames
                return SynchronizedFrames(chosen, float(np.mean(stamps)),
                                          float(skew), True)
            # Advance the camera holding the earliest candidate.
            earliest = min(chosen, key=lambda c: chosen[c].timestamp)
            self.queues[earliest].popleft()
            self.stats[earliest]["stale"] += 1
            self.rejected_sets += 1
        return None


def interpolate_camera_pose(pose_a: CameraPose, pose_b: CameraPose,
                             timestamp_s: float,
                             max_gap_s: float = 0.1) -> CameraPose:
    """Interpolate translation linearly and rotation on SO(3) using SLERP.

    This is suitable for small gaps and smooth motion. It is not a substitute
    for IMU integration or a continuous-time trajectory estimator.
    """
    ta, tb = float(pose_a.timestamp), float(pose_b.timestamp)
    if tb <= ta or tb - ta > max_gap_s:
        raise ValueError("Pose timestamps must be ordered and within max_gap_s")
    if timestamp_s < ta or timestamp_s > tb:
        raise ValueError("Interpolation timestamp must lie inside the pose interval")
    if pose_a.frame_id != pose_b.frame_id or pose_a.camera_id != pose_b.camera_id:
        raise ValueError("Pose frame_id and camera_id must match")
    alpha = (timestamp_s - ta) / (tb - ta)
    rotations = Rotation.from_matrix(np.stack([
        pose_a.T_world_camera[:3, :3], pose_b.T_world_camera[:3, :3]]))
    rotation = Slerp([ta, tb], rotations)([timestamp_s]).as_matrix()[0]
    translation = ((1.0 - alpha) * pose_a.T_world_camera[:3, 3]
                   + alpha * pose_b.T_world_camera[:3, 3])
    return CameraPose(make_se3(rotation, translation), float(timestamp_s),
                      pose_a.frame_id, pose_a.camera_id)
