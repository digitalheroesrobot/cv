#!/usr/bin/env python3
"""Timestamp-based multi-camera frame synchronization.

This is software synchronization, not hardware trigger synchronization.
All timestamps supplied to one synchronizer MUST share a clock domain or have
been converted into one. The synchronizer uses a bounded queue per camera and
greedily selects the closest available frame to the earliest queue head.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional
from common import CameraFrame


@dataclass
class SynchronizedFrames:
    frames: Dict[str, CameraFrame]
    reference_timestamp: float
    max_skew_s: float
    synchronized: bool


class FrameSynchronizer:
    def __init__(self, camera_ids: Iterable[str], tolerance_s: float = 0.015,
                 queue_size: int = 30):
        ids = tuple(camera_ids)
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("at least two unique camera IDs are required")
        if tolerance_s < 0 or queue_size < 1:
            raise ValueError("tolerance_s must be >= 0 and queue_size >= 1")
        self.camera_ids = ids
        self.tolerance_s = float(tolerance_s)
        self.queues: Dict[str, Deque[CameraFrame]] = {
            camera_id: deque(maxlen=queue_size) for camera_id in ids
        }
        self.dropped_out_of_order = {camera_id: 0 for camera_id in ids}

    def push(self, frame: CameraFrame) -> None:
        if frame.camera_id not in self.queues:
            raise KeyError(f"Unknown camera ID: {frame.camera_id}")
        q = self.queues[frame.camera_id]
        if q and frame.timestamp < q[-1].timestamp:
            self.dropped_out_of_order[frame.camera_id] += 1
            return
        q.append(frame)

    def pop(self) -> Optional[SynchronizedFrames]:
        """Return a synchronized set, or discard stale heads and return None.

        A result is emitted only when each camera has a frame within tolerance
        of the earliest queued timestamp. This avoids pairing distant frames
        merely because every queue is non-empty.
        """
        if any(not q for q in self.queues.values()):
            return None
        reference = min(q[0].timestamp for q in self.queues.values())
        selected = {}
        for camera_id, q in self.queues.items():
            # Discard frames too old to match the earliest reference.
            while q and q[0].timestamp < reference - self.tolerance_s:
                q.popleft()
            if not q:
                return None
            candidate = min(q, key=lambda f: abs(f.timestamp - reference))
            if abs(candidate.timestamp - reference) > self.tolerance_s:
                # Advance the globally earliest stream on the next call.
                earliest_id = min(self.queues, key=lambda k: self.queues[k][0].timestamp)
                self.queues[earliest_id].popleft()
                return None
            selected[camera_id] = candidate
        stamps = [f.timestamp for f in selected.values()]
        skew = max(stamps) - min(stamps)
        if skew > self.tolerance_s:
            earliest_id = min(selected, key=lambda k: selected[k].timestamp)
            self.queues[earliest_id].popleft()
            return None
        # Consume through selected frames, including any older frames skipped.
        for camera_id, chosen in selected.items():
            q = self.queues[camera_id]
            while q:
                current = q.popleft()
                if current is chosen:
                    break
        return SynchronizedFrames(
            frames=selected,
            reference_timestamp=sum(stamps) / len(stamps),
            max_skew_s=skew,
            synchronized=True,
        )
