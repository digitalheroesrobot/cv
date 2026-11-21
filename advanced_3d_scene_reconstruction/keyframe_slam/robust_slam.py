#!/usr/bin/env python3
"""Robust long-running SLAM utilities: tracking state, submaps, relocalization,
loop-closure policy, bounded memory, and numerical health monitoring.

This module is deliberately backend-agnostic. It can supervise the Lesson 49
RGB-D pipeline without silently pretending that descriptor similarity is a
geometric pose measurement.
"""
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Tuple
import numpy as np


class TrackingState(str, Enum):
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    DEGRADED = "degraded"
    LOST = "lost"
    RELOCALIZING = "relocalizing"


@dataclass
class TrackingDecision:
    state: TrackingState
    accepted: bool
    start_new_submap: bool
    reason: str


class TrackingMonitor:
    """Hysteretic tracker state machine; avoids one-frame state oscillations."""
    def __init__(self, degrade_score=0.35, lost_score=0.15,
                 lost_after=3, recover_after=3):
        if not 0 <= lost_score < degrade_score <= 1:
            raise ValueError("Require 0 <= lost_score < degrade_score <= 1")
        if lost_after < 1 or recover_after < 1:
            raise ValueError("Frame counts must be positive")
        self.degrade_score = float(degrade_score)
        self.lost_score = float(lost_score)
        self.lost_after = int(lost_after)
        self.recover_after = int(recover_after)
        self.state = TrackingState.INITIALIZING
        self._bad_count = 0
        self._good_count = 0

    def update(self, score: float, pose_valid=True) -> TrackingDecision:
        score = float(score)
        if not np.isfinite(score):
            score = 0.0
        score = float(np.clip(score, 0.0, 1.0))
        if not pose_valid or score < self.lost_score:
            self._bad_count += 1
            self._good_count = 0
        elif score < self.degrade_score:
            self._bad_count = max(self._bad_count, 1)
            self._good_count = 0
        else:
            self._bad_count = 0
            self._good_count += 1

        old = self.state
        if self.state == TrackingState.INITIALIZING:
            if self._good_count >= self.recover_after:
                self.state = TrackingState.TRACKING
        elif self.state in (TrackingState.TRACKING, TrackingState.DEGRADED):
            if self._bad_count >= self.lost_after:
                self.state = TrackingState.LOST
            elif self._bad_count > 0:
                self.state = TrackingState.DEGRADED
            elif self._good_count >= self.recover_after:
                self.state = TrackingState.TRACKING
        elif self.state == TrackingState.LOST:
            if self._good_count >= 1:
                self.state = TrackingState.RELOCALIZING
        elif self.state == TrackingState.RELOCALIZING:
            if self._good_count >= self.recover_after:
                self.state = TrackingState.TRACKING
            elif self._bad_count >= self.lost_after:
                self.state = TrackingState.LOST

        start_submap = self.state == TrackingState.LOST and old != TrackingState.LOST
        accepted = bool(pose_valid and score >= self.lost_score and
                        self.state not in (TrackingState.LOST, TrackingState.INITIALIZING))
        return TrackingDecision(
            self.state, accepted, start_submap,
            f"{old.value}->{self.state.value}; score={score:.3f}; pose_valid={bool(pose_valid)}")

    def reset(self):
        self.state = TrackingState.INITIALIZING
        self._bad_count = self._good_count = 0


@dataclass
class Submap:
    submap_id: int
    keyframe_ids: List[int] = field(default_factory=list)
    anchor_pose: Optional[np.ndarray] = None
    point_count: int = 0
    frozen: bool = False


class SubmapManager:
    """Bounded submap registry; old submaps can be frozen, retained, or evicted."""
    def __init__(self, max_submaps=8, max_keyframes_per_submap=40,
                 max_total_points=1_000_000):
        if min(max_submaps, max_keyframes_per_submap, max_total_points) < 1:
            raise ValueError("All submap limits must be positive")
        self.max_submaps = int(max_submaps)
        self.max_keyframes_per_submap = int(max_keyframes_per_submap)
        self.max_total_points = int(max_total_points)
        self.submaps: Deque[Submap] = deque()
        self._next_id = 0

    @property
    def current(self):
        return self.submaps[-1] if self.submaps else None

    @property
    def total_points(self):
        return sum(s.point_count for s in self.submaps)

    def start(self, anchor_pose=None):
        if self.current is not None:
            self.current.frozen = True
        pose = None if anchor_pose is None else np.asarray(anchor_pose, dtype=float).copy()
        if pose is not None and (pose.shape != (4, 4) or not np.all(np.isfinite(pose))):
            raise ValueError("anchor_pose must be a finite 4x4 matrix")
        item = Submap(self._next_id, anchor_pose=pose)
        self._next_id += 1
        self.submaps.append(item)
        self._enforce_limits()
        return item

    def add_keyframe(self, keyframe_id, point_count):
        if self.current is None:
            self.start()
        if point_count < 0:
            raise ValueError("point_count cannot be negative")
        if len(self.current.keyframe_ids) >= self.max_keyframes_per_submap:
            self.start()
        self.current.keyframe_ids.append(int(keyframe_id))
        self.current.point_count += int(point_count)
        self._enforce_limits()
        return self.current

    def _enforce_limits(self):
        while len(self.submaps) > self.max_submaps or self.total_points > self.max_total_points:
            if len(self.submaps) <= 1:
                # Keep one active submap even if it alone exceeds the budget.
                break
            self.submaps.popleft()

    def merge_metadata(self, source_id, target_id):
        """Merge keyframe membership and point counts; geometry must be transformed externally."""
        by_id = {s.submap_id: s for s in self.submaps}
        if source_id not in by_id or target_id not in by_id:
            raise KeyError("Unknown submap id")
        if source_id == target_id:
            return by_id[target_id]
        source, target = by_id[source_id], by_id[target_id]
        target.keyframe_ids.extend(source.keyframe_ids)
        target.point_count += source.point_count
        target.keyframe_ids = sorted(set(target.keyframe_ids))
        self.submaps = deque(s for s in self.submaps if s.submap_id != source_id)
        self._enforce_limits()
        return target


@dataclass
class RelocalizationCandidate:
    keyframe_id: int
    appearance_score: float
    payload: object = None


class KeyframeRelocalizationIndex:
    """Small descriptor index; ranking only. A caller must perform PnP/ICP checks."""
    def __init__(self, min_temporal_gap=20, max_candidates=5):
        if min_temporal_gap < 1 or max_candidates < 1:
            raise ValueError("Invalid relocalization index limits")
        self.min_temporal_gap = int(min_temporal_gap)
        self.max_candidates = int(max_candidates)
        self._items: Dict[int, Tuple[np.ndarray, object]] = {}

    def add(self, keyframe_id, descriptors, payload=None):
        if descriptors is None:
            return
        desc = np.asarray(descriptors, dtype=np.uint8)
        if desc.ndim != 2 or len(desc) == 0:
            return
        self._items[int(keyframe_id)] = (desc.copy(), payload)

    def query(self, query_keyframe_id, descriptors):
        if descriptors is None or len(descriptors) == 0:
            return []
        import cv2
        qid = int(query_keyframe_id)
        query = np.asarray(descriptors, dtype=np.uint8)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ranked = []
        for kid, (desc, payload) in self._items.items():
            if abs(qid - kid) < self.min_temporal_gap:
                continue
            matches = matcher.match(query, desc)
            if not matches:
                continue
            # Lower Hamming distance is better; normalize for descriptor count.
            score = float(np.mean([m.distance for m in matches]))
            ranked.append(RelocalizationCandidate(kid, score, payload))
        ranked.sort(key=lambda c: c.appearance_score)
        return ranked[:self.max_candidates]


@dataclass
class LoopDecision:
    accepted: bool
    reason: str
    translation_residual_m: float = float("inf")
    rotation_residual_rad: float = float("inf")


class LoopClosurePolicy:
    """Reject duplicate, weak, or physically inconsistent loop constraints."""
    def __init__(self, min_inliers=20, max_reprojection_px=2.5,
                 max_translation_residual_m=0.15,
                 max_rotation_residual_rad=0.15):
        self.min_inliers = int(min_inliers)
        self.max_reprojection_px = float(max_reprojection_px)
        self.max_translation_residual_m = float(max_translation_residual_m)
        self.max_rotation_residual_rad = float(max_rotation_residual_rad)
        self.accepted_pairs = set()

    def evaluate(self, i, j, inliers, reprojection_rmse_px,
                 T_i_j_visual, T_i_j_icp):
        pair = tuple(sorted((int(i), int(j))))
        if i == j:
            return LoopDecision(False, "self-loop")
        if pair in self.accepted_pairs:
            return LoopDecision(False, "duplicate-loop")
        if int(inliers) < self.min_inliers:
            return LoopDecision(False, "insufficient-pnp-inliers")
        if not np.isfinite(reprojection_rmse_px) or reprojection_rmse_px > self.max_reprojection_px:
            return LoopDecision(False, "poor-reprojection")
        A = np.asarray(T_i_j_visual, dtype=float)
        B = np.asarray(T_i_j_icp, dtype=float)
        if A.shape != (4, 4) or B.shape != (4, 4) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
            return LoopDecision(False, "invalid-transform")
        delta = np.linalg.inv(A) @ B
        t_res = float(np.linalg.norm(delta[:3, 3]))
        from scipy.spatial.transform import Rotation
        r_res = float(Rotation.from_matrix(delta[:3, :3]).magnitude())
        if t_res > self.max_translation_residual_m:
            return LoopDecision(False, "translation-disagreement", t_res, r_res)
        if r_res > self.max_rotation_residual_rad:
            return LoopDecision(False, "rotation-disagreement", t_res, r_res)
        self.accepted_pairs.add(pair)
        return LoopDecision(True, "geometrically-consistent", t_res, r_res)


def information_from_covariance(covariance, eigenvalue_floor=1e-9):
    """Convert a 6x6 covariance into a symmetric positive-definite information matrix."""
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (6, 6) or not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be finite 6x6")
    cov = 0.5 * (cov + cov.T)
    values, vectors = np.linalg.eigh(cov)
    values = np.maximum(values, float(eigenvalue_floor))
    return (vectors * (1.0 / values)) @ vectors.T


def whiten_pose_residual(residual, information):
    """Whiten a 6D residual using a Cholesky factor of its information matrix."""
    r = np.asarray(residual, dtype=float).reshape(6)
    info = np.asarray(information, dtype=float)
    if info.shape != (6, 6) or not np.all(np.isfinite(info)):
        raise ValueError("information must be finite 6x6")
    info = 0.5 * (info + info.T)
    try:
        L = np.linalg.cholesky(info)
    except np.linalg.LinAlgError as exc:
        raise ValueError("information matrix must be positive definite") from exc
    return L.T @ r


@dataclass
class MemorySnapshot:
    keyframes: int
    point_count: int
    estimated_bytes: int
    within_budget: bool


class MemoryBudget:
    """Track explicit payload sizes; does not pretend to measure Python RSS."""
    def __init__(self, max_keyframes=500, max_points=1_000_000,
                 max_payload_bytes=512 * 1024 * 1024):
        if min(max_keyframes, max_points, max_payload_bytes) < 1:
            raise ValueError("Memory limits must be positive")
        self.max_keyframes = int(max_keyframes)
        self.max_points = int(max_points)
        self.max_payload_bytes = int(max_payload_bytes)
        self._payloads: Dict[int, int] = {}
        self._points: Dict[int, int] = {}

    def register(self, item_id, payload_bytes, point_count=0):
        if payload_bytes < 0 or point_count < 0:
            raise ValueError("Payload sizes cannot be negative")
        self._payloads[int(item_id)] = int(payload_bytes)
        self._points[int(item_id)] = int(point_count)
        return self.snapshot()

    def evict_oldest(self):
        if not self._payloads:
            return None
        oldest = min(self._payloads)
        self._payloads.pop(oldest)
        self._points.pop(oldest, None)
        return oldest

    def snapshot(self):
        count = len(self._payloads)
        points = sum(self._points.values())
        size = sum(self._payloads.values())
        return MemorySnapshot(count, points, size,
            count <= self.max_keyframes and points <= self.max_points and
            size <= self.max_payload_bytes)

    def enforce(self):
        evicted = []
        while self._payloads and not self.snapshot().within_budget:
            evicted.append(self.evict_oldest())
        return evicted, self.snapshot()


class SLAMHealthMonitor:
    """Rolling operational metrics for long-running camera sessions."""
    def __init__(self, window=100, max_failure_rate=0.25,
                 max_frame_interval_s=0.5):
        if window < 2 or not 0 <= max_failure_rate <= 1 or max_frame_interval_s <= 0:
            raise ValueError("Invalid health monitor configuration")
        self.samples = deque(maxlen=int(window))
        self.max_failure_rate = float(max_failure_rate)
        self.max_frame_interval_s = float(max_frame_interval_s)
        self._last_timestamp = None

    def update(self, timestamp, success, map_points=0):
        timestamp = float(timestamp)
        interval = None if self._last_timestamp is None else timestamp - self._last_timestamp
        self._last_timestamp = timestamp
        self.samples.append((bool(success), int(map_points), interval))
        failures = sum(not x[0] for x in self.samples)
        rate = failures / len(self.samples)
        intervals = [x[2] for x in self.samples if x[2] is not None]
        mean_dt = float(np.mean(intervals)) if intervals else 0.0
        warnings = []
        if rate > self.max_failure_rate:
            warnings.append("high-frame-failure-rate")
        if intervals and max(intervals) > self.max_frame_interval_s:
            warnings.append("frame-gap")
        if map_points < 0:
            warnings.append("invalid-map-size")
        return {"failure_rate": rate, "mean_frame_interval_s": mean_dt,
                "warnings": warnings, "samples": len(self.samples)}
