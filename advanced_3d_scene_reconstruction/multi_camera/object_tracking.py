#!/usr/bin/env python3
"""Multi-camera 3D detection fusion and constant-velocity Kalman tracking.

Input detections must already be expressed in the same metric world frame.
This module performs cross-camera association, not image-space detection or
camera-pose estimation. The Hungarian assignment is gated by class, Mahalanobis
distance, object size, and optional appearance embeddings.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import ObjectTrack


@dataclass
class Detection3D:
    """One camera's 3D detection, expressed in the shared world frame."""
    center: np.ndarray
    dimensions: np.ndarray
    label: str
    timestamp: float
    camera_id: str
    confidence: float = 1.0
    quaternion_xyzw: np.ndarray = field(
        default_factory=lambda: np.array([0., 0., 0., 1.]))
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3).copy()
        self.dimensions = np.asarray(self.dimensions, dtype=np.float64).reshape(3).copy()
        self.quaternion_xyzw = np.asarray(
            self.quaternion_xyzw, dtype=np.float64).reshape(4).copy()
        if not np.all(np.isfinite(self.center)):
            raise ValueError("Detection center must be finite")
        if not np.all(np.isfinite(self.dimensions)) or np.any(self.dimensions <= 0):
            raise ValueError("Detection dimensions must be finite and positive")
        if not self.label or not self.camera_id:
            raise ValueError("Detection label and camera_id cannot be empty")
        if not np.isfinite(self.timestamp) or self.timestamp < 0:
            raise ValueError("Detection timestamp must be finite and non-negative")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("Detection confidence must be in [0,1]")
        qnorm = np.linalg.norm(self.quaternion_xyzw)
        if qnorm < 1e-12:
            raise ValueError("Detection quaternion cannot be zero")
        self.quaternion_xyzw /= qnorm
        if self.embedding is not None:
            self.embedding = np.asarray(self.embedding, dtype=np.float64).reshape(-1).copy()
            if not np.all(np.isfinite(self.embedding)):
                raise ValueError("Appearance embedding must be finite")
            norm = np.linalg.norm(self.embedding)
            if norm < 1e-12:
                raise ValueError("Appearance embedding cannot be zero")
            self.embedding /= norm


@dataclass
class FusedDetection3D:
    center: np.ndarray
    dimensions: np.ndarray
    label: str
    timestamp: float
    confidence: float
    quaternion_xyzw: np.ndarray
    camera_ids: Tuple[str, ...]
    embedding: Optional[np.ndarray] = None
    source_count: int = 1


@dataclass
class TrackState:
    track_id: int
    label: str
    state: np.ndarray                 # [x,y,z,vx,vy,vz]
    covariance: np.ndarray            # 6x6 state covariance
    dimensions: np.ndarray
    quaternion_xyzw: np.ndarray
    confidence: float
    timestamp: float
    first_timestamp: float
    age: int = 1
    hits: int = 1
    missed: int = 0
    camera_ids: Tuple[str, ...] = ()
    embedding: Optional[np.ndarray] = None


class MultiCameraDetectionFusion:
    """Greedy cross-camera duplicate suppression with one vote per camera.

    Detections from the same camera are never merged. Candidate edges are
    processed from smallest normalized 3D distance to largest. This conservative
    baseline is not a substitute for learned multi-view identity matching.
    """
    def __init__(self, max_center_distance_m=0.18,
                 max_size_ratio=2.5, min_embedding_similarity=0.35):
        if max_center_distance_m <= 0 or max_size_ratio < 1:
            raise ValueError("Invalid cross-camera fusion thresholds")
        self.max_center_distance_m = float(max_center_distance_m)
        self.max_size_ratio = float(max_size_ratio)
        self.min_embedding_similarity = float(min_embedding_similarity)

    @staticmethod
    def _embedding_similarity(a, b):
        if a is None or b is None:
            return None
        if a.shape != b.shape:
            return -1.0
        return float(np.clip(a @ b, -1.0, 1.0))

    def fuse(self, detections: Sequence[Detection3D]) -> List[FusedDetection3D]:
        detections = list(detections)
        if not detections:
            return []
        timestamps = np.array([d.timestamp for d in detections])
        if timestamps.max() - timestamps.min() > 0.05:
            raise ValueError("Cross-camera detections exceed 50 ms temporal skew")
        edges = []
        for i in range(len(detections)):
            a = detections[i]
            for j in range(i + 1, len(detections)):
                b = detections[j]
                if a.camera_id == b.camera_id or a.label != b.label:
                    continue
                distance = float(np.linalg.norm(a.center - b.center))
                size_ratio = float(np.max(np.maximum(
                    a.dimensions / b.dimensions, b.dimensions / a.dimensions)))
                if distance > self.max_center_distance_m or size_ratio > self.max_size_ratio:
                    continue
                similarity = self._embedding_similarity(a.embedding, b.embedding)
                if similarity is not None and similarity < self.min_embedding_similarity:
                    continue
                edges.append((distance / self.max_center_distance_m, i, j))
        edges.sort()
        groups = [{i} for i in range(len(detections))]
        owner = list(range(len(detections)))
        for _, i, j in edges:
            gi, gj = owner[i], owner[j]
            if gi == gj:
                continue
            members_i, members_j = groups[gi], groups[gj]
            cams_i = {detections[k].camera_id for k in members_i}
            cams_j = {detections[k].camera_id for k in members_j}
            if cams_i & cams_j:
                continue
            # Prevent a chain of weak pairwise matches from creating a group
            # whose members are mutually inconsistent.
            combined = members_i | members_j
            centers = np.stack([detections[k].center for k in combined])
            if np.max(np.linalg.norm(centers[:, None] - centers[None, :], axis=2)) > self.max_center_distance_m:
                continue
            new_owner = min(gi, gj)
            old_owner = max(gi, gj)
            groups[new_owner] = combined
            groups[old_owner] = set()
            for k in combined:
                owner[k] = new_owner
        fused = []
        for members in groups:
            if not members:
                continue
            ds = [detections[k] for k in sorted(members)]
            weights = np.array([max(d.confidence, 1e-3) for d in ds])
            weights /= weights.sum()
            center = np.sum(np.stack([d.center for d in ds]) * weights[:, None], axis=0)
            dimensions = np.sum(np.stack([d.dimensions for d in ds]) * weights[:, None], axis=0)
            # Quaternion sign ambiguity: align all quaternions to the first.
            q0 = ds[0].quaternion_xyzw
            quats = [q if q @ q0 >= 0 else -q for q in
                     [d.quaternion_xyzw for d in ds]]
            q = np.sum(np.stack(quats) * weights[:, None], axis=0)
            q /= max(np.linalg.norm(q), 1e-12)
            embeddings = [d.embedding for d in ds if d.embedding is not None]
            embedding = None
            if embeddings and all(e.shape == embeddings[0].shape for e in embeddings):
                embedding = np.mean(embeddings, axis=0)
                embedding /= max(np.linalg.norm(embedding), 1e-12)
            confidence = 1.0 - float(np.prod([1.0-d.confidence for d in ds]))
            fused.append(FusedDetection3D(
                center, dimensions, ds[0].label,
                float(np.average([d.timestamp for d in ds], weights=weights)),
                confidence, q, tuple(sorted({d.camera_id for d in ds})),
                embedding, len(ds)))
        return fused


class ConstantVelocityKalman:
    """Linear 3D constant-velocity filter with white-acceleration process noise."""
    def __init__(self, acceleration_noise_mps2=2.0,
                 measurement_noise_m=0.04):
        if acceleration_noise_mps2 <= 0 or measurement_noise_m <= 0:
            raise ValueError("Kalman noise parameters must be positive")
        self.acceleration_noise = float(acceleration_noise_mps2)
        self.measurement_noise = float(measurement_noise_m)

    def predict(self, state, covariance, dt):
        if dt < 0:
            raise ValueError("Prediction dt cannot be negative")
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.acceleration_noise ** 2
        Q = np.block([
            [np.eye(3) * (dt**4/4*q), np.eye(3) * (dt**3/2*q)],
            [np.eye(3) * (dt**3/2*q), np.eye(3) * (dt**2*q)]
        ])
        return F @ state, F @ covariance @ F.T + Q

    def update(self, state, covariance, measurement, confidence=1.0):
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        # Lower-confidence detections carry larger measurement uncertainty.
        variance = self.measurement_noise**2 / max(confidence, 0.05)
        R = np.eye(3) * variance
        innovation = measurement - H @ state
        S = H @ covariance @ H.T + R
        K = np.linalg.solve(S, H @ covariance).T
        updated = state + K @ innovation
        I_KH = np.eye(6) - K @ H
        # Joseph form keeps covariance symmetric/PSD under finite precision.
        P = I_KH @ covariance @ I_KH.T + K @ R @ K.T
        return updated, 0.5 * (P + P.T), innovation, S


class MultiCamera3DTracker:
    """Fuse multi-view detections, associate them to persistent 3D tracks."""
    def __init__(self, fusion: Optional[MultiCameraDetectionFusion] = None,
                 max_mahalanobis=16.27, max_missed_frames=8,
                 max_track_age_s=1.0, min_hits=2,
                 acceleration_noise_mps2=2.0, measurement_noise_m=0.04,
                 max_embedding_distance=0.8):
        if max_mahalanobis <= 0 or max_missed_frames < 0 or max_track_age_s <= 0 or min_hits < 1:
            raise ValueError("Invalid tracker limits")
        self.fusion = fusion or MultiCameraDetectionFusion()
        self.kalman = ConstantVelocityKalman(
            acceleration_noise_mps2, measurement_noise_m)
        self.max_mahalanobis = float(max_mahalanobis)
        self.max_missed_frames = int(max_missed_frames)
        self.max_track_age_s = float(max_track_age_s)
        self.min_hits = int(min_hits)
        self.max_embedding_distance = float(max_embedding_distance)
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        self.last_timestamp = None

    def _cost(self, track: TrackState, detection: FusedDetection3D):
        if track.label != detection.label:
            return np.inf
        ratio = np.maximum(track.dimensions / detection.dimensions,
                           detection.dimensions / track.dimensions)
        if np.max(ratio) > 3.0:
            return np.inf
        if track.embedding is not None and detection.embedding is not None:
            if track.embedding.shape != detection.embedding.shape:
                return np.inf
            similarity = float(track.embedding @ detection.embedding)
            if 1.0 - similarity > self.max_embedding_distance:
                return np.inf
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        innovation = detection.center - H @ track.state
        S = H @ track.covariance @ H.T + np.eye(3) * self.kalman.measurement_noise**2
        try:
            mahalanobis = float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return np.inf
        if mahalanobis > self.max_mahalanobis:
            return np.inf
        size_penalty = float(np.mean(np.abs(np.log(ratio))))
        return mahalanobis + 2.0 * size_penalty

    def _new_track(self, detection):
        state = np.zeros(6)
        state[:3] = detection.center
        covariance = np.diag([0.05**2]*3 + [1.0**2]*3)
        track = TrackState(
            self.next_track_id, detection.label, state, covariance,
            detection.dimensions.copy(), detection.quaternion_xyzw.copy(),
            detection.confidence, detection.timestamp, detection.timestamp,
            camera_ids=detection.camera_ids,
            embedding=None if detection.embedding is None else detection.embedding.copy())
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def update(self, detections: Sequence[Detection3D], timestamp: float) -> List[ObjectTrack]:
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Tracker timestamp must be finite and non-negative")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("Tracker timestamps must be strictly increasing")
        fused = self.fusion.fuse(detections)
        if any(abs(d.timestamp - timestamp) > 0.05 for d in fused):
            raise ValueError("Detection timestamps do not match tracker update time")
        dt_global = 0.0 if self.last_timestamp is None else timestamp - self.last_timestamp
        for track in self.tracks.values():
            dt = timestamp - track.timestamp
            track.state, track.covariance = self.kalman.predict(
                track.state, track.covariance, dt)
            track.age += 1
            track.missed += 1
            track.timestamp = timestamp

        track_ids = list(self.tracks)
        cost = np.full((len(track_ids), len(fused)), 1e9, dtype=np.float64)
        for i, track_id in enumerate(track_ids):
            for j, detection in enumerate(fused):
                value = self._cost(self.tracks[track_id], detection)
                if np.isfinite(value):
                    cost[i, j] = value
        matched_tracks, matched_detections = set(), set()
        if cost.size:
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= 1e8:
                    continue
                track = self.tracks[track_ids[i]]
                detection = fused[j]
                track.state, track.covariance, _, _ = self.kalman.update(
                    track.state, track.covariance, detection.center,
                    detection.confidence)
                alpha = float(np.clip(detection.confidence, 0.15, 0.8))
                track.dimensions = (1-alpha)*track.dimensions + alpha*detection.dimensions
                q = detection.quaternion_xyzw
                if q @ track.quaternion_xyzw < 0:
                    q = -q
                track.quaternion_xyzw = (1-alpha)*track.quaternion_xyzw + alpha*q
                track.quaternion_xyzw /= max(np.linalg.norm(track.quaternion_xyzw), 1e-12)
                track.confidence = detection.confidence
                track.hits += 1
                track.missed = 0
                track.camera_ids = detection.camera_ids
                if detection.embedding is not None:
                    if track.embedding is None or track.embedding.shape != detection.embedding.shape:
                        track.embedding = detection.embedding.copy()
                    else:
                        emb = (1-alpha)*track.embedding + alpha*detection.embedding
                        track.embedding = emb / max(np.linalg.norm(emb), 1e-12)
                matched_tracks.add(track.track_id)
                matched_detections.add(j)
        for j, detection in enumerate(fused):
            if j not in matched_detections:
                self._new_track(detection)
        expired = [
            track_id for track_id, track in self.tracks.items()
            if track.missed > self.max_missed_frames
            or timestamp - track.first_timestamp > self.max_track_age_s + dt_global * self.max_missed_frames
        ]
        for track_id in expired:
            del self.tracks[track_id]
        self.last_timestamp = timestamp
        return self.snapshot(confirmed_only=True)

    def snapshot(self, confirmed_only=True) -> List[ObjectTrack]:
        output = []
        for track in self.tracks.values():
            if confirmed_only and track.hits < self.min_hits:
                continue
            output.append(ObjectTrack(
                track.track_id, track.label, track.state[:3],
                track.timestamp, track.state[3:],
                track.dimensions, track.quaternion_xyzw,
                "world", track.confidence, track.covariance[:3, :3]))
        return sorted(output, key=lambda t: t.track_id)

    def diagnostics(self):
        return {
            "active_tracks": len(self.tracks),
            "confirmed_tracks": sum(t.hits >= self.min_hits for t in self.tracks.values()),
            "next_track_id": self.next_track_id,
            "last_timestamp": self.last_timestamp,
        }
