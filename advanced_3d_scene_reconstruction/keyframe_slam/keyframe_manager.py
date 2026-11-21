#!/usr/bin/env python3
"""Local keyframe storage and appearance-only loop-closure candidate generation."""
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class Keyframe:
    keyframe_id: int
    timestamp: float
    T_world_camera: np.ndarray
    image: Optional[np.ndarray] = None
    depth_m: Optional[np.ndarray] = None
    descriptors: Optional[np.ndarray] = None
    keypoints: Optional[list] = None

    def __post_init__(self):
        self.T_world_camera = np.asarray(self.T_world_camera, dtype=float).copy()
        if self.T_world_camera.shape != (4, 4):
            raise ValueError("T_world_camera must be 4x4")
        if self.depth_m is not None:
            self.depth_m = np.asarray(self.depth_m, dtype=np.float32).copy()


class KeyframeManager:
    """Select keyframes from motion, elapsed frames, and tracking degradation."""
    def __init__(self, translation_threshold_m=0.20,
                 rotation_threshold_rad=0.15, max_frame_gap=15):
        if translation_threshold_m <= 0 or rotation_threshold_rad <= 0 or max_frame_gap < 1:
            raise ValueError("Invalid keyframe thresholds")
        self.translation_threshold_m = float(translation_threshold_m)
        self.rotation_threshold_rad = float(rotation_threshold_rad)
        self.max_frame_gap = int(max_frame_gap)
        self.keyframes: List[Keyframe] = []
        self.last_selected_frame = -1

    @staticmethod
    def _rotation_angle(Ra, Rb):
        from scipy.spatial.transform import Rotation
        return float(Rotation.from_matrix(Ra.T @ Rb).magnitude())

    def should_insert(self, frame_index, T_world_camera, tracking_quality=1.0):
        if not self.keyframes:
            return True
        last = self.keyframes[-1]
        T = np.asarray(T_world_camera, dtype=float)
        translation = np.linalg.norm(T[:3, 3] - last.T_world_camera[:3, 3])
        rotation = self._rotation_angle(last.T_world_camera[:3, :3], T[:3, :3])
        gap = int(frame_index) - self.last_selected_frame
        return (translation >= self.translation_threshold_m or
                rotation >= self.rotation_threshold_rad or
                gap >= self.max_frame_gap or tracking_quality < 0.35)

    def add(self, frame_index, timestamp, T_world_camera, image=None,
            depth_m=None, descriptors=None, keypoints=None):
        keyframe = Keyframe(len(self.keyframes), float(timestamp),
                            T_world_camera, image, depth_m, descriptors, keypoints)
        self.keyframes.append(keyframe)
        self.last_selected_frame = int(frame_index)
        return keyframe


class LoopCandidateDetector:
    """ORB descriptor matcher: proposes candidates, never certifies loop closure.

    Geometric verification (RGB-D PnP/ICP plus consistency checks) must be
    performed before adding a loop edge to the pose graph.
    """
    def __init__(self, min_id_gap=10, ratio_test=0.75, min_good_matches=20):
        self.min_id_gap = int(min_id_gap)
        self.ratio_test = float(ratio_test)
        self.min_good_matches = int(min_good_matches)
        if self.min_id_gap < 1 or not 0 < self.ratio_test < 1 or self.min_good_matches < 1:
            raise ValueError("Invalid loop detector configuration")

    def detect(self, query: Keyframe, candidates):
        if query.descriptors is None or len(query.descriptors) == 0:
            return []
        import cv2
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ranked = []
        for candidate in candidates:
            if abs(query.keyframe_id - candidate.keyframe_id) < self.min_id_gap:
                continue
            if candidate.descriptors is None or len(candidate.descriptors) == 0:
                continue
            pairs = matcher.knnMatch(query.descriptors, candidate.descriptors, k=2)
            good = [m for pair in pairs if len(pair) == 2
                    for m, n in [pair] if m.distance < self.ratio_test * n.distance]
            if len(good) >= self.min_good_matches:
                ranked.append((len(good), candidate))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked
