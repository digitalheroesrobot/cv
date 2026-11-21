#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local ORB + RGB-D PnP visual odometry front-end.

Matches image features between consecutive frames, uses Essential Matrix
RANSAC as a monocular geometric consistency diagnostic, then estimates metric
relative motion from 3D points in the previous RGB-D frame to 2D features in
the current frame with solvePnPRansac.

OpenCV solvePnP convention:
    X_current = R_current_previous * X_previous + t_current_previous
so its output is T_current_previous. The camera pose update is:
    T_world_current = T_world_previous @ inverse(T_current_previous)

Input images and depth must be aligned, rectified pinhole images. Depth is
optical-axis Z in metres. No remote calls or automatic downloads are made.
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np
from common.geometry import make_se3, invert_se3, compose_se3


@dataclass
class VisualOdometryResult:
    success: bool
    T_current_previous: Optional[np.ndarray]
    T_world_current: np.ndarray
    feature_matches: int
    essential_inliers: int
    pnp_inliers: int
    reprojection_rmse_px: float
    message: str


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required. Install opencv-python locally.") from exc


def backproject_keypoints(keypoints_uv, depth_m, K, min_depth_m=0.15,
                          max_depth_m=8.0, patch_radius=2):
    """Sample robust local median Z and back-project keypoint pixels to XYZ."""
    uv = np.asarray(keypoints_uv, dtype=np.float64)
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or depth.ndim != 2 or K.shape != (3, 3):
        raise ValueError("expected uv=(N,2), depth=(H,W), K=(3,3)")
    h, w = depth.shape
    points, valid_indices = [], []
    invK = np.linalg.inv(K)
    for i, (u, v) in enumerate(uv):
        x, y = int(round(u)), int(round(v))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x0, x1 = max(0, x-patch_radius), min(w, x+patch_radius+1)
        y0, y1 = max(0, y-patch_radius), min(h, y+patch_radius+1)
        patch = depth[y0:y1, x0:x1]
        good = patch[np.isfinite(patch) & (patch >= min_depth_m) & (patch <= max_depth_m)]
        if good.size == 0:
            continue
        z = float(np.median(good))
        ray = invK @ np.array([u, v, 1.0])
        xyz = ray * (z / ray[2])
        points.append(xyz)
        valid_indices.append(i)
    if not points:
        return np.empty((0, 3)), np.empty(0, dtype=np.int64)
    return np.asarray(points, dtype=np.float64), np.asarray(valid_indices, dtype=np.int64)


def estimate_rgbd_pnp(previous_uv, current_uv, previous_depth_m, K,
                      min_depth_m=0.15, max_depth_m=8.0,
                      reprojection_threshold_px=2.5, iterations=150,
                      confidence=0.999, min_inliers=12, seed=7,
                      patch_radius=2):
    """Estimate metric T_current_previous using previous-frame depth + PnP.

    Correspondences are image features in previous/current frames. Previous
    keypoints are lifted to 3D using their aligned metric depth; current
    keypoints provide 2D observations. PnP RANSAC rejects mismatches/outliers.
    """
    cv2 = _require_cv2()
    prev_uv = np.asarray(previous_uv, dtype=np.float64)
    curr_uv = np.asarray(current_uv, dtype=np.float64)
    if prev_uv.shape != curr_uv.shape or prev_uv.ndim != 2 or prev_uv.shape[1] != 2:
        raise ValueError("previous_uv and current_uv must both be Nx2")
    object_points, valid = backproject_keypoints(
        prev_uv, previous_depth_m, K, min_depth_m, max_depth_m,
        patch_radius=patch_radius)
    if len(valid) < min_inliers:
        raise RuntimeError(f"only {len(valid)} valid depth correspondences")
    image_points = curr_uv[valid].astype(np.float32)
    object_points = object_points.astype(np.float32)
    cv2.setRNGSeed(int(seed))
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        object_points, image_points, np.asarray(K, dtype=np.float64), None,
        iterationsCount=int(iterations),
        reprojectionError=float(reprojection_threshold_px),
        confidence=float(confidence),
        flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers) < min_inliers:
        raise RuntimeError("RGB-D PnP RANSAC did not find enough inliers")
    # Refine the robust consensus with iterative PnP.
    selected = inliers.reshape(-1)
    ok_refine, rvec, tvec = cv2.solvePnP(
        object_points[selected], image_points[selected],
        np.asarray(K, dtype=np.float64), None, rvec, tvec,
        useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok_refine:
        pass
    R, _ = cv2.Rodrigues(rvec)
    T_current_previous = make_se3(R, tvec.reshape(3))
    projected, _ = cv2.projectPoints(
        object_points[selected], rvec, tvec,
        np.asarray(K, dtype=np.float64), None)
    residual = projected.reshape(-1, 2) - image_points[selected]
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return T_current_previous, len(selected), rmse, valid[selected]


class ORBRGBDVisualOdometry:
    """Stateful local feature VO. The first accepted frame defines world origin."""
    def __init__(self, K, min_depth_m=0.15, max_depth_m=8.0,
                 max_features=2500, ratio_test=0.78, min_matches=20,
                 min_pnp_inliers=12, reprojection_threshold_px=2.5,
                 essential_threshold_px=1.5):
        cv2 = _require_cv2()
        self.K = np.asarray(K, dtype=np.float64)
        if self.K.shape != (3, 3):
            raise ValueError("K must be 3x3")
        self.min_depth_m, self.max_depth_m = min_depth_m, max_depth_m
        self.ratio_test = ratio_test
        self.min_matches, self.min_pnp_inliers = min_matches, min_pnp_inliers
        self.reprojection_threshold_px = reprojection_threshold_px
        self.essential_threshold_px = essential_threshold_px
        self.orb = cv2.ORB_create(nfeatures=int(max_features), scaleFactor=1.2,
                                  nlevels=8, edgeThreshold=19, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.previous_gray = None
        self.previous_depth = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.T_world_camera = np.eye(4)
        self.frame_index = 0

    def reset(self):
        self.previous_gray = None
        self.previous_depth = None
        self.previous_keypoints = None
        self.previous_descriptors = None
        self.T_world_camera = np.eye(4)
        self.frame_index = 0

    def _failure(self, message, matches=0, essential=0):
        return VisualOdometryResult(False, None, self.T_world_camera.copy(),
                                    matches, essential, 0, float("inf"), message)

    def process(self, image, depth_m):
        cv2 = _require_cv2()
        image = np.asarray(image)
        depth = np.asarray(depth_m, dtype=np.float32)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError("image must be grayscale or HxWx3 BGR")
        if depth.shape != gray.shape:
            raise ValueError("depth must be aligned to image and have same HxW")
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < self.min_matches:
            return self._failure("not enough ORB features")
        if self.previous_gray is None:
            self.previous_gray, self.previous_depth = gray.copy(), depth.copy()
            self.previous_keypoints, self.previous_descriptors = keypoints, descriptors
            self.frame_index = 1
            return VisualOdometryResult(True, np.eye(4), self.T_world_camera.copy(),
                                        0, 0, 0, 0.0, "initialized")
        pairs = self.matcher.knnMatch(self.previous_descriptors, descriptors, k=2)
        good = [m for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio_test * n.distance]
        if len(good) < self.min_matches:
            return self._failure("not enough descriptor matches", len(good))
        prev_uv = np.asarray([self.previous_keypoints[m.queryIdx].pt for m in good])
        curr_uv = np.asarray([keypoints[m.trainIdx].pt for m in good])
        # Essential matrix gives rotation and translation direction up to scale.
        # It is a diagnostic/consistency gate; RGB-D PnP supplies metric scale.
        E, essential_mask = cv2.findEssentialMat(
            prev_uv, curr_uv, self.K, method=cv2.RANSAC,
            prob=0.999, threshold=self.essential_threshold_px)
        essential_inliers = 0
        if E is not None and essential_mask is not None:
            essential_inliers = int(essential_mask.ravel().sum())
        try:
            T_current_previous, pnp_inliers, rmse, _ = estimate_rgbd_pnp(
                prev_uv, curr_uv, self.previous_depth, self.K,
                self.min_depth_m, self.max_depth_m,
                self.reprojection_threshold_px, min_inliers=self.min_pnp_inliers)
        except (RuntimeError, ValueError) as exc:
            return self._failure(f"RGB-D PnP failed: {exc}", len(good), essential_inliers)
        # Camera pose is the inverse of the world-to-current motion.
        T_previous_current = invert_se3(T_current_previous)
        T_world_current = compose_se3(self.T_world_camera, T_previous_current)
        self.T_world_camera = T_world_current
        self.previous_gray, self.previous_depth = gray.copy(), depth.copy()
        self.previous_keypoints, self.previous_descriptors = keypoints, descriptors
        self.frame_index += 1
        return VisualOdometryResult(
            True, T_current_previous, self.T_world_camera.copy(), len(good),
            essential_inliers, pnp_inliers, rmse, "ok")
