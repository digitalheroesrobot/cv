#!/usr/bin/env python3
"""Chessboard intrinsic/stereo calibration helpers built on OpenCV.

Extrinsic convention returned by stereoCalibrate:
    X_right = R_right_left @ X_left + T_right_left
Therefore T_right_left maps points from the left camera frame to right.
The inverse T_left_right is provided as well.

Calibration quality must be assessed on held-out views too. A low training
reprojection error does not prove the board was detected over a sufficiently
wide range of image positions, distances, and orientations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple
import numpy as np
from common import CameraCalibration, make_se3, invert_se3


@dataclass
class StereoCalibrationResult:
    left: CameraCalibration
    right: CameraCalibration
    R_right_left: np.ndarray
    T_right_left: np.ndarray
    T_left_right: np.ndarray
    essential: np.ndarray
    fundamental: np.ndarray
    rms: float
    per_view_reprojection_rmse: np.ndarray


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise RuntimeError("Calibration requires opencv-python") from exc


def chessboard_object_points(board_size: Tuple[int, int],
                             square_size_m: float) -> np.ndarray:
    cols, rows = board_size
    if cols < 2 or rows < 2 or square_size_m <= 0:
        raise ValueError("board dimensions must be >=2 and square size positive")
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= float(square_size_m)
    return points


def _detect(image: np.ndarray, board_size: Tuple[int, int]):
    cv2 = _cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def calibrate_intrinsics_from_chessboards(
    images: Sequence[np.ndarray], board_size: Tuple[int, int],
    square_size_m: float
):
    cv2 = _cv2()
    if len(images) < 5:
        raise ValueError("At least five usable calibration images are recommended")
    obj_template = chessboard_object_points(board_size, square_size_m)
    obj_points, img_points = [], []
    image_size = None
    for image in images:
        h, w = image.shape[:2]
        if image_size is None: image_size = (w, h)
        if image_size != (w, h):
            raise ValueError("All calibration images must have the same resolution")
        corners = _detect(image, board_size)
        if corners is not None:
            obj_points.append(obj_template.copy())
            img_points.append(corners)
    if len(obj_points) < 5:
        raise RuntimeError(f"Only {len(obj_points)} valid chessboards detected; need >=5")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    calibration = CameraCalibration(
        K=K, width=image_size[0], height=image_size[1],
        distortion=dist.reshape(-1), calibration_id="opencv_chessboard",
    )
    errors = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((img.reshape(-1, 2) -
                                                    projected.reshape(-1, 2)) ** 2, axis=1)))))
    return calibration, float(rms), np.asarray(errors), len(obj_points)


def calibrate_stereo_from_chessboards(
    left_images: Sequence[np.ndarray], right_images: Sequence[np.ndarray],
    board_size: Tuple[int, int], square_size_m: float,
    left_calibration: CameraCalibration,
    right_calibration: CameraCalibration,
    fix_intrinsics: bool = True,
) -> StereoCalibrationResult:
    cv2 = _cv2()
    if len(left_images) != len(right_images) or len(left_images) < 5:
        raise ValueError("Need at least five paired left/right calibration images")
    if (left_calibration.width, left_calibration.height) != (
        right_calibration.width, right_calibration.height
    ):
        raise ValueError("Stereo image dimensions must match for this helper")
    obj_template = chessboard_object_points(board_size, square_size_m)
    obj_points, left_points, right_points = [], [], []
    for li, ri in zip(left_images, right_images):
        if li.shape[:2] != ri.shape[:2]:
            raise ValueError("Left/right image resolutions differ")
        lc, rc = _detect(li, board_size), _detect(ri, board_size)
        if lc is not None and rc is not None:
            obj_points.append(obj_template.copy())
            left_points.append(lc)
            right_points.append(rc)
    if len(obj_points) < 5:
        raise RuntimeError(f"Only {len(obj_points)} valid stereo pairs; need >=5")
    flags = cv2.CALIB_FIX_INTRINSIC if fix_intrinsics else 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K1, d1, K2, d2, R, T, E, F = cv2.stereoCalibrate(
        obj_points, left_points, right_points,
        left_calibration.K.copy(), left_calibration.distortion.copy(),
        right_calibration.K.copy(), right_calibration.distortion.copy(),
        (left_calibration.width, left_calibration.height),
        criteria=criteria, flags=flags,
    )
    T_right_left = make_se3(R, T.reshape(3))
    T_left_right = invert_se3(T_right_left)
    per_view = []
    for obj, lpts, rpts in zip(obj_points, left_points, right_points):
        # Stereo calibration RMS is global; these per-view values are a useful
        # diagnostic using the estimated left/right camera extrinsics.
        # Estimate each board pose in left camera then transform it into right.
        ok, rvec, tvec = cv2.solvePnP(obj, lpts, K1, d1)
        if not ok:
            continue
        lp, _ = cv2.projectPoints(obj, rvec, tvec, K1, d1)
        R_board, _ = cv2.Rodrigues(rvec)
        R_board_right = R @ R_board
        t_board_right = R @ tvec.reshape(3, 1) + T
        rvec_right, _ = cv2.Rodrigues(R_board_right)
        rp, _ = cv2.projectPoints(obj, rvec_right, t_board_right, K2, d2)
        e = np.concatenate([
            np.sum((lp.reshape(-1, 2) - lpts.reshape(-1, 2)) ** 2, axis=1),
            np.sum((rp.reshape(-1, 2) - rpts.reshape(-1, 2)) ** 2, axis=1),
        ])
        per_view.append(float(np.sqrt(np.mean(e))))
    return StereoCalibrationResult(
        left=CameraCalibration(K1, left_calibration.width, left_calibration.height, d1.reshape(-1), calibration_id="stereo_left"),
        right=CameraCalibration(K2, right_calibration.width, right_calibration.height, d2.reshape(-1), calibration_id="stereo_right"),
        R_right_left=R, T_right_left=T_right_left, T_left_right=T_left_right,
        essential=E, fundamental=F, rms=float(rms),
        per_view_reprojection_rmse=np.asarray(per_view),
    )


def reprojection_rmse(observed_uv: np.ndarray, projected_uv: np.ndarray) -> float:
    observed = np.asarray(observed_uv, dtype=np.float64)
    projected = np.asarray(projected_uv, dtype=np.float64)
    if observed.shape != projected.shape or observed.ndim != 2 or observed.shape[1] != 2:
        raise ValueError("observed and projected pixels must both have shape (N,2)")
    if len(observed) == 0:
        raise ValueError("at least one correspondence is required")
    return float(np.sqrt(np.mean(np.sum((observed - projected) ** 2, axis=1))))
