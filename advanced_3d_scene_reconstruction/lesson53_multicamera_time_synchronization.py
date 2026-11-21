#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 53 — MULTI-CAMERA CLOCK SYNCHRONIZATION AND TIME ALIGNMENT

LEARNING OBJECTIVES
-------------------
1. Distinguish sensor clock time, host arrival time, and master time.
2. Estimate clock offset and drift from paired synchronization events.
3. Convert independent camera timestamps into a common clock domain.
4. Pair frames under a maximum skew budget and account for dropped frames.
5. Interpolate camera poses with linear translation and SO(3) SLERP.

The synthetic lab is deterministic and offline. It simulates camera clocks
with independent offsets and ppm drift, plus a synchronization-event outlier.
No camera, network, cloud service, or automatic download is used.

REAL HARDWARE NOTE
------------------
A shared hardware trigger or a precisely timestamped synchronization pulse
is needed to generate trustworthy clock-pair observations. Pairing arbitrary
frame arrival times is not clock calibration because USB/host transport latency
is variable. RealSense metadata exposes a timestamp domain; do not assume
independent devices share the same clock merely because timestamps are in ms.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from common import CameraFrame, CameraPose, make_se3
from multi_camera.time_sync import (
    ClockModel, MultiCameraClockSynchronizer, fit_clock_model,
    interpolate_camera_pose,
)


def simulate_clock_pairs(seed=53, n=80):
    rng = np.random.default_rng(seed)
    master = np.linspace(10.0, 30.0, n)
    specs = {
        "cam0": (1.0 + 18e-6, 0.0),
        "cam1": (1.0 - 42e-6, 0.137),
        "cam2": (1.0 + 73e-6, -0.084),
    }
    observations = {}
    truth = {}
    for camera_id, (scale, offset) in specs.items():
        # Invert master = scale * device + offset.
        device = (master - offset) / scale
        measured = master + rng.normal(0.0, 0.00018, n)
        if camera_id == "cam2":
            measured[17] += 0.018  # One bad synchronization observation.
        observations[camera_id] = (device, measured)
        truth[camera_id] = (scale, offset)
    return observations, truth


def run_demo(seed=53, visualize=False):
    observations, truth = simulate_clock_pairs(seed)
    models, fit_reports = {}, {}
    for camera_id, (device, master) in observations.items():
        fit = fit_clock_model(device, master, outlier_threshold_s=0.001)
        models[camera_id] = fit.model
        fit_reports[camera_id] = {
            "estimated_offset_at_reference_s": fit.model.offset_s,
            "estimated_drift_ppm": fit.model.drift_ppm,
            "fit_rmse_s": fit.model.residual_rmse_s,
            "inliers": int(np.count_nonzero(fit.inlier_mask)),
            "rejected": int(len(fit.inlier_mask) - np.count_nonzero(fit.inlier_mask)),
            "true_offset_at_reference_s": (truth[camera_id][0] - 1.0) * np.mean(observations[camera_id][0]) + truth[camera_id][1],
            "true_drift_ppm": (truth[camera_id][0] - 1.0) * 1e6,
        }
    sync = MultiCameraClockSynchronizer(
        ["cam0", "cam1", "cam2"], models, tolerance_s=0.002, queue_size=16)
    # Feed asynchronous frame arrivals; timestamps are in each local device clock.
    local_frame_times = {
        c: observations[c][0][::4] for c in ("cam0", "cam1", "cam2")
    }
    for frame_index in range(len(local_frame_times["cam0"])):
        for camera_id in ("cam2", "cam0", "cam1"):
            t = float(local_frame_times[camera_id][frame_index])
            sync.push(CameraFrame(np.zeros((2, 2, 3), np.uint8), t,
                                  camera_id, frame_index, "BGR"))
    matched, skews = 0, []
    while True:
        item = sync.pop()
        if item is None:
            break
        matched += 1
        skews.append(item.max_skew_s)
    # Pose interpolation test: 10 cm translation and 20 degree yaw over 50 ms.
    Ta = make_se3(np.eye(3), np.array([0.0, 0.0, 0.0]))
    Tb = make_se3(Rotation.from_euler("z", 20, degrees=True).as_matrix(),
                  np.array([0.10, 0.0, 0.0]))
    pa = CameraPose(Ta, 1.0, "world", "cam0")
    pb = CameraPose(Tb, 1.05, "world", "cam0")
    pm = interpolate_camera_pose(pa, pb, 1.025)
    report = {
        "lesson": 53,
        "offline": True,
        "clock_fits": fit_reports,
        "frame_sync": {
            "matched_sets": matched,
            "mean_skew_s": float(np.mean(skews)) if skews else None,
            "max_skew_s": float(np.max(skews)) if skews else None,
            "emitted": sync.emitted,
            "rejected_sets": sync.rejected_sets,
            "camera_stats": sync.stats,
        },
        "pose_interpolation": {
            "timestamp_s": pm.timestamp,
            "translation_m": pm.T_world_camera[:3, 3].tolist(),
            "rotation_yaw_deg": float(Rotation.from_matrix(
                pm.T_world_camera[:3, :3]).as_euler("xyz", degrees=True)[2]),
        },
        "limitations": [
            "Clock fit requires paired observations of the same synchronization event.",
            "The synthetic event pairing is idealized; real trigger/transport uncertainty must be measured.",
            "Software frame pairing does not provide hardware exposure synchronization.",
            "Linear translation plus SLERP is not a continuous-time motion model."
        ]
    }
    output = Path("outputs")
    output.mkdir(parents=True, exist_ok=True)
    (output / "lesson53_time_sync_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if visualize:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))
        names = list(fit_reports)
        offsets = [fit_reports[n]["estimated_offset_s"] * 1000 for n in names]
        drifts = [fit_reports[n]["estimated_drift_ppm"] for n in names]
        axes[0].bar(names, offsets)
        axes[0].set_ylabel("Estimated offset (ms)")
        axes[0].set_title("Camera clock offset estimates")
        axes[1].bar(names, drifts)
        axes[1].set_ylabel("Estimated drift (ppm)")
        axes[1].set_title("Camera clock drift estimates")
        axes[1].grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=53)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    report = run_demo(args.seed, args.visualize)
    print(json.dumps({
        "clock_fits": report["clock_fits"],
        "frame_sync": report["frame_sync"],
        "pose_interpolation": report["pose_interpolation"],
        "report": "outputs/lesson53_time_sync_report.json"
    }, indent=2))


if __name__ == "__main__":
    main()
