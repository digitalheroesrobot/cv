#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 55 — MULTI-CAMERA 3D OBJECT TRACKING AND DATA ASSOCIATION

LEARNING OBJECTIVES
-------------------
1. Represent per-camera 3D detections in one calibrated world frame.
2. Fuse duplicate observations across cameras without merging same-camera objects.
3. Associate fused detections with persistent tracks using Hungarian assignment.
4. Estimate 3D position and velocity with a constant-velocity Kalman filter.
5. Gate associations by class, Mahalanobis distance, size, and optional embedding.
6. Maintain IDs through short occlusions and expire stale tracks.
7. Visualize trajectories and report tracking quality.

This is an offline algorithm lab with synthetic detections. It does not include
a neural image detector, learned re-identification network, or camera calibration.
No network access or model downloads are used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from multi_camera.object_tracking import Detection3D, MultiCamera3DTracker


def make_synthetic_sequence(frames=100, camera_ids=("cam0", "cam1", "cam2"),
                            seed=12):
    rng = np.random.default_rng(seed)
    sequence = []
    for frame in range(frames):
        timestamp = frame / 15.0
        detections = []
        # Two same-class objects cross in image/world space; their persistent
        # IDs must be maintained by motion prediction, not nearest-neighbour only.
        trajectories = [
            (np.array([-0.8 + 0.018*frame, 0.0, 2.5]), np.array([0.42, 0.28, 0.32])),
            (np.array([0.8 - 0.018*frame, 0.05, 2.8]), np.array([0.38, 0.30, 0.35])),
        ]
        for object_index, (position, dimensions) in enumerate(trajectories):
            if frame in range(38, 43) and object_index == 0:
                continue  # short occlusion
            for camera_index, camera_id in enumerate(camera_ids):
                # Simulate each camera seeing only part of the scene.
                if (object_index + camera_index + frame//17) % 4 == 0:
                    continue
                noise = rng.normal(0, 0.018, 3)
                center = position + noise
                embedding = np.zeros(8)
                embedding[object_index] = 1.0
                detections.append(Detection3D(
                    center, dimensions * (1+rng.normal(0, .025)),
                    "box", timestamp, camera_id,
                    confidence=float(rng.uniform(.72, .98)),
                    embedding=embedding))
        sequence.append((timestamp, detections))
    return sequence


def run_demo(frames=100, visualize=False, output_dir="outputs"):
    tracker = MultiCamera3DTracker(
        max_mahalanobis=18.0, max_missed_frames=8,
        max_track_age_s=1.0, min_hits=2,
        acceleration_noise_mps2=1.0, measurement_noise_m=0.035)
    history = {}
    reports = []
    started = time.perf_counter()
    for timestamp, detections in make_synthetic_sequence(frames=frames):
        tracks = tracker.update(detections, timestamp)
        for track in tracks:
            history.setdefault(track.track_id, []).append(
                [timestamp, *track.position.tolist(), *track.velocity.tolist()])
        reports.append({
            "timestamp": timestamp,
            "detections": len(detections),
            "confirmed_tracks": len(tracks),
            "track_ids": [t.track_id for t in tracks],
        })
    elapsed = time.perf_counter() - started
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "lesson55_trajectories.json"
    report_path = output / "lesson55_tracking_report.json"
    trajectory_path.write_text(json.dumps({
        str(k): v for k, v in history.items()
    }, indent=2), encoding="utf-8")
    final_tracks = tracker.snapshot(confirmed_only=False)
    report = {
        "lesson": 55,
        "frames": frames,
        "elapsed_s": elapsed,
        "processing_rate_hz": frames/elapsed if elapsed > 0 else 0.0,
        "final_active_tracks": tracker.diagnostics()["active_tracks"],
        "final_confirmed_tracks": tracker.diagnostics()["confirmed_tracks"],
        "trajectory_count": len(history),
        "final_tracks": [
            {"track_id": t.track_id, "label": t.label,
             "position_m": t.position.tolist(),
             "velocity_mps": t.velocity.tolist(),
             "dimensions_m": t.dimensions.tolist()}
            for t in final_tracks
        ],
        "frames": reports,
        "outputs": {"trajectories": str(trajectory_path),
                    "report": str(report_path)},
        "limitations": [
            "Synthetic detections validate the association/tracking backend, not a detector.",
            "All detections must already be in one calibrated world coordinate frame.",
            "Cross-camera fusion uses geometric proximity and optional embeddings.",
            "Constant-velocity Kalman motion is a baseline and can fail under abrupt acceleration.",
            "No learned ReID, occlusion reasoning, tracklet graph, or global multi-object tracking optimization."
        ]
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if visualize:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        for track_id, values in history.items():
            a = np.asarray(values)
            ax.plot(a[:, 1], a[:, 2], a[:, 3], label=f"Track {track_id}")
            ax.scatter(a[-1, 1], a[-1, 2], a[-1, 3], s=45)
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_zlabel("World Z (m)")
        ax.set_title("Lesson 55 — Multi-Camera 3D Object Trajectories")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    report = run_demo(args.frames, args.visualize)
    print(json.dumps({
        "processing_rate_hz": report["processing_rate_hz"],
        "trajectory_count": report["trajectory_count"],
        "final_tracks": report["final_tracks"],
        "outputs": report["outputs"]
    }, indent=2))


if __name__ == "__main__":
    main()
