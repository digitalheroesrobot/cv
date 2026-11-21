#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 57 — ADVANCED SCENE FLOW AND MOTION PREDICTION

LEARNING OBJECTIVES
-------------------
1. Interpolate sparse correspondences into a local dense-ish 3D motion field.
2. Separate locally coherent (rigid-like) and inconsistent (non-rigid-like) flow.
3. Register consecutive dynamic object clouds with point-to-point ICP.
4. Predict future 3D object positions from tracked position and velocity.
5. Estimate closest approach and conservative sphere-based collision risk.
6. Visualize 3D motion vectors and future object trajectories.
7. Export motion and collision diagnostics.

This is an offline-first classical geometry lab. Input coordinates are metric
and must already be ego-motion compensated into a stable world frame. No
network access, pretrained model, or external service is used.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from common import ObjectTrack
from multi_camera.motion_prediction import (
    estimate_local_scene_flow, decompose_rigid_nonrigid,
    register_dynamic_cloud, predict_track, predict_pair_collision,
    motion_vector_field,
)


def make_sequence(frames=30, seed=57):
    rng = np.random.default_rng(seed)
    # Static background and a moving rigid object with persistent point samples.
    static = rng.uniform([-1.5, -0.8, 2.5], [1.5, -0.7, 3.5], (350, 3))
    local = rng.uniform([-.18, -.12, -.10], [.18, .12, .10], (160, 3))
    sequence = []
    for frame in range(frames):
        timestamp = frame / 15.0
        center = np.array([-.65 + .025*frame, -.05, 3.0])
        moving = local + center
        # A small deforming patch demonstrates non-rigid local flow.
        deform = moving.copy()
        deform[:, 1] += .025*np.sin(frame*.25 + (local[:, 0]+.18)*8)
        points = np.concatenate((static, deform), axis=0)
        points += rng.normal(0, .0015, points.shape)
        sequence.append((timestamp, points, static.copy(), deform, center))
    return sequence


def run_demo(frames=30, visualize=False, output_dir="outputs"):
    if frames < 3:
        raise ValueError("frames must be at least 3")
    sequence = make_sequence(frames)
    flow_reports = []
    previous_dynamic = None
    final_flow = None
    final_points = None
    registration = None
    for timestamp, points, static, dynamic, center in sequence:
        if previous_dynamic is not None:
            local = estimate_local_scene_flow(
                previous_dynamic, dynamic, 1/15,
                max_distance_m=.12, neighbors=5)
            decomposition = decompose_rigid_nonrigid(
                dynamic, local.flow, local.valid,
                neighborhood_radius_m=.12,
                rigid_residual_threshold_mps=.08,
                min_neighbors=3)
            flow_reports.append({
                "timestamp": timestamp,
                "valid_flow_points": int(local.valid.sum()),
                "mean_speed_mps": float(local.speed_mps[local.valid].mean())
                    if np.any(local.valid) else 0.0,
                "rigid_like_points": int(decomposition.rigid_mask.sum()),
                "nonrigid_like_points": int(decomposition.nonrigid_mask.sum()),
                "uncertain_points": int(decomposition.uncertain_mask.sum()),
            })
            registration = register_dynamic_cloud(
                previous_dynamic, dynamic, max_correspondence_m=.10,
                max_iterations=20, min_correspondences=10)
            final_flow, final_points = local, dynamic
        previous_dynamic = dynamic

    timestamp, points, static, dynamic, center = sequence[-1]
    # Two approaching objects; their future closest approach is assessed.
    object_a = ObjectTrack(
        101, "robot", np.array([-0.55, 0., 2.0]), timestamp,
        velocity=np.array([.35, 0., 0.]), dimensions=np.array([.5, .5, .7]))
    object_b = ObjectTrack(
        202, "obstacle", np.array([.65, 0., 2.0]), timestamp,
        velocity=np.array([-.10, 0., 0.]), dimensions=np.array([.45, .45, .6]))
    predictions = [predict_track(t, 2.0) for t in (object_a, object_b)]
    collision = predict_pair_collision(object_a, object_b, horizon_s=3.0,
                                       safety_margin_m=.08)
    elapsed_placeholder = 0.0
    report = {
        "lesson": 57, "frames": frames,
        "frame_history": flow_reports,
        "registration": None if registration is None else {
            "converged": registration.converged,
            "iterations": registration.iterations,
            "rmse_m": registration.rmse_m,
            "correspondences": registration.correspondences,
            "T_target_source": registration.T_target_source.tolist(),
        },
        "predictions": [{
            "track_id": p.track_id, "timestamp": p.timestamp,
            "horizon_s": p.horizon_s, "position_m": p.position.tolist(),
            "velocity_mps": p.velocity.tolist()
        } for p in predictions],
        "collision_prediction": {
            "track_id_a": collision.track_id_a,
            "track_id_b": collision.track_id_b,
            "time_to_closest_s": collision.time_to_closest_s,
            "closest_distance_m": collision.closest_distance_m,
            "combined_radius_m": collision.combined_radius_m,
            "collision_predicted": collision.collision_predicted,
        },
        "limitations": [
            "Local flow interpolation can smear motion across object boundaries.",
            "Local flow consistency is a rigid-like heuristic, not a rigid-body proof.",
            "ICP assumes sufficient overlap and a good initial alignment.",
            "Prediction assumes constant velocity and does not model control inputs.",
            "Collision checking uses circumscribed spheres, not exact oriented-box geometry.",
            "This is not a safety-rated robot collision avoidance system.",
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report["outputs"] = {"report": str(out / "lesson57_motion_prediction_report.json")}
    (out / "lesson57_motion_prediction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    if visualize and final_flow is not None:
        import matplotlib.pyplot as plt
        p, v = motion_vector_field(final_points, final_flow.flow,
                                   final_flow.valid, max_vectors=100)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(static[:, 0], static[:, 1], static[:, 2],
                   s=2, label="Static background")
        ax.scatter(final_points[:, 0], final_points[:, 1],
                   final_points[:, 2], s=5, label="Dynamic object")
        if len(p):
            ax.quiver(p[:, 0], p[:, 1], p[:, 2],
                      v[:, 0], v[:, 1], v[:, 2], length=1.0)
        for track, prediction in zip((object_a, object_b), predictions):
            ax.plot([track.position[0], prediction.position[0]],
                    [track.position[1], prediction.position[1]],
                    [track.position[2], prediction.position[2]],
                    marker="o", label=f"Track {track.track_id} prediction")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_zlabel("World Z (m)")
        ax.set_title("Lesson 57 — Scene Flow and Motion Prediction")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    report = run_demo(args.frames, args.visualize)
    print(json.dumps({
        "registration": report["registration"],
        "predictions": report["predictions"],
        "collision_prediction": report["collision_prediction"],
        "outputs": report["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
