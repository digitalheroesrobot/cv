#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 50 — ROBUST SLAM, SUBMAP MANAGEMENT AND RELOCALIZATION

LEARNING OBJECTIVES
-------------------
1. Detect tracking degradation and lost tracking with hysteresis.
2. Start bounded local submaps after a tracking reset.
3. Rank old keyframes for relocalization without treating appearance as pose.
4. Reject duplicate or geometrically inconsistent loop constraints.
5. Convert pose covariance into information weights for graph optimization.
6. Enforce explicit keyframe, point, and payload-memory budgets.
7. Monitor frame timing and tracking failures during long sessions.

This is an offline, deterministic engineering lab. The reusable classes are
also suitable for supervising a live camera pipeline. It does not claim that
synthetic descriptors or a synthetic pose graph constitute a real relocalizer.
No network access or automatic downloads are used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from keyframe_slam.robust_slam import (
    TrackingMonitor, TrackingState, SubmapManager, LoopClosurePolicy,
    information_from_covariance, whiten_pose_residual, MemoryBudget,
    SLAMHealthMonitor,
)


def make_pose(x=0.0, y=0.0, yaw=0.0):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    T[:3, 3] = [x, y, 0.0]
    return T


def run_demo(seed=50):
    rng = np.random.default_rng(seed)

    # 1) A brief tracking failure must transition through a controlled state.
    tracker = TrackingMonitor(lost_after=2, recover_after=2)
    tracking_trace = []
    for score in [0.9, 0.85, 0.2, 0.1, 0.8, 0.9]:
        decision = tracker.update(score, pose_valid=True)
        tracking_trace.append({"score": score, "state": decision.state.value,
                               "accepted": decision.accepted,
                               "new_submap": decision.start_new_submap})

    # 2) Submaps and memory remain bounded as the session grows.
    submaps = SubmapManager(max_submaps=3, max_keyframes_per_submap=4,
                            max_total_points=5000)
    memory = MemoryBudget(max_keyframes=6, max_points=5000,
                          max_payload_bytes=2_000_000)
    for k in range(12):
        if submaps.current is None or len(submaps.current.keyframe_ids) >= 4:
            submaps.start(make_pose(k * 0.1, 0.0))
        submaps.add_keyframe(k, 300)
        memory.register(k, payload_bytes=220_000, point_count=300)
        memory.enforce()

    # 3) Loop closure is accepted only if independent geometry agrees.
    policy = LoopClosurePolicy(min_inliers=18, max_reprojection_px=2.0)
    T = make_pose(0.5, -0.1, 0.08)
    T_consistent = T.copy()
    accepted = policy.evaluate(0, 11, 42, 0.7, T, T_consistent)
    duplicate = policy.evaluate(0, 11, 42, 0.7, T, T_consistent)
    bad = policy.evaluate(1, 10, 35, 0.8, T, make_pose(1.0, 0.0, 0.5))

    # 4) Covariance -> information; whitened residual has finite energy.
    std = np.array([0.02, 0.02, 0.03, 0.01, 0.01, 0.02])
    covariance = np.diag(std ** 2)
    information = information_from_covariance(covariance)
    residual = rng.normal(0.0, 0.01, 6)
    whitened = whiten_pose_residual(residual, information)

    # 5) Rolling health metrics surface frame drops and excessive latency.
    health = SLAMHealthMonitor(window=5, max_failure_rate=0.3,
                               max_frame_interval_s=0.25)
    health_trace = []
    for i, (ok, dt) in enumerate([(True, 0.033), (True, 0.034),
                                  (False, 0.40), (True, 0.033),
                                  (False, 0.50)]):
        t = i * 0.033 if i < 2 else (0.067 + (i - 2) * dt)
        health_trace.append(health.update(t, ok, map_points=1200))

    mem = memory.snapshot()
    report = {
        "lesson": 50,
        "offline": True,
        "tracking_trace": tracking_trace,
        "submaps_retained": len(submaps.submaps),
        "submap_ids": [s.submap_id for s in submaps.submaps],
        "submap_keyframes": [s.keyframe_ids for s in submaps.submaps],
        "memory": {"keyframes": mem.keyframes, "points": mem.point_count,
                   "estimated_payload_bytes": mem.estimated_bytes,
                   "within_budget": mem.within_budget},
        "loop_closure": {
            "consistent_accepted": accepted.accepted,
            "consistent_reason": accepted.reason,
            "duplicate_rejected": not duplicate.accepted,
            "duplicate_reason": duplicate.reason,
            "inconsistent_rejected": not bad.accepted,
            "inconsistent_reason": bad.reason,
            "translation_residual_m": bad.translation_residual_m,
            "rotation_residual_rad": bad.rotation_residual_rad,
        },
        "uncertainty": {
            "information_eigenvalues": np.linalg.eigvalsh(information).tolist(),
            "whitened_residual_norm": float(np.linalg.norm(whitened)),
        },
        "health_trace": health_trace,
        "limitations": [
            "Appearance retrieval ranks candidates only; real RGB-D PnP/ICP must verify them.",
            "Submap metadata merge does not transform or fuse point-cloud geometry.",
            "The 6x6 covariance utility must receive covariance in the same tangent ordering and frame as the pose residual.",
            "Payload accounting estimates registered NumPy payload bytes, not process RSS or GPU memory.",
            "This lab does not implement IMU fusion, a production relocalizer, or a globally consistent multi-session map."
        ]
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--output", default="outputs/lesson50_robust_slam.json")
    args = parser.parse_args()
    report = run_demo(args.seed)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "tracking_states": [x["state"] for x in report["tracking_trace"]],
        "submaps_retained": report["submaps_retained"],
        "memory_within_budget": report["memory"]["within_budget"],
        "loop_closure": report["loop_closure"],
        "health_warnings": [x["warnings"] for x in report["health_trace"]],
        "report": str(path)
    }, indent=2))


if __name__ == "__main__":
    main()
