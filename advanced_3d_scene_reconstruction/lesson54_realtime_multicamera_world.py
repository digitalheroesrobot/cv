#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LESSON 54 — REAL-TIME MULTI-CAMERA RGB-D WORLD RECONSTRUCTION

LEARNING OBJECTIVES
-------------------
1. Capture RGB-D samples concurrently from independent camera adapters.
2. Convert device timestamps into one configured master time domain.
3. Pair complete camera bundles under a maximum temporal skew.
4. Apply calibrated T_world_camera rig extrinsics.
5. Fuse per-camera depth clouds into one world-frame point cloud.
6. Monitor throughput, latency, frame drops, queue pressure, and errors.

OFFLINE MODE is the default and uses deterministic synthetic camera samples.
REALSENSE MODE requires local pyrealsense2, connected devices, a JSON file
containing measured rig extrinsics, and validated clock models if device clocks
do not already share a master time domain. No network access or auto-download.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

from common import CameraCalibration, CameraFrame, DepthMap, make_se3
from camera_adapters import CameraSample, RealSenseD435Adapter
from multi_camera import MultiCameraReconstructor, ClockModel
from multi_camera.capture import MultiCameraCaptureManager


class SyntheticRGBDAdapter:
    """Offline camera source with deterministic depth and moving foreground."""
    def __init__(self, camera_id, T_world_camera, fps=15, width=160, height=120,
                 frame_limit=45, phase=0.0):
        self.camera_id = camera_id
        self.T_world_camera = T_world_camera
        self.fps, self.width, self.height = fps, width, height
        self.frame_limit, self.phase = frame_limit, phase
        self.index = 0
        self.start_time = None
        self.calibration = None

    def open(self):
        self.index = 0
        self.start_time = time.monotonic()
        fx = fy = 140.0
        K = np.array([[fx, 0, self.width/2],
                      [0, fy, self.height/2], [0, 0, 1]], dtype=float)
        self.calibration = CameraCalibration(
            K, self.width, self.height, np.zeros(5), "pinhole",
            f"{self.camera_id}_synthetic")
    def read(self):
        if self.index >= self.frame_limit:
            return None
        target = self.start_time + self.index / self.fps
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        t = time.monotonic()
        yy, xx = np.mgrid[:self.height, :self.width]
        depth = np.full((self.height, self.width), 2.0, np.float32)
        # A simple foreground patch moves in image coordinates over time.
        cx = int(self.width * (0.35 + 0.18*np.sin(self.index*0.12+self.phase)))
        mask = (np.abs(xx-cx) < 12) & (np.abs(yy-self.height*0.55) < 16)
        depth[mask] = 1.25
        image = np.zeros((self.height, self.width, 3), np.uint8)
        image[..., 1] = 70
        image[mask] = (30, 30, 230)
        fid = self.index
        self.index += 1
        return CameraSample(
            CameraFrame(image, t, self.camera_id, fid, "BGR",
                        metadata={"source": "synthetic"}),
            self.calibration,
            DepthMap(depth, t, self.camera_id, unit="m"))
    def close(self):
        pass


def save_ply(path, points, colors=None):
    points = np.asarray(points, dtype=np.float32)
    if colors is None:
        colors = np.full((len(points), 3), 180, np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[2])} {int(c[1])} {int(c[0])}\n")


def load_pose_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    poses = {}
    for camera_id, matrix in data["T_world_camera"].items():
        poses[camera_id] = np.asarray(matrix, dtype=np.float64)
    return poses


def load_clock_config(path, camera_ids):
    if path is None:
        return {c: ClockModel() for c in camera_ids}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    models = {}
    for camera_id in camera_ids:
        item = data[camera_id]
        models[camera_id] = ClockModel(
            scale=float(item["scale"]),
            offset_s=float(item.get("offset_s", 0.0)),
            reference_device_s=float(item.get("reference_device_s", 0.0)))
    return models


def make_synthetic_rig(camera_ids):
    poses = {}
    for index, camera_id in enumerate(camera_ids):
        T = np.eye(4)
        T[0, 3] = index * 0.22
        T[1, 3] = (index % 2) * 0.08
        poses[camera_id] = T
    return poses


def run_pipeline(adapters, poses, clock_models, max_frames=30,
                 tolerance_s=0.025, visualize=False, output_dir="outputs"):
    camera_ids = list(adapters)
    reconstructor = MultiCameraReconstructor(
        poses, max_timestamp_skew_s=tolerance_s,
        stride=2, min_depth_m=0.2, max_depth_m=4.0,
        voxel_size_m=0.025, max_points=400_000)
    manager = MultiCameraCaptureManager(
        adapters, clock_models, tolerance_s=tolerance_s,
        queue_size=10, read_retry_s=0.003)
    clouds, frame_latencies, skews = [], [], []
    processed = 0
    started = time.perf_counter()
    with manager:
        while processed < max_frames:
            bundle = manager.read_bundle(timeout_s=2.0)
            if bundle is None:
                break
            t0 = time.perf_counter()
            result = reconstructor.process(bundle.samples)
            frame_latencies.append(time.perf_counter()-t0)
            skews.append(result.max_skew_s)
            clouds.append(result.fused_cloud)
            processed += 1
    elapsed = time.perf_counter() - started
    stats = manager.snapshot_stats()
    if clouds:
        points = np.concatenate([c.points for c in clouds], axis=0)
        colors = (np.concatenate([c.colors for c in clouds], axis=0)
                  if all(c.colors is not None for c in clouds) else None)
        # Keep the most recent bounded world snapshot, then voxel fuse it.
        from multi_camera.fusion import fuse_point_clouds
        from common import PointCloud
        cloud = PointCloud(points, clouds[-1].timestamp, "world", colors)
        # Reuse the same voxel aggregator with identity pose.
        cloud = fuse_point_clouds({"world": cloud}, {"world": np.eye(4)},
                                  target_frame="world", voxel_size=0.025)
    else:
        from common import PointCloud
        cloud = PointCloud(np.empty((0, 3)), 0.0, "world")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ply_path = output / "lesson54_realtime_multicamera_world.ply"
    report_path = output / "lesson54_realtime_multicamera_report.json"
    save_ply(ply_path, cloud.points, cloud.colors)
    report = {
        "lesson": 54, "offline_or_live": "live" if any(
            isinstance(a, RealSenseD435Adapter) for a in adapters.values()) else "synthetic",
        "camera_ids": camera_ids, "processed_bundles": processed,
        "elapsed_s": elapsed,
        "bundle_rate_hz": processed/elapsed if elapsed > 0 else 0.0,
        "mean_fusion_latency_ms": float(np.mean(frame_latencies)*1000) if frame_latencies else None,
        "max_fusion_latency_ms": float(np.max(frame_latencies)*1000) if frame_latencies else None,
        "mean_timestamp_skew_ms": float(np.mean(skews)*1000) if skews else None,
        "max_timestamp_skew_ms": float(np.max(skews)*1000) if skews else None,
        "world_points": int(len(cloud.points)),
        "capture_stats": stats,
        "outputs": {"point_cloud": str(ply_path), "report": str(report_path)},
        "limitations": [
            "Synthetic mode is a plumbing/performance demonstration, not physical sensor validation.",
            "Supplied extrinsics must be measured and expressed as T_world_camera.",
            "Identity clock models are valid only when all timestamps already share a validated clock domain.",
            "Software timestamp pairing does not synchronize exposure or remove rolling-shutter effects.",
            "This lesson fuses a bounded point-cloud snapshot; it is not dynamic SLAM or a temporally consistent object tracker."
        ]
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if visualize and len(cloud.points):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        step = max(1, len(cloud.points)//12000)
        p = cloud.points[::step]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=1)
        for camera_id, T in poses.items():
            c = T[:3, 3]
            ax.scatter(*c, marker="^", s=70, label=camera_id)
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_zlabel("World Z (m)")
        ax.set_title("Lesson 54 — Fused Multi-Camera World Snapshot")
        ax.legend()
        plt.tight_layout()
        plt.show()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--realsense", action="store_true",
                        help="Use connected Intel RealSense D435 cameras")
    parser.add_argument("--serials", type=str, default="",
                        help="Comma-separated device serial numbers")
    parser.add_argument("--poses-json", type=str,
                        help="JSON with T_world_camera matrices for live cameras")
    parser.add_argument("--clock-json", type=str,
                        help="Optional per-camera affine clock models")
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--tolerance-ms", type=float, default=25.0)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if args.realsense:
        serials = [s.strip() for s in args.serials.split(",") if s.strip()]
        if len(serials) < 2:
            parser.error("--realsense requires at least two serials via --serials")
        camera_ids = [f"cam{i}" for i in range(len(serials))]
        if not args.poses_json:
            parser.error("--realsense requires --poses-json with measured rig extrinsics")
        poses = load_pose_config(args.poses_json)
        if set(poses) != set(camera_ids):
            parser.error("poses JSON camera IDs must match generated cam0, cam1, ... IDs")
        adapters = {
            camera_id: RealSenseD435Adapter(
                camera_id=camera_id, serial=serial)
            for camera_id, serial in zip(camera_ids, serials)
        }
    else:
        if args.cameras < 2:
            parser.error("--cameras must be at least 2")
        camera_ids = [f"cam{i}" for i in range(args.cameras)]
        poses = make_synthetic_rig(camera_ids)
        adapters = {
            camera_id: SyntheticRGBDAdapter(
                camera_id, poses[camera_id], frame_limit=args.frames, phase=i*0.7)
            for i, camera_id in enumerate(camera_ids)
        }
    clocks = load_clock_config(args.clock_json, camera_ids)
    report = run_pipeline(
        adapters, poses, clocks, max_frames=args.frames,
        tolerance_s=args.tolerance_ms/1000.0,
        visualize=args.visualize)
    print(json.dumps({
        "mode": report["offline_or_live"],
        "processed_bundles": report["processed_bundles"],
        "bundle_rate_hz": report["bundle_rate_hz"],
        "mean_fusion_latency_ms": report["mean_fusion_latency_ms"],
        "max_timestamp_skew_ms": report["max_timestamp_skew_ms"],
        "world_points": report["world_points"],
        "outputs": report["outputs"]
    }, indent=2))


if __name__ == "__main__":
    main()
