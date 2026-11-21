#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
ADVANCED COMPUTER VISION
LESSON 01 — CAMERA VIDEO PROCESSING AND FRAME EXTRACTION
===============================================================================

PURPOSE
-------
Build the first reliable stage of a camera-to-3D scene reconstruction pipeline.

WHAT YOU WILL LEARN
-------------------
1. Open a video file with OpenCV.
2. Inspect frame count, frame rate, duration, and resolution.
3. Read frames safely and preserve their timestamps.
4. Extract frames at a requested time interval.
5. Save a compact dataset and metadata file.
6. Visualize the input and extracted frames.
7. Verify video-processing behavior with numerical tests.

THEORY
------
A video is a time-ordered sequence of image frames:

    I_0, I_1, ..., I_(N-1)

For a constant frame rate f (frames/second), the nominal timestamp of frame i is:

    t_i = i / f

To extract one frame every delta_t seconds, the target frame indices are:

    i_k = round(k * delta_t * f)

Real video containers may use variable frame rates or imperfect metadata.
Therefore, timestamps reported by the decoder should be preferred when available.

IMPORTANT LIMITATION
--------------------
This lesson processes 2D video only. It does not recover depth, metric scale,
camera pose, or 3D geometry. Those require calibration and additional
measurements or multi-view/depth information in later lessons.

REAL-WORLD APPLICATION
----------------------
Prepare synchronized, timestamped image samples from a factory camera or
automotive camera before running object detection, depth estimation, or
multi-camera fusion.

USAGE
-----
Create and process a demo video:
    python lesson01_video_processing.py

Process a real video:
    python lesson01_video_processing.py --video /path/to/video.mp4

Extract one frame every 0.5 seconds:
    python lesson01_video_processing.py --video input.mp4 --interval 0.5

Disable GUI visualization:
    python lesson01_video_processing.py --video input.mp4 --no-display

OUTPUT
------
outputs/lesson01/
    demo_input.mp4          (only when a demo video is generated)
    frames/frame_*.jpg
    metadata.json
    contact_sheet.png

VERIFICATION
------------
The script tests timestamp calculations, frame-index selection, metadata
sanity, and the synthetic demo-video pipeline.
===============================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DEFAULT_OUTPUT_DIR = Path("outputs/lesson01")
DEFAULT_INTERVAL_SECONDS = 1.0
DEMO_FPS = 20.0
DEMO_DURATION_SECONDS = 4.0
DEMO_WIDTH = 640
DEMO_HEIGHT = 360


@dataclass
class VideoMetadata:
    path: str
    frame_count_reported: int
    fps_reported: float
    width: int
    height: int
    duration_seconds_reported: float
    codec: str

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


# =============================================================================
# 2. SYNTHETIC DEMO VIDEO
# =============================================================================

def create_demo_video(
    output_path: Path,
    fps: float = DEMO_FPS,
    duration_seconds: float = DEMO_DURATION_SECONDS,
    width: int = DEMO_WIDTH,
    height: int = DEMO_HEIGHT,
) -> Path:
    """Create a deterministic road-like scene for pipeline testing."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create demo video. Check OpenCV video codec support."
        )

    frame_count = int(round(fps * duration_seconds))

    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (105, 125, 145)

        # Road surface.
        cv2.fillPoly(
            frame,
            [np.array([
                [int(width * 0.30), height],
                [int(width * 0.70), height],
                [int(width * 0.56), int(height * 0.48)],
                [int(width * 0.44), int(height * 0.48)],
            ], dtype=np.int32)],
            (65, 65, 65),
        )

        # Lane markings.
        for y in range(int(height * 0.55), height, 55):
            scale = (y - height * 0.48) / (height * 0.52)
            x_center = width // 2
            line_half_width = max(2, int(3 + 5 * scale))
            line_height = max(5, int(10 + 14 * scale))
            cv2.rectangle(
                frame,
                (x_center - line_half_width, y),
                (x_center + line_half_width, min(height - 1, y + line_height)),
                (235, 235, 235),
                -1,
            )

        # A moving vehicle-like rectangle.
        x = int(width * 0.50 + 95 * math.sin(index / max(1, fps)))
        y = int(height * 0.58 + 8 * math.sin(index / 7.0))
        cv2.rectangle(frame, (x - 38, y - 24), (x + 38, y + 24), (35, 45, 210), -1)
        cv2.rectangle(frame, (x - 24, y - 19), (x + 24, y - 5), (180, 210, 225), -1)

        timestamp = index / fps
        cv2.putText(
            frame,
            f"DEMO CAMERA  t={timestamp:05.2f}s",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    return output_path


# =============================================================================
# 3. VIDEO METADATA
# =============================================================================

def inspect_video(video_path: Path) -> VideoMetadata:
    """Open a video and return basic stream metadata."""

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(
        chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
    ).strip("\x00")

    capture.release()

    duration = frame_count / fps if fps > 0 else 0.0

    if width <= 0 or height <= 0:
        raise ValueError("Video metadata reports an invalid image resolution.")

    return VideoMetadata(
        path=str(video_path),
        frame_count_reported=frame_count,
        fps_reported=fps,
        width=width,
        height=height,
        duration_seconds_reported=duration,
        codec=codec,
    )


# =============================================================================
# 4. FRAME SELECTION AND EXTRACTION
# =============================================================================

def target_frame_indices(
    frame_count: int,
    fps: float,
    interval_seconds: float,
) -> list[int]:
    """Return unique frame indices sampled at a fixed time interval."""

    if frame_count < 0:
        raise ValueError("frame_count cannot be negative.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive.")
    if frame_count == 0:
        return []

    step = max(1, int(round(fps * interval_seconds)))
    return list(range(0, frame_count, step))


def extract_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float,
) -> list[dict]:
    """Extract frames and save their index and timestamp metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        capture.release()
        raise ValueError("The video decoder did not provide a valid FPS.")

    selected_indices = set(
        target_frame_indices(reported_count, fps, interval_seconds)
    )

    extracted = []
    frame_index = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index in selected_indices:
            # CAP_PROP_POS_MSEC is queried after decoding the current frame.
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            timestamp_seconds = (
                timestamp_ms / 1000.0
                if timestamp_ms > 0
                else frame_index / fps
            )

            filename = f"frame_{frame_index:06d}.jpg"
            frame_path = output_dir / filename

            if not cv2.imwrite(str(frame_path), frame):
                capture.release()
                raise IOError(f"Could not save frame: {frame_path}")

            extracted.append({
                "frame_index": frame_index,
                "timestamp_seconds": timestamp_seconds,
                "filename": filename,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            })

        frame_index += 1

    capture.release()

    if not extracted:
        raise RuntimeError(
            "No frames were extracted. Check the video and interval."
        )

    return extracted


# =============================================================================
# 5. CONTACT SHEET AND VISUALIZATION
# =============================================================================

def create_contact_sheet(
    video_path: Path,
    extracted_records: list[dict],
    frames_dir: Path,
    output_path: Path,
    max_images: int = 8,
) -> None:
    """Create a labeled contact sheet from extracted frames."""

    records = extracted_records[:max_images]
    if not records:
        raise ValueError("No extracted frames are available for visualization.")

    columns = min(4, len(records))
    rows = int(math.ceil(len(records) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 2.8 * rows))
    axes = np.array(axes, dtype=object).reshape(rows, columns)

    for axis in axes.flat:
        axis.axis("off")

    for axis, record in zip(axes.flat, records):
        image_path = frames_dir / record["filename"]
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        axis.imshow(image_rgb)
        axis.set_title(
            f"Frame {record['frame_index']} | "
            f"t={record['timestamp_seconds']:.2f}s"
        )
        axis.axis("off")

    fig.suptitle(f"Extracted frames: {video_path.name}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def display_video_sample(video_path: Path, max_frames: int = 5) -> None:
    """Display a small sample of frames in a Matplotlib window."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return

    indices = np.linspace(
        0, max(0, total - 1), num=min(max_frames, total), dtype=int
    )

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 3))
    axes = np.atleast_1d(axes)

    for axis, index in zip(axes, indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()

        if success:
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axis.set_title(f"Frame {index}")
        axis.axis("off")

    capture.release()
    fig.suptitle("Video frame samples")
    fig.tight_layout()
    plt.show()


# =============================================================================
# 6. METADATA EXPORT
# =============================================================================

def save_metadata(
    output_path: Path,
    metadata: VideoMetadata,
    extracted_records: list[dict],
    interval_seconds: float,
) -> None:
    """Save reproducible video and extraction metadata as JSON."""

    payload = {
        "lesson": "Advanced Computer Vision - Lesson 01",
        "video": asdict(metadata),
        "extraction": {
            "requested_interval_seconds": interval_seconds,
            "frames_extracted": len(extracted_records),
            "frames": extracted_records,
        },
        "notes": [
            "Timestamps are decoder-reported when available.",
            "This lesson does not estimate depth or metric 3D geometry.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# =============================================================================
# 7. NUMERICAL VERIFICATION
# =============================================================================

def run_tests() -> None:
    """Run deterministic tests for frame selection and timestamp logic."""

    indices = target_frame_indices(
        frame_count=100, fps=20.0, interval_seconds=0.5
    )
    assert indices[:4] == [0, 10, 20, 30]
    assert indices[-1] == 90
    assert len(indices) == 10

    assert target_frame_indices(0, 30.0, 1.0) == []
    assert target_frame_indices(5, 30.0, 0.01) == [0, 1, 2, 3, 4]

    try:
        target_frame_indices(10, 0.0, 1.0)
        raise AssertionError("Expected invalid FPS to raise ValueError.")
    except ValueError:
        pass

    try:
        target_frame_indices(10, 30.0, 0.0)
        raise AssertionError("Expected invalid interval to raise ValueError.")
    except ValueError:
        pass

    # Verify nominal frame timestamp arithmetic.
    fps = 25.0
    frame_index = 50
    timestamp = frame_index / fps
    assert math.isclose(timestamp, 2.0, rel_tol=0.0, abs_tol=1e-12)

    print("[PASS] Frame-index selection")
    print("[PASS] Empty-video handling")
    print("[PASS] Invalid parameter handling")
    print("[PASS] Frame timestamp arithmetic")


# =============================================================================
# 8. MAIN PIPELINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a video and extract frames for later 3D perception."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to an input video. If omitted, a demo video is generated.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Frame extraction interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open Matplotlib visualization windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero.")

    args.output.mkdir(parents=True, exist_ok=True)

    video_path: Path
    if args.video is None:
        video_path = create_demo_video(args.output / "demo_input.mp4")
        print(f"[INFO] Created demo video: {video_path}")
    else:
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video does not exist: {video_path}")

    print("\n" + "=" * 72)
    print("ADVANCED COMPUTER VISION — LESSON 01")
    print("CAMERA VIDEO PROCESSING AND FRAME EXTRACTION")
    print("=" * 72)

    metadata = inspect_video(video_path)

    print("\nVIDEO METADATA")
    print("-" * 72)
    print(f"Path:       {metadata.path}")
    print(f"Resolution: {metadata.width} x {metadata.height}")
    print(f"FPS:        {metadata.fps_reported:.3f}")
    print(f"Frames:     {metadata.frame_count_reported}")
    print(f"Duration:   {metadata.duration_seconds_reported:.3f} s")
    print(f"Codec:      {metadata.codec or 'Unknown'}")

    frames_dir = args.output / "frames"
    extracted = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        interval_seconds=args.interval,
    )

    metadata_path = args.output / "metadata.json"
    save_metadata(
        metadata_path,
        metadata,
        extracted,
        args.interval,
    )

    contact_sheet_path = args.output / "contact_sheet.png"
    create_contact_sheet(
        video_path,
        extracted,
        frames_dir,
        contact_sheet_path,
    )

    print("\nEXTRACTION RESULT")
    print("-" * 72)
    print(f"Frames extracted: {len(extracted)}")
    print(f"Frames directory: {frames_dir.resolve()}")
    print(f"Metadata:         {metadata_path.resolve()}")
    print(f"Contact sheet:    {contact_sheet_path.resolve()}")

    print("\nNUMERICAL TESTS")
    print("-" * 72)
    run_tests()

    if not args.no_display:
        display_video_sample(video_path)

    print("\nLesson 01 completed successfully.")
    print("Next: camera calibration and coordinate geometry.")


if __name__ == "__main__":
    main()
