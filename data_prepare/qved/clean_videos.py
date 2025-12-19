import json
import cv2
import os
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path

# ---------------- CONFIG ---------------- #

MIN_WIDTH = 320
MIN_HEIGHT = 240
MIN_BRIGHTNESS = 40
MIN_SHARPNESS = 30
MIN_MOTION = 1.0
MAX_FRAMES_TO_SAMPLE = 30

# ---------------------------------------- #


def sample_frames(cap, max_frames):
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return frames

    step = max(1, total // max_frames)
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(frame)
        idx += 1

    return frames


def compute_metrics(frames):
    brightness = []
    sharpness = []
    motion = []

    prev_gray = None

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness.append(np.mean(gray))
        sharpness.append(cv2.Laplacian(gray, cv2.CV_64F).var())

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motion.append(np.mean(diff))

        prev_gray = gray

    return {
        "brightness": np.mean(brightness) if brightness else 0,
        "sharpness": np.mean(sharpness) if sharpness else 0,
        "motion": np.mean(motion) if motion else 0,
    }


def is_video_valid(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot_open"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        cap.release()
        return False, "low_resolution"

    frames = sample_frames(cap, MAX_FRAMES_TO_SAMPLE)
    cap.release()

    if not frames:
        return False, "no_frames"

    metrics = compute_metrics(frames)

    if metrics["brightness"] < MIN_BRIGHTNESS:
        return False, "too_dark"

    if metrics["sharpness"] < MIN_SHARPNESS:
        return False, "blurry"

    if metrics["motion"] < MIN_MOTION:
        return False, "static"

    return True, None


def main(dataset_root, delete_bad):
    gt_path = Path(dataset_root) / "ground_truth.json"

    with open(gt_path, "r") as f:
        data = json.load(f)

    cleaned = {}
    removed = []

    for label, videos in tqdm(data.items(), desc="Cleaning videos"):
        valid_videos = []

        for video_path in videos:
            video_path = Path(video_path)
            ok, reason = is_video_valid(video_path)

            if ok:
                valid_videos.append(str(video_path))
            else:
                removed.append((str(video_path), reason))
                if delete_bad and video_path.exists():
                    video_path.unlink()

        if valid_videos:
            cleaned[label] = valid_videos

    # Save cleaned JSON
    with open(gt_path, "w") as f:
        json.dump(cleaned, f, indent=2)

    # Save report
    report_path = Path(dataset_root) / "cleaning_report.csv"
    with open(report_path, "w") as f:
        f.write("video_path,reason\n")
        for v, r in removed:
            f.write(f"{v},{r}\n")

    print(f"\nCleaning completed")
    print(f"Removed videos: {len(removed)}")
    print(f"Updated: {gt_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", help="Dataset root directory")
    parser.add_argument(
        "--delete-bad",
        action="store_true",
        help="Delete bad videos instead of just removing from JSON",
    )
    args = parser.parse_args()

    main(args.dataset_root, args.delete_bad)
