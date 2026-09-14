"""
CCTV Frame Extraction Script v2 - Canteen Hygiene Project
---------------------------------------------------------
Two-pass extraction:
  1. GENERAL pass  - sparse sampling across full videos (for hair_cap variety
     + general scene diversity)
  2. NO_CAP DENSE pass - dense sampling only around the known timestamps where
     the capless person appears, to maximize no_hair_cap examples with
     different angles/poses as they walk through frame.

This DELETES the old cctv_frames folder first and rebuilds it fresh.

Usage:
    python extract_frames.py
"""

import cv2
import os
import shutil

# ---- BASE DIR (auto-resolves to wherever this script is placed) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- CONFIG ----
VIDEOS = {
    "video1": os.path.join(BASE_DIR, "Screen Recording 2026-08-17 105642.mp4"),
    "video2": os.path.join(BASE_DIR, "Screen Recording 2026-08-17 111050.mp4"),
}

OUT_DIR = os.path.join(BASE_DIR, "cctv_frames")

GENERAL_INTERVAL_SEC = 1.5   # sparse pass across full video
NOCAP_INTERVAL_SEC = 0.3     # dense pass just around no-cap windows

JPEG_QUALITY = 95

# Known no_hair_cap windows: (video_key, start_sec, end_sec)
NOCAP_WINDOWS = [
    ("video1", 43, 47),
    ("video1", 127, 149),   # 2:07 - 2:29
    ("video1", 207, 232),   # 3:27 - 3:52
    ("video2", 50, 57),
    ("video2", 101, 103),   # 1:41 - 1:43
]


def get_video_props(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total_frames


def extract_general(video_key, video_path, out_dir, interval_sec, jpeg_quality, skip_windows):
    if not os.path.exists(video_path):
        print(f"  [SKIP] File not found: {video_path}")
        return 0

    fps, total_frames = get_video_props(video_path)
    frame_interval = max(1, int(fps * interval_sec))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"  [GENERAL] {video_key} | FPS: {fps:.2f} | Duration: {duration_sec:.1f}s")

    cap = cv2.VideoCapture(video_path)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    count, saved = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_sec = count / fps
        # skip frames that fall inside a no-cap window (handled by dense pass instead)
        in_skip_window = any(s <= current_sec <= e for (vk, s, e) in skip_windows if vk == video_key)

        if count % frame_interval == 0 and not in_skip_window:
            out_path = os.path.join(out_dir, f"{video_key}_general_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame, encode_params)
            saved += 1
        count += 1

    cap.release()
    print(f"    -> {saved} general frames saved\n")
    return saved


def extract_nocap_dense(video_key, video_path, out_dir, start_sec, end_sec, interval_sec, jpeg_quality):
    if not os.path.exists(video_path):
        return 0

    fps, total_frames = get_video_props(video_path)
    frame_interval = max(1, int(fps * interval_sec))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    saved = 0
    frame_idx = start_frame
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            out_path = os.path.join(
                out_dir, f"{video_key}_nocap_{start_sec}s_{saved:03d}.jpg"
            )
            cv2.imwrite(out_path, frame, encode_params)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"  [NO_CAP DENSE] {video_key} {start_sec}s-{end_sec}s -> {saved} frames")
    return saved


def main():
    # --- wipe old folder ---
    if os.path.exists(OUT_DIR):
        print(f"Deleting old folder: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fresh output directory created: {OUT_DIR}\n")

    total_general = 0
    total_nocap = 0

    print("=" * 60)
    print("PASS 1: GENERAL SPARSE EXTRACTION")
    print("=" * 60)
    for key, path in VIDEOS.items():
        total_general += extract_general(
            key, path, OUT_DIR, GENERAL_INTERVAL_SEC, JPEG_QUALITY, NOCAP_WINDOWS
        )

    print("=" * 60)
    print("PASS 2: NO_CAP DENSE EXTRACTION")
    print("=" * 60)
    for video_key, start_sec, end_sec in NOCAP_WINDOWS:
        video_path = VIDEOS[video_key]
        total_nocap += extract_nocap_dense(
            video_key, video_path, OUT_DIR, start_sec, end_sec, NOCAP_INTERVAL_SEC, JPEG_QUALITY
        )

    print("\n" + "=" * 60)
    print(f"DONE.")
    print(f"  General frames:  {total_general}")
    print(f"  No-cap frames:   {total_nocap}")
    print(f"  TOTAL frames:    {total_general + total_nocap}")
    print(f"  Saved to: {OUT_DIR}")
    print("=" * 60)
    print("\nNext step: skim the folder, delete any blurry/empty frames,")
    print("then upload the remaining images to Roboflow for labeling.")
    print("When labeling, make sure to label EVERY person in the")
    print("'nocap_' prefixed images carefully - these are your key")
    print("no_hair_cap examples across different angles/poses.")


if __name__ == "__main__":
    main()
