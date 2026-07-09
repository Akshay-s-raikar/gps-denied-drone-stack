#!/usr/bin/env python3
"""
camera_bridge.py — IRoC-U 2026
================================
Reads from real camera (video0) and writes to v4l2loopback (video10).
Allows multiple processes to read the same camera simultaneously:
  - precision_landing (container) → reads video10
  - rock_detection (HOST)         → reads video10

Run INSIDE Docker container.

Prerequisites (on HOST, one time):
  sudo modprobe v4l2loopback devices=1 video_nr=10 \
    card_label=CamBridge exclusive_caps=0

Usage:
  python3 camera_bridge.py
  python3 camera_bridge.py --src 0 --dst 10
"""

import cv2
import time
import argparse
import sys

try:
    import pyfakewebcam
except ImportError:
    print("[ERROR] Run: pip install pyfakewebcam")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

SRC_INDEX = 0    # real camera  → /dev/video0
DST_INDEX = 10   # virtual cam  → /dev/video10
CAM_W     = 1280
CAM_H     = 720
CAM_FPS   = 30


def main(src: int, dst: int):
    print(f"[Bridge] Opening source /dev/video{src}...")
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[Bridge] ERROR: Cannot open /dev/video{src}")
        sys.exit(1)

    # Set camera format
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.read()  # flush first frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Read one frame to get actual resolution
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[Bridge] ERROR: Cannot read from camera")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"[Bridge] Camera resolution: {w}x{h}")

    print(f"[Bridge] Opening destination /dev/video{dst}...")
    try:
        fake = pyfakewebcam.FakeWebcam(f"/dev/video{dst}", w, h)
    except Exception as e:
        print(f"[Bridge] ERROR: Cannot open /dev/video{dst}: {e}")
        print("  Make sure v4l2loopback is loaded with exclusive_caps=0")
        sys.exit(1)

    print(f"[Bridge] Bridge running: /dev/video{src} → /dev/video{dst}")
    print("[Bridge] Ctrl+C to stop\n")

    frame_count = 0
    fps_t       = time.monotonic()
    fps         = 0.0
    fail_count  = 0
    MAX_FAILS   = 30

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            fail_count += 1
            if fail_count >= MAX_FAILS:
                print(f"[Bridge] ERROR: {MAX_FAILS} consecutive read failures — exiting")
                break
            time.sleep(0.01)
            continue

        fail_count = 0

        # pyfakewebcam expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            fake.schedule_frame(rgb)
        except Exception as e:
            print(f"[Bridge] Write error: {e}")
            break

        frame_count += 1
        now = time.monotonic()
        if now - fps_t >= 3.0:
            fps = frame_count / (now - fps_t)
            print(f"[Bridge] FPS: {fps:.1f}  frames: {frame_count}", flush=True)
            frame_count = 0
            fps_t = now

    cap.release()
    print("[Bridge] Stopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, default=SRC_INDEX,
                    help="Source camera index (default 0)")
    ap.add_argument("--dst", type=int, default=DST_INDEX,
                    help="Destination v4l2loopback index (default 10)")
    args = ap.parse_args()
    try:
        main(args.src, args.dst)
    except KeyboardInterrupt:
        print("\n[Bridge] Stopped.")
