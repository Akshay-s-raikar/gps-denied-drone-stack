# Autonomous GPS-Denied Drone Stack — IRoC-U 2026

**Team Drone Aacharya (Team 10278)** — an autonomous quadcopter that searches a GPS-denied arena, detects target rocks using a rotation-invariant vision pipeline, and performs AprilTag-based precision landing with active yaw correction.

[![Watch the demo](thumbnail.jpg)](https://drive.google.com/file/d/1gg_JRKSxSIIwGsoBJ_Ruk7hK_8Kna9sr/view?usp=sharing)
*(Click to watch the full autonomous arena mission flight)*

---

## Overview

This repo contains the full autonomous mission stack flown for IRoC-U 2026: search-pattern navigation, real-time rock detection, precision landing, and the supporting camera/telemetry infrastructure that ties it together on a Jetson + Pixhawk setup.

**Key result:** completed a full autonomous arena mission — armed in GUIDED, flew a lawnmower search pattern, detected and logged target rocks in real time, and landed autonomously on an AprilTag using active yaw alignment.

---

## Hardware

| Component | Spec |
|---|---|
| Frame | F450 quadcopter |
| Flight controller | Pixhawk 4 |
| Companion computer | NVIDIA Jetson Orin NX 16GB |
| Camera | USB camera (v4l2loopback-shared, 1280×720 @30fps) |
| Telemetry link | Serial, 921600 baud |

## Software Stack

- **Flight stack:** ArduCopter, MAVROS (ROS2), MAVProxy (serial → UDP bridge)
- **Detection:** YOLOv8 (`yolov8n`) + DINOv2 (`dinov2_vits14`) rotation-invariant similarity matching
- **Precision landing:** AprilTag (`tag36h11`) detection with dual-tag altitude switching + PD yaw control
- **Telemetry:** ZeroMQ (PUB/SUB) + MessagePack, UDP bridges between containerized and host processes

---

## What's in this repo

### `rock_detector_v6.py` — Rock detection pipeline
YOLO proposes candidate boxes on each frame → each crop is embedded with DINOv2 at multiple rotation angles (12 inference angles vs. 24 seed angles) and matched against a reference "seed bank" by cosine similarity, taking the **max similarity across all rotations** — this makes matching robust to the rock's orientation in frame, without needing a rotation-specific model. On a confirmed match it saves the full frame, reads the drone's live X/Y/Z from a UDP position feed, and sends a (reference, detected) image pair over ZMQ to the ground station, respecting a global cooldown so the same object isn't reported repeatedly.

### `tracktor_beam.py` — Precision landing (v6, anti-oscillation)
AprilTag-based precision landing with a large tag (0.40m) for far-range detection and a small tag (0.15m) that takes over below 1.3m altitude for close-range accuracy. Computes X/Y/yaw corrections via a smoothed PD controller and sends them to ArduPilot as landing-target and yaw RC-override MAVLink messages, gated to only apply during LAND/RTL modes. The file's own version history documents a real oscillation-tuning process — yaw gain dropped from 200→60, derivative term removed until proportional control was stable, and smoothing significantly increased — the kind of iterative flight tuning that doesn't show up unless you've actually flown it.

### `arena_web_v2.py` — Autonomous search pattern
MAVROS/ROS2 node that auto-generates a lawnmower search pattern over the arena and flies it using position setpoints. Uses separate step sizes for forward legs (1.0m) vs. sideways shifts near the walls (0.3m, deliberately smaller/slower for safety), waits for EKF position lock before arming, and falls back to RTL on waypoint timeout or MAVLink signal loss.

### `set_ekf_origin.py` — GPS-denied EKF origin setup
Publishes an EKF origin so ArduPilot can compute RTL without a GPS fix — required once per session before arming, since RTL always returns to the arm location, not the EKF origin coordinates themselves.

### `camera_bridge.py` — Shared camera access
Reads the real camera (`/dev/video0`) and re-publishes it to a v4l2loopback virtual device (`/dev/video10`) so both the precision-landing process (inside Docker) and the rock-detection process (on host) can read the same physical camera simultaneously.

### `telem_sender.py` — Host-side telemetry relay
Receives telemetry forwarded from inside the Docker container, republishes it over ZMQ (MessagePack-encoded) to the ground control station, and separately forwards live X/Y/Z position over UDP for the detection pipeline to consume.

### `start_arena.py` — Mission launcher
Single script that starts the full stack in order: MAVProxy → EKF origin/GPS setup → camera bridge → precision landing → arena search, with cleanup of leftover processes on start and a graceful LAND-then-kill-all on Ctrl+C.

---

## Repo Layout

```
├── detection/
│   └── rock_detector_v6.py
├── navigation/
│   ├── tracktor_beam.py       # precision landing
│   ├── arena_web_v2.py         # search pattern
│   └── set_ekf_origin.py
├── infra/
│   ├── camera_bridge.py
│   ├── telem_sender.py
│   └── start_arena.py
├── docs/
│   └── images, flight footage links
└── README.md
```

---

## Demo

- 🎥 Full autonomous arena mission: [https://drive.google.com/file/d/1gg_JRKSxSIIwGsoBJ_Ruk7hK_8Kna9sr/view?usp=sharing]
- 📷 Hardware photos: `docs/hardware.jpg`

---

## Team

Team Drone Aacharya (Team 10278) — IRoC-U 2026
