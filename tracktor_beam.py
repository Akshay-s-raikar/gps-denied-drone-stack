#!/usr/bin/env python3
"""
tracktor_beam.py — Precision Landing v6
=========================================
Changes from v5 (anti-oscillation fixes):
  - YAW_KP reduced 200→60, YAW_KD=0 (D term removed until P stable)
  - YAW_MAX_PWM reduced 250→120
  - Smoother alpha heavily reduced: x/y 0.6→0.25, yaw 0.5→0.15
  - RC_OVERRIDE_TIME enforced to 0.5 in check_params (was 3.0 in logs)
  - D term removed from send_yaw_rc_override (was computing noise)
  - Altitude source fixed: LOCAL_POSITION_NED z (EKF-fused) not GLOBAL_POSITION_INT
  - SEND_HZ reduced 20→15 to reduce override spam
  - Tag-lost hold reduced 1.0→0.5s to avoid driving on stale corrections
  - Added per-loop dt guard: skip send if dt < 40ms to prevent burst
  - Yaw PD re-enable notes in config
"""

import cv2
import numpy as np
import time
import math
import threading
import argparse
import sys
import csv
import os
from collections import deque
from datetime import datetime
from pymavlink import mavutil

try:
    import pupil_apriltags as apriltag
except ImportError:
    print("[ERROR] Run: pip install pupil-apriltags")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

TAG_FAMILY       = "tag36h11"
TAG_LARGE_ID     = 0
TAG_SMALL_ID     = 1
TAG_LARGE_SIZE_M = 0.40
TAG_SMALL_SIZE_M = 0.15

SWITCH_DOWN_M    = 1.3
SWITCH_UP_M      = 1.7
INVERT_ANGLE_Y   = True
INVERT_ANGLE_X   = True

INVERT_YAW       = False

# ── Yaw PD ───────────────────────────────────────────────────────
# Tuning ladder:
#   Step 1: fly with KP=60, KD=0, MAX_PWM=120
#   Step 2: if still oscillating → KP=30
#   Step 3: if sluggish → raise KP by +15 at a time
#   Step 4: once P stable → add KD=10, raise by +5 until D-kick appears
YAW_KP           = 60.0        # was 200.0 — start here
YAW_KD           = 0.0         # keep at 0 until P is stable
YAW_MAX_PWM      = 120         # was 250 — hard authority cap
YAW_DEADBAND_RAD = math.radians(5.0)   # was 10° — tighter

CAM_INDEX        = 10
CAM_W            = 1280
CAM_H            = 720
CAM_FPS          = 30
CAM_AUTO_EXP     = True
CAM_EXPOSURE     = 150

DEFAULT_DEVICE     = "udp:127.0.0.1:14551"
DEFAULT_BAUD       = 921600
HEARTBEAT_TIMEOUT  = 5.0
RECONNECT_COOLDOWN = 5.0

SEND_HZ              = 15          # was 20 — reduce override spam
SEND_MIN_DT          = 0.040       # guard: skip if <40ms since last send
CONFIRM_N            = 3
SEARCH_TIMEOUT       = 10.0
MIN_DECISION_MARGIN  = 15.0
TAG_LOST_HOLD_S      = 0.5         # was 1.0 — don't hold stale corrections long

SHOW_DISPLAY         = False
FONT                 = cv2.FONT_HERSHEY_SIMPLEX

# Active modes for precision landing
ACTIVE_MODES = ("LAND", "RTL")

# ═══════════════════════════════════════════════════════════════════
# THREADED CAMERA
# ═══════════════════════════════════════════════════════════════════

class ThreadedCamera:
    def __init__(self, index):
        self._frame  = None
        self._lock   = threading.Lock()
        self._ok     = False
        self._stop   = False
        self.width   = 0
        self.height  = 0
        self.fps_measured = 0.0

        print(f"[Camera] Opening /dev/video{index} (V4L2 + MJPEG)...")
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"[Camera] FAILED to open index {index}")
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M','J','P','G'))
        self.cap.read()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if CAM_AUTO_EXP:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            print("[Camera] Auto-exposure → V4L2 mode 3")
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, CAM_EXPOSURE)
            print(f"[Camera] Manual exposure = {CAM_EXPOSURE}")

        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[Camera] Cannot read frame — check USB connection")
            sys.exit(1)

        self.width  = frame.shape[1]
        self.height = frame.shape[0]
        print(f"[Camera] Resolution: {self.width}x{self.height}")

        if (self.width, self.height) != (CAM_W, CAM_H):
            print(f"[Camera] WARNING: wanted {CAM_W}x{CAM_H}, "
                  f"got {self.width}x{self.height}")

        b, g, r = cv2.split(frame)
        bm, gm, rm = b.mean(), g.mean(), r.mean()
        print(f"[Camera] Channel means B:{bm:.1f} G:{gm:.1f} R:{rm:.1f}")
        if rm > bm * 1.8:
            print("[Camera] WARNING: red-dominant frame — "
                  "possible YUYV→BGR conversion issue")

        t0, n = time.monotonic(), 0
        while time.monotonic() - t0 < 1.5:
            ok, _ = self.cap.read()
            if ok: n += 1
        self.fps_measured = n / 1.5
        label = "OK" if self.fps_measured >= 20 else "LOW"
        print(f"[Camera] Pre-thread FPS: {self.fps_measured:.1f} [{label}]")
        if self.fps_measured < 20:
            print("[Camera] Tip: check 'v4l2-ctl --list-formats-ext -d "
                  f"/dev/video{index}' — confirm MJPEG@{CAM_W}x{CAM_H}")

        self._frame = frame
        self._ok    = True
        t = threading.Thread(target=self._reader, daemon=True)
        t.start()
        print("[Camera] Background capture thread started ✓")

    def _reader(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            with self._lock:
                self._frame = frame
                self._ok    = True

    def read(self):
        with self._lock:
            return self._ok, (self._frame.copy()
                              if self._frame is not None else None)

    def release(self):
        self._stop = True
        time.sleep(0.1)
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════
# CAMERA PARAMS
# ═══════════════════════════════════════════════════════════════════

def load_camera_params(w, h):
    calib = "camera_calib.npz"
    if os.path.exists(calib):
        d  = np.load(calib)
        K  = d["camera_matrix"]
        fx = float(K[0,0]); fy = float(K[1,1])
        cx = float(K[0,2]); cy = float(K[1,2])
        print(f"[Camera] Calibration loaded ✓ fx={fx:.1f}")
        return fx, fy, cx, cy, d["dist_coeffs"]
    fov = math.radians(70.0)
    fx  = (w / 2.0) / math.tan(fov / 2.0)
    print(f"[Camera] Estimated fx={fx:.1f} (run calibrate_camera.py)")
    return fx, fx, w / 2.0, h / 2.0, np.zeros((5, 1))


# ═══════════════════════════════════════════════════════════════════
# SMOOTHER
# ═══════════════════════════════════════════════════════════════════

class Smoother:
    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.val   = None

    def update(self, v):
        self.val = v if self.val is None else (
            self.alpha * v + (1 - self.alpha) * self.val)
        return self.val

    def reset(self):
        self.val = None


# ═══════════════════════════════════════════════════════════════════
# MAVLINK
# ═══════════════════════════════════════════════════════════════════

class MAVLink:
    def __init__(self, device, baud):
        self.device          = device
        self.baud            = baud
        self.master          = None
        self.altitude        = 2.0
        self.mode            = "UNKNOWN"
        self.armed           = False
        self._lock           = threading.Lock()
        self._running        = False
        self._last_hb        = 0.0
        self._last_reconnect = 0.0

    def connect(self):
        print(f"[MAVLink] Connecting {self.device}...")
        try:
            self.master = mavutil.mavlink_connection(
                self.device, baud=self.baud)
            self.master.wait_heartbeat(timeout=10)
            self._last_hb = time.monotonic()
            print(f"[MAVLink] Connected ✓ sys={self.master.target_system}")
        except Exception as e:
            print(f"[MAVLink] FAILED: {e}")
            sys.exit(1)

        self.master.mav.request_data_stream_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

        self._running = True
        threading.Thread(target=self._read, daemon=True).start()
        print("[MAVLink] Telemetry thread started")

    def _try_reconnect(self):
        now = time.monotonic()
        if now - self._last_reconnect < RECONNECT_COOLDOWN:
            return
        self._last_reconnect = now
        print("\n[MAVLink] Heartbeat lost — reconnecting...")
        try:
            nm = mavutil.mavlink_connection(self.device, baud=self.baud)
            nm.wait_heartbeat(timeout=5)
            with self._lock:
                self.master   = nm
                self._last_hb = time.monotonic()
            self.master.mav.request_data_stream_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)
            print("[MAVLink] Reconnected ✓")
        except Exception as e:
            print(f"[MAVLink] Reconnect failed: {e}")

    def _read(self):
        mode_map = {
            0:"STABILIZE", 2:"ALT_HOLD", 3:"AUTO",
            4:"GUIDED",    5:"LOITER",   6:"RTL",
            9:"LAND",      16:"POSHOLD",
        }
        while self._running:
            try:
                if time.monotonic() - self._last_hb > HEARTBEAT_TIMEOUT:
                    self._try_reconnect()
                msg = self.master.recv_match(blocking=True, timeout=1.0)
                if not msg: continue
                t = msg.get_type()
                if t == "HEARTBEAT":
                    self._last_hb = time.monotonic()
                    with self._lock:
                        self.armed = bool(
                            msg.base_mode &
                            mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        self.mode = mode_map.get(
                            msg.custom_mode, f"MODE_{msg.custom_mode}")
                elif t == "LOCAL_POSITION_NED":
                    # Use EKF-fused altitude (z is down in NED, negate for alt)
                    with self._lock:
                        self.altitude = max(0.0, -float(msg.z))
            except Exception:
                pass

    def get_alt(self):
        with self._lock: return self.altitude

    def get_mode(self):
        with self._lock: return self.mode

    def is_armed(self):
        with self._lock: return self.armed

    def check_params(self):
        print("[Params] Checking ArduPilot PLND params...")
        checks = [
            ("PLND_ENABLED",     1,   "Precision landing enabled"),
            ("PLND_TYPE",        1,   "Companion computer backend"),
            ("RC_OVERRIDE_TIME", 0.5, "RC override timeout — MUST be 0.5, not 3.0"),
        ]
        for param, expected, desc in checks:
            try:
                self.master.mav.param_request_read_send(
                    self.master.target_system,
                    self.master.target_component,
                    param.encode(), -1)
                t0  = time.monotonic()
                got = False
                while time.monotonic() - t0 < 5.0:
                    msg = self.master.recv_match(
                        type="PARAM_VALUE", blocking=True, timeout=1.0)
                    if not msg: continue
                    name = msg.param_id
                    if isinstance(name, (bytes, bytearray)):
                        name = name.decode(errors="ignore")
                    name = name.rstrip("\x00")
                    if name != param: continue
                    val = msg.param_value
                    ok  = abs(val - expected) < 0.05
                    status = "✓" if ok else f"✗ SHOULD BE {expected}"
                    print(f"  {param}={val:.3f} {status} — {desc}")
                    if param == "RC_OVERRIDE_TIME" and val > 1.0:
                        print(f"  !! RC_OVERRIDE_TIME={val:.1f} is too high — "
                              f"causes oscillation.")
                        print(f"     Set RC_OVERRIDE_TIME=0.5 in Mission Planner NOW.")
                    got = True
                    break
                if not got:
                    print(f"  {param}=TIMEOUT")
            except Exception as e:
                print(f"  {param}=SKIPPED ({e})")

    def send_landing_target(self, angle_x, angle_y, distance):
        if not self.master: return
        time_us = int(time.monotonic() * 1e6)
        with self._lock:
            try:
                self.master.mav.landing_target_send(
                    time_us, 0,
                    mavutil.mavlink.MAV_FRAME_BODY_FRD,
                    float(angle_x), float(angle_y),
                    float(distance), 0.0, 0.0)
            except Exception:
                pass

    def send_yaw_rc_override(self, yaw_error_rad):
        """
        P-only yaw correction via RC channel 4 override.
        D term intentionally removed — was computing noise from jittery
        wall-clock dt. Re-enable only after P is stable:
            output = YAW_KP * yaw_error_rad + YAW_KD * yaw_rate
        where yaw_rate should come from ATTITUDE msg, not error delta.
        """
        if not self.master: return

        if abs(yaw_error_rad) < YAW_DEADBAND_RAD:
            pwm = 65535  # release channel — let ArduPilot hold yaw
        else:
            sign   = -1 if INVERT_YAW else 1
            output = YAW_KP * yaw_error_rad   # P only
            offset = int(sign * output)
            offset = max(-YAW_MAX_PWM, min(YAW_MAX_PWM, offset))
            pwm    = 1500 + offset

        with self._lock:
            try:
                self.master.mav.rc_channels_override_send(
                    self.master.target_system,
                    self.master.target_component,
                    65535, 65535, 65535, pwm,
                    65535, 65535, 65535, 65535)
            except Exception:
                pass

    def release_yaw_override(self):
        if not self.master: return
        with self._lock:
            try:
                self.master.mav.rc_channels_override_send(
                    self.master.target_system,
                    self.master.target_component,
                    65535, 65535, 65535, 65535,
                    65535, 65535, 65535, 65535)
            except Exception:
                pass

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════
# APRILTAG DETECTOR
# ═══════════════════════════════════════════════════════════════════

class AprilTagDetector:
    def __init__(self, fx, fy, cx, cy, D):
        self.fx = fx; self.fy = fy
        self.cx = cx; self.cy = cy
        self.D  = D
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._current_tag = TAG_LARGE_ID
        self.detector = apriltag.Detector(
            families          = TAG_FAMILY,
            nthreads          = 4,
            quad_decimate     = 2.0,
            quad_sigma        = 0.8,
            refine_edges      = True,
            decode_sharpening = 0.25,
        )
        print(f"[AprilTag] {TAG_FAMILY} ✓ (quad_decimate=2.0 for speed)")

    def _select_tag(self, altitude):
        if self._current_tag == TAG_LARGE_ID and altitude < SWITCH_DOWN_M:
            self._current_tag = TAG_SMALL_ID
            print(f"\n[Tag] → SMALL (alt={altitude:.2f}m)")
        elif self._current_tag == TAG_SMALL_ID and altitude > SWITCH_UP_M:
            self._current_tag = TAG_LARGE_ID
            print(f"\n[Tag] → LARGE (alt={altitude:.2f}m)")
        size = TAG_LARGE_SIZE_M if self._current_tag == TAG_LARGE_ID else TAG_SMALL_SIZE_M
        return self._current_tag, size

    def detect(self, frame, altitude=2.0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)

        target_id, target_size = self._select_tag(altitude)
        dets = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(self.fx, self.fy, self.cx, self.cy),
            tag_size=target_size,
        )

        best = fallback = None
        for d in dets:
            if d.decision_margin < MIN_DECISION_MARGIN: continue
            if d.tag_id == target_id:
                if best is None or d.decision_margin > best.decision_margin:
                    best = d
            elif d.tag_id in (TAG_LARGE_ID, TAG_SMALL_ID):
                if fallback is None or d.decision_margin > fallback.decision_margin:
                    fallback = d

        chosen     = best if best is not None else fallback
        is_primary = best is not None
        if chosen is None: return None

        if not is_primary:
            chosen_size = TAG_SMALL_SIZE_M if target_id == TAG_LARGE_ID else TAG_LARGE_SIZE_M
            redets = self.detector.detect(
                gray, estimate_tag_pose=True,
                camera_params=(self.fx, self.fy, self.cx, self.cy),
                tag_size=chosen_size)
            for rd in redets:
                if rd.tag_id == chosen.tag_id:
                    chosen = rd; break
        else:
            chosen_size = target_size

        cx_tag  = float(chosen.center[0])
        cy_tag  = float(chosen.center[1])
        angle_x = math.atan2(cx_tag - self.cx, self.fx)
        angle_y = math.atan2(cy_tag - self.cy, self.fy)
        if INVERT_ANGLE_Y: angle_y = -angle_y
        if INVERT_ANGLE_X: angle_x = -angle_x

        yaw_error = 0.0
        if chosen.pose_R is not None:
            R = chosen.pose_R
            yaw_error = math.atan2(float(R[1][0]), float(R[0][0]))
            if yaw_error > math.pi / 2:
                yaw_error -= math.pi
            elif yaw_error < -math.pi / 2:
                yaw_error += math.pi
            yaw_error = max(-math.pi/2, min(math.pi/2, yaw_error))

        if chosen.pose_t is not None:
            distance = float(abs(chosen.pose_t[2][0]))
            distance = max(0.1, min(distance, 20.0))
        else:
            corners = chosen.corners
            side_px = float(np.mean([
                np.linalg.norm(corners[i] - corners[(i+1)%4])
                for i in range(4)]))
            distance = (chosen_size * self.fx) / max(side_px, 1.0)

        return {
            "tag_id":          chosen.tag_id,
            "angle_x":         angle_x,
            "angle_y":         angle_y,
            "yaw_error":       yaw_error,
            "distance":        distance,
            "center":          (cx_tag, cy_tag),
            "corners":         chosen.corners,
            "marker_size":     chosen_size,
            "decision_margin": chosen.decision_margin,
            "is_primary":      is_primary,
        }


# ═══════════════════════════════════════════════════════════════════
# LOGGER
# ═══════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"logs/landing_{ts}.csv"
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["time_s","state","alt_m","mode",
                         "tag_id","ax_deg","ay_deg","yaw_err_deg",
                         "dist_m","margin","fps"])
        print(f"[Log] {path}")

    def log(self, state, alt, mode, tag_id, ax, ay, yaw_err, dist, margin, fps):
        self.w.writerow([f"{time.monotonic():.3f}", state, f"{alt:.3f}",
                         mode, tag_id, f"{math.degrees(ax):.3f}",
                         f"{math.degrees(ay):.3f}",
                         f"{math.degrees(yaw_err):.3f}",
                         f"{dist:.3f}", f"{margin:.1f}", f"{fps:.1f}"])
        self.f.flush()

    def close(self): self.f.close()


# ═══════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════

def draw(frame, det, state, alt, mode, ax_s, ay_s, yaw_s,
         n_confirm, fps, armed):
    H, W = frame.shape[:2]
    out  = frame.copy()
    cx, cy = W//2, H//2

    cv2.line(out, (cx-30,cy),(cx+30,cy),(80,80,80),1)
    cv2.line(out, (cx,cy-30),(cx,cy+30),(80,80,80),1)

    if det:
        pts = det["corners"].astype(int)
        col = (0,255,0) if det["is_primary"] else (0,165,255)
        for j in range(4):
            cv2.line(out, tuple(pts[j]), tuple(pts[(j+1)%4]), col, 2)
        mx, my = int(det["center"][0]), int(det["center"][1])
        cv2.circle(out, (mx,my), 8, col, -1)
        cv2.line(out, (cx,cy), (mx,my), (0,200,255), 1)
        ddx = int((mx-cx)*0.4); ddy = int((my-cy)*0.4)
        if abs(ddx)+abs(ddy) > 8:
            cv2.arrowedLine(out,(cx,cy),(cx+ddx,cy+ddy),
                            (0,165,255),2,tipLength=0.3)
        yaw_deg = math.degrees(yaw_s)
        yaw_col = (0,255,255) if abs(yaw_deg) < 5 else (0,100,255)
        cv2.ellipse(out, (mx,my), (25,25), 0,
                    -90, -90 + int(yaw_deg), yaw_col, 3)
        tn = f"LARGE(ID{TAG_LARGE_ID})" if det["tag_id"]==TAG_LARGE_ID \
             else f"SMALL(ID{TAG_SMALL_ID})"
        yaw_aligned = abs(yaw_deg) < 5.0
        lines = [
            f"Tag: {tn} [{'PRIMARY' if det['is_primary'] else 'FALLBACK'}]",
            f"Size: {det['marker_size']*100:.0f}cm",
            f"Dist(Z): {det['distance']:.2f}m",
            f"Ax: {math.degrees(det['angle_x']):+.1f}°"
            f"{' (inv)' if INVERT_ANGLE_X else ''}",
            f"Ay: {math.degrees(det['angle_y']):+.1f}°"
            f"{' (inv)' if INVERT_ANGLE_Y else ''}",
            f"Yaw err: {yaw_deg:+.1f}° "
            f"{'[ALIGNED]' if yaw_aligned else '[ROTATING]'}"
            f"{' (inv)' if INVERT_YAW else ''}",
            f"Smooth Ax: {math.degrees(ax_s):+.1f}",
            f"Smooth Ay: {math.degrees(ay_s):+.1f}",
            f"Margin: {det['decision_margin']:.1f}",
            f"Confirm: {n_confirm}/{CONFIRM_N}",
        ]
        y0 = H - 10 - len(lines)*20
        for k, ln in enumerate(lines):
            ln_col = (0,255,255) if (k == 5 and yaw_aligned) else col
            cv2.putText(out, ln, (10, y0+k*20),
                        FONT, 0.42, ln_col, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0,0), (W,70), (10,10,20),-1)
    sc = {"SEARCHING":(0,165,255),"CONFIRMING":(0,255,255),
          "TRACKING":(0,255,0),"NO_MARKER":(80,80,80)}.get(state,(200,200,200))
    cv2.putText(out, f"STATE: {state}", (10,33),
                FONT, 0.82, sc, 2, cv2.LINE_AA)
    inv_str = " YAW-INV" if INVERT_YAW else ""
    cv2.putText(out,
                f"MODE:{mode}  FPS:{fps:.1f}  "
                f"ALT:{alt:.1f}m  "
                f"{'ARMED' if armed else 'DISARMED'}{inv_str}",
                (10,58), FONT, 0.38, (160,160,160), 1, cv2.LINE_AA)
    if state == "TRACKING":
        cv2.circle(out, (W-20,33), 12, (0,255,0), -1)
        cv2.putText(out, "SENDING", (W-95,55),
                    FONT, 0.38, (0,255,0), 1, cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def run(args):
    print("\n" + "="*60)
    print("  Precision Landing v6 — Anti-Oscillation")
    print("  KP=60 KD=0 MAX_PWM=120 | Smoother alpha=0.15/0.25")
    print("  Altitude: LOCAL_POSITION_NED (EKF-fused, GPS-denied safe)")
    print("="*60)
    print(f"[Yaw] INVERT_YAW={INVERT_YAW}  KP={YAW_KP}  KD={YAW_KD}")
    print(f"[Yaw] MAX_PWM={YAW_MAX_PWM}  DEADBAND={math.degrees(YAW_DEADBAND_RAD):.0f}deg")
    print(f"[Yaw] If drone spins wrong way: --invert-yaw flag or set INVERT_YAW=True")
    print(f"[CRITICAL] RC_OVERRIDE_TIME must be 0.5 in Mission Planner (not 3.0)\n")

    cam = ThreadedCamera(args.cam)
    fx, fy, cx, cy, D = load_camera_params(cam.width, cam.height)

    mav = MAVLink(args.device, args.baud)
    mav.connect()
    time.sleep(1)
    mav.check_params()

    detector   = AprilTagDetector(fx, fy, cx, cy, D)
    smooth_x   = Smoother(alpha=0.25)   # was 0.6 — much slower
    smooth_y   = Smoother(alpha=0.25)   # was 0.6
    smooth_yaw = Smoother(alpha=0.15)   # was 0.5 — very slow for yaw
    logger     = Logger()

    if SHOW_DISPLAY:
        cv2.namedWindow("Precision Landing", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Precision Landing", 1280, 720)

    state         = "SEARCHING"
    confirm_buf   = []
    search_start  = time.monotonic()
    send_iv       = 1.0 / SEND_HZ
    last_send     = 0.0
    ax_s = ay_s   = 0.0
    yaw_s         = 0.0
    last_det      = None
    lost_t        = None
    ss_n          = 0
    fps_times     = []
    fps_d         = 0.0
    last_yaw_send = 0.0

    print(f"[Config] SEND_HZ={SEND_HZ}  SEND_MIN_DT={SEND_MIN_DT*1000:.0f}ms")
    print(f"[Config] TAG_LOST_HOLD={TAG_LOST_HOLD_S}s  CONFIRM_N={CONFIRM_N}")
    print(f"[Config] Mode gate: LAND + RTL")
    print("[Controls] Q=quit  S=screenshot\n")

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.001)
            continue

        now = time.monotonic()
        fps_times.append(now)
        fps_times = [t for t in fps_times if now - t < 1.0]
        fps_d     = float(len(fps_times))

        alt   = mav.get_alt()
        mode  = mav.get_mode()
        armed = mav.is_armed()

        det = detector.detect(frame, alt)

        if det:
            if state == "SEARCHING":
                confirm_buf  = []
                state        = "CONFIRMING"
                search_start = now

            if state in ("CONFIRMING", "TRACKING"):
                confirm_buf.append(det)
                if len(confirm_buf) > CONFIRM_N * 2:
                    confirm_buf = confirm_buf[-CONFIRM_N:]
                if len(confirm_buf) >= CONFIRM_N:
                    state = "TRACKING"
                ax_s  = smooth_x.update(det["angle_x"])
                ay_s  = smooth_y.update(det["angle_y"])
                yaw_s = smooth_yaw.update(det["yaw_error"])
                last_det = det
                lost_t   = None

            # Rate-limit sends: interval check + minimum dt guard
            dt_since_send = now - last_send
            if (state == "TRACKING"
                    and dt_since_send >= send_iv
                    and dt_since_send >= SEND_MIN_DT):
                if mode in ACTIVE_MODES:
                    mav.send_landing_target(ax_s, ay_s, det["distance"])
                    mav.send_yaw_rc_override(yaw_s)
                    last_yaw_send = now
                else:
                    mav.release_yaw_override()
                    last_yaw_send = 0.0
                last_send = now
                print(f"\r[TRACK] alt={alt:.2f}m dist={det['distance']:.2f}m "
                      f"ax={math.degrees(ax_s):+.1f}° "
                      f"ay={math.degrees(ay_s):+.1f}° "
                      f"yaw={math.degrees(yaw_s):+.1f}° "
                      f"fps={fps_d:.0f} mode={mode}  ",
                      end="", flush=True)

            # Release yaw if mode changed away from ACTIVE_MODES
            if last_yaw_send > 0 and mode not in ACTIVE_MODES:
                mav.release_yaw_override()
                last_yaw_send = 0.0

            logger.log(state, alt, mode, det["tag_id"],
                       det["angle_x"], det["angle_y"], det["yaw_error"],
                       det["distance"], det["decision_margin"], fps_d)

        else:
            if state == "TRACKING":
                if lost_t is None:
                    lost_t = now
                    print(f"\n[WARN] Tag lost — holding {TAG_LOST_HOLD_S}s")
                elif now - lost_t < TAG_LOST_HOLD_S:
                    # Hold XY with last known — but release yaw immediately
                    dt_since_send = now - last_send
                    if (last_det
                            and dt_since_send >= send_iv
                            and dt_since_send >= SEND_MIN_DT):
                        if mode in ACTIVE_MODES:
                            mav.send_landing_target(ax_s, ay_s, last_det["distance"])
                        mav.release_yaw_override()
                        last_yaw_send = 0.0
                        last_send = now
                else:
                    print("\n[WARN] Tag lost → SEARCHING")
                    state        = "SEARCHING"
                    confirm_buf  = []
                    last_det     = None
                    lost_t       = None
                    search_start = now
                    smooth_x.reset()
                    smooth_y.reset()
                    smooth_yaw.reset()
                    yaw_s = 0.0
                    mav.release_yaw_override()
                    last_yaw_send = 0.0

            elif state == "CONFIRMING":
                state        = "SEARCHING"
                confirm_buf  = []
                search_start = now

            if state == "SEARCHING" and now - search_start > SEARCH_TIMEOUT:
                print(f"\n[WARN] No tag {SEARCH_TIMEOUT:.0f}s — GPS landing")
                search_start = now

            logger.log("SEARCHING", alt, mode, -1, 0, 0, 0, 0, 0, fps_d)

        if SHOW_DISPLAY:
            disp_state = state if det else "NO_MARKER"
            disp = draw(frame, det, disp_state, alt, mode,
                        ax_s, ay_s, yaw_s, len(confirm_buf), fps_d, armed)
            cv2.imshow("Precision Landing", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[Quit]"); break
            elif key == ord('s'):
                fn = f"landing_ss_{ss_n:03d}.jpg"
                cv2.imwrite(fn, disp); print(f"\n[SS] {fn}")
                ss_n += 1

    mav.release_yaw_override()
    logger.close()
    mav.stop()
    cam.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    print("\n[Done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Precision Landing v6")
    ap.add_argument("--device",     type=str, default=DEFAULT_DEVICE)
    ap.add_argument("--baud",       type=int, default=DEFAULT_BAUD)
    ap.add_argument("--cam",        type=int, default=CAM_INDEX)
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--manual-exp", action="store_true")
    ap.add_argument("--invert-yaw", action="store_true")
    ap.add_argument("--kp",         type=float, default=None,
                    help="Override YAW_KP (default 60)")
    ap.add_argument("--max-pwm",    type=int,   default=None,
                    help="Override YAW_MAX_PWM (default 120)")
    args = ap.parse_args()

    if args.no_display:  SHOW_DISPLAY = False
    if args.manual_exp:  CAM_AUTO_EXP = False
    if args.invert_yaw:  INVERT_YAW   = True
    if args.kp is not None:      YAW_KP      = args.kp
    if args.max_pwm is not None: YAW_MAX_PWM = args.max_pwm

    run(args)