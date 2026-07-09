#!/usr/bin/env python3
"""
start_arena.py — IRoC-U 2026 Mission Launcher
===============================================
Launch order:
  1. Cleanup
  2. MAVProxy (serial → UDP 14550 MAVROS + UDP 14551 landing)
  3. gps.py (EKF origin, visible)
  4. camera_bridge.py (video0 → video10, background)
  5. new_prisision_landing.py (background, cam 10, pymavlink UDP:14551)
  6. arena_web.py (foreground, MAVROS ROS2, no camera)

Camera sharing:
  camera_bridge.py → reads video0 → writes video10
  precision_landing → cam 10 (container)
  rock_detection    → cam 10 (HOST, run separately)

Run separately in other terminals:
  python3 ~/drone_missions/telemetry_data_send.py
  python3 ~/drone_missions/post_landing_sender.py

Ctrl+C → LAND → kill all
"""

import subprocess
import sys
import os
import time
import signal

SCRIPTS_DIR  = os.path.expanduser("~/drone_missions")
MAVPROXY_BIN = os.path.expanduser("~/.local/bin/mavproxy.py")
MAVLINK_DEV  = "/dev/ttyUSB0"
MAVLINK_BAUD = 921600
UDP_MAVROS   = "127.0.0.1:14550"
UDP_LANDING  = "127.0.0.1:14551"

processes     = []
mavproxy_proc = None
_stopping     = False
DEVNULL       = open(os.devnull, 'w')


def cleanup():
    print("[CLEAN] Killing leftover processes...")
    for pat in ["mavproxy", "arena_web", "arena_no_cam",
                "new_prisision_landing", "camera_bridge",
                "telemetry_data_send", "post_landing_sender", "gps.py"]:
        subprocess.call(["pkill", "-f", pat],
                        stdout=DEVNULL, stderr=DEVNULL)
    time.sleep(1)
    print("[CLEAN] Freeing camera...")
    for dev in ["/dev/video0", "/dev/video10"]:
        if os.path.exists(dev):
            subprocess.call(["sudo", "fuser", "-k", dev],
                            stdout=DEVNULL, stderr=DEVNULL)
    time.sleep(0.5)
    print("[CLEAN] Done ✓\n")


def send_land():
    try:
        print("[STOP] Sending LAND...")
        from pymavlink import mavutil
        m = mavutil.mavlink_connection("udp:127.0.0.1:14550",
                                       source_system=255)
        m.mav.command_long_send(
            1, 1,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            9, 0, 0, 0, 0, 0)
        time.sleep(1.5)
        m.close()
        print("[STOP] LAND sent ✓")
    except Exception as e:
        print(f"[STOP] Could not send LAND: {e}")


def signal_handler(sig, frame):
    global _stopping
    if _stopping:
        print("\n[STOP] Force killing...")
        os.system("pkill -9 -f mavproxy       2>/dev/null")
        os.system("pkill -9 -f arena_web      2>/dev/null")
        os.system("pkill -9 -f new_prisision  2>/dev/null")
        os.system("pkill -9 -f camera_bridge  2>/dev/null")
        os.system("pkill -9 -f gps            2>/dev/null")
        os.system("sudo fuser -k /dev/video0  2>/dev/null")
        os.system("sudo fuser -k /dev/video10 2>/dev/null")
        DEVNULL.close()
        sys.exit(1)

    _stopping = True
    print("\n" + "="*45)
    print("[STOP] Ctrl+C — shutting down")
    print("       Press Ctrl+C again to force kill")
    print("="*45)

    send_land()

    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"  killing {name}...")
            proc.kill()
            try: proc.wait(timeout=2)
            except subprocess.TimeoutExpired: pass

    if mavproxy_proc and mavproxy_proc.poll() is None:
        print("  killing MAVProxy...")
        mavproxy_proc.kill()
        try: mavproxy_proc.wait(timeout=2)
        except subprocess.TimeoutExpired: pass

    os.system("pkill -f mavproxy           2>/dev/null")
    os.system("pkill -f arena_web          2>/dev/null")
    os.system("pkill -f arena_no_cam       2>/dev/null")
    os.system("pkill -f new_prisision      2>/dev/null")
    os.system("pkill -f camera_bridge      2>/dev/null")
    os.system("pkill -f telemetry_data_send 2>/dev/null")
    os.system("pkill -f post_landing_sender 2>/dev/null")
    os.system("pkill -f gps               2>/dev/null")
    os.system("sudo fuser -k /dev/video0  2>/dev/null")
    os.system("sudo fuser -k /dev/video10 2>/dev/null")

    DEVNULL.close()
    print("[STOP] All stopped ✓")
    sys.exit(0)


signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def start_mavproxy():
    global mavproxy_proc
    print("[1/5] Starting MAVProxy...")
    if not os.path.exists(MAVPROXY_BIN):
        print(f"[ERROR] mavproxy not found: {MAVPROXY_BIN}")
        sys.exit(1)

    mavproxy_proc = subprocess.Popen([
        MAVPROXY_BIN,
        f"--master={MAVLINK_DEV}",
        f"--baudrate={MAVLINK_BAUD}",
        f"--out=udpout:{UDP_MAVROS}",
        f"--out=udpout:{UDP_LANDING}",
        "--daemon",
    ], stdout=DEVNULL, stderr=DEVNULL)

    time.sleep(4)
    if mavproxy_proc.poll() is not None:
        print("[ERROR] MAVProxy exited — check /dev/ttyUSB0")
        sys.exit(1)

    print("[1/5] MAVProxy running ✓")
    print(f"       {UDP_MAVROS}  → MAVROS (ROS2)")
    print(f"       {UDP_LANDING} → precision landing (pymavlink)")


def run_silent(script, extra_args=None):
    path = os.path.join(SCRIPTS_DIR, script)
    if not os.path.exists(path):
        print(f"  WARNING: {script} not found — skipping")
        return None
    proc = subprocess.Popen(
        [sys.executable, path] + (extra_args or []),
        cwd=SCRIPTS_DIR, stdout=DEVNULL, stderr=DEVNULL)
    processes.append((script, proc))
    return proc


def run_visible(script, extra_args=None):
    path = os.path.join(SCRIPTS_DIR, script)
    if not os.path.exists(path):
        print(f"  WARNING: {script} not found — skipping")
        return None
    proc = subprocess.Popen(
        [sys.executable, path] + (extra_args or []),
        cwd=SCRIPTS_DIR)
    processes.append((script, proc))
    return proc


def main():
    print("\n" + "="*55)
    print("  IRoC-U 2026 — Mission Launcher")
    print("="*55)
    print("  camera_bridge       → video0 → video10")
    print("  precision_landing   → cam 10, UDP:14551")
    print("  arena_web           → MAVROS ROS2, no camera")
    print()
    print("  Run separately:")
    print("    python3 telemetry_data_send.py")
    print("    python3 post_landing_sender.py")
    print("    python3 newrock_with_send.py --cam 10 (HOST)")
    print("="*55 + "\n")

    cleanup()

    # Step 1 — MAVProxy
    start_mavproxy()

    # Step 2 — EKF origin
    print("[2/5] Setting EKF origin...")
    proc = run_visible("gps.py")
    if proc:
        try:
            proc.wait(timeout=35)
            print("[2/5] EKF origin set ✓")
        except subprocess.TimeoutExpired:
            print("[2/5] gps.py timeout — continuing anyway")
            proc.kill()
    time.sleep(1)

    # Step 3 — camera bridge (video0 → video10)
    print("[3/5] Starting camera bridge (video0 → video10)...")
    if not os.path.exists("/dev/video10"):
        print("  WARNING: /dev/video10 not found")
        print("  Run on HOST: sudo modprobe v4l2loopback "
              "devices=1 video_nr=10 card_label=CamBridge exclusive_caps=0")
        landing_cam = "0"
    else:
        proc = run_silent("camera_bridge.py",
                          extra_args=["--src", "0", "--dst", "10"])
        time.sleep(2)
        if proc and proc.poll() is not None:
            print("[3/5] WARNING: camera bridge failed — landing uses cam 0")
            landing_cam = "0"
        else:
            print("[3/5] Camera bridge running ✓ (video0 → video10)")
            landing_cam = "10"

    # Step 4 — precision landing (background)
    print(f"[4/5] Starting precision landing (cam {landing_cam}, UDP:14551)...")
    proc = run_silent("new_prisision_landing.py",
                      extra_args=["--device", "udp:127.0.0.1:14551",
                                  "--cam",    landing_cam])
    time.sleep(3)
    if proc and proc.poll() is not None:
        print("[4/5] WARNING: precision landing crashed — check manually:")
        print(f"      python3 ~/drone_missions/new_prisision_landing.py "
              f"--device udp:127.0.0.1:14551 --cam {landing_cam}")
    else:
        print(f"[4/5] Precision landing running ✓ (cam {landing_cam})")

    # Step 5 — arena mission (foreground)
    print("[5/5] Starting arena mission (MAVROS, no camera)...\n")
    print("="*55)
    print("  MAVROS must be running:")
    print("  ros2 launch /home/jetson/VSLAM-UAV/vslam/mavrospy.launch.py")
    print()
    print("  Switch RC to GUIDED + ARM to fly")
    print("  Ctrl+C → LAND → stop all")
    print("="*55 + "\n")

    proc = run_visible("arena_web.py")
    if proc:
        try:
            proc.wait()
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
        if proc.poll() == 0:
            print("\n[MISSION] Arena complete ✓")
        else:
            print("\n[MISSION] Arena stopped")

    # Mission finished (or arena_web.py exited) — do NOT shut anything down.
    # camera_bridge, precision_landing, MAVProxy keep running.
    # Press Ctrl+C to stop everything.
    print("\n" + "="*55)
    print("  Arena script finished — all other processes still running:")
    print("    MAVProxy, camera_bridge, precision_landing")
    print("  Press Ctrl+C to stop everything.")
    print("="*55 + "\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()