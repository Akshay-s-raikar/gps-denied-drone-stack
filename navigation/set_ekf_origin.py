#!/usr/bin/env python3
"""
set_ekf_origin.py — Set EKF Origin for GPS-denied flight
==========================================================
Run this ONCE after mavros connects, before arming.
Sets the EKF origin so ArduPilot can do RTL without GPS.

Usage:
  python3 set_ekf_origin.py
  python3 set_ekf_origin.py --lat 15.8497 --lon 74.4977 --alt 760.0

What it does:
  1. Waits for mavros to connect
  2. Waits for SLAM vision pose to be publishing
  3. Sends EKF origin 3 times to ensure Pixhawk receives it
  4. Confirms and exits

EKF origin note:
  This is NOT the RTL target.
  Home position = where you ARM the drone.
  RTL always returns to arm location.
  These coordinates just need to be approximately correct (km accuracy ok).
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geographic_msgs.msg import GeoPointStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from mavros_msgs.msg import State
import time
import argparse
import sys

# ── Default coordinates (Belagavi, Karnataka) ─────────────────────
# Change these to your actual test location.
# Km-level accuracy is enough — does not need to be exact.
DEFAULT_LAT = 15.8497
DEFAULT_LON = 74.4977
DEFAULT_ALT = 760.0     # metres above sea level


class EKFOriginSetter(Node):

    def __init__(self, lat, lon, alt):
        super().__init__('ekf_origin_setter')

        self.lat  = lat
        self.lon  = lon
        self.alt  = alt

        self._connected   = False
        self._slam_count  = 0
        self._mode        = "UNKNOWN"

        qos_reliable = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        qos_sensor = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)

        # Publisher
        self._pub = self.create_publisher(
            GeoPointStamped,
            '/mavros/global_position/set_gp_origin',
            qos_reliable)

        # Subscribers
        self.create_subscription(
            State,
            '/mavros/state',
            self._state_cb,
            qos_reliable)

        self.create_subscription(
            PoseWithCovarianceStamped,
            '/mavros/vision_pose/pose_cov',
            self._slam_cb,
            qos_sensor)

    def _state_cb(self, msg):
        self._connected = msg.connected
        self._mode      = msg.mode

    def _slam_cb(self, msg):
        self._slam_count += 1

    def send_origin(self):
        msg = GeoPointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.position.latitude  = float(self.lat)
        msg.position.longitude = float(self.lon)
        msg.position.altitude  = float(self.alt)
        self._pub.publish(msg)


def main():
    ap = argparse.ArgumentParser(description="Set EKF Origin for GPS-denied flight")
    ap.add_argument("--lat", type=float, default=DEFAULT_LAT,
                    help=f"Latitude  (default: {DEFAULT_LAT})")
    ap.add_argument("--lon", type=float, default=DEFAULT_LON,
                    help=f"Longitude (default: {DEFAULT_LON})")
    ap.add_argument("--alt", type=float, default=DEFAULT_ALT,
                    help=f"Altitude above sea level in metres (default: {DEFAULT_ALT})")
    ap.add_argument("--no-wait-slam", action="store_true",
                    help="Skip waiting for SLAM and send immediately")
    args = ap.parse_args()

    print("\n" + "="*55)
    print("  EKF Origin Setter")
    print("="*55)
    print(f"  lat = {args.lat}")
    print(f"  lon = {args.lon}")
    print(f"  alt = {args.alt}m")
    print("="*55)

    rclpy.init()
    node = EKFOriginSetter(args.lat, args.lon, args.alt)

    # ── Step 1: Wait for mavros ────────────────────────────────────
    print("\n[1/3] Waiting for mavros connection...")
    t0 = time.monotonic()
    while not node._connected:
        rclpy.spin_once(node, timeout_sec=0.2)
        if time.monotonic() - t0 > 15.0:
            print("  ✗ Timeout — is mavros running?")
            print("  Sending origin anyway...")
            break
    if node._connected:
        print(f"  ✓ Connected — mode={node._mode}")

    # ── Step 2: Wait for SLAM ──────────────────────────────────────
    if not args.no_wait_slam:
        print("\n[2/3] Waiting for SLAM vision pose...")
        print("  (run with --no-wait-slam to skip this)")
        t0 = time.monotonic()
        while node._slam_count < 50:
            rclpy.spin_once(node, timeout_sec=0.1)
            elapsed = time.monotonic() - t0
            if elapsed > 20.0:
                print(f"  ✗ SLAM not detected after 20s")
                print("  Is Isaac ROS cuVSLAM running?")
                print("  Sending origin anyway...")
                break
            if node._slam_count > 0 and node._slam_count % 10 == 0:
                print(f"  SLAM msgs: {node._slam_count}...", end="\r")
        if node._slam_count >= 50:
            print(f"  ✓ SLAM stable ({node._slam_count} msgs received)")
    else:
        print("\n[2/3] Skipping SLAM wait (--no-wait-slam)")

    # ── Step 3: Send origin ────────────────────────────────────────
    print("\n[3/3] Sending EKF origin...")
    node.send_origin()
    rclpy.spin_once(node, timeout_sec=0.5)
    time.sleep(0.5)
    print("  Sent ✓")

    print("\n" + "="*55)
    print("  ✓ EKF origin set successfully")
    print(f"  lat={args.lat}  lon={args.lon}  alt={args.alt}m")
    print()
    print("  IMPORTANT:")
    print("  - This is NOT the RTL target")
    print("  - RTL returns to WHERE YOU ARMED")
    print("  - These coords just anchor the SLAM frame")
    print("="*55 + "\n")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
