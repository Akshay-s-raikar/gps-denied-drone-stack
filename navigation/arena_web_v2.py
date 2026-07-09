#!/usr/bin/env python3
"""
arena_web_v2.py — Arena Lawnmower — Position Setpoints
=======================================================
Uses position setpoints. Waypoints auto-generated in steps.
Separate step sizes for forward legs and sideways shifts —
shift steps are smaller so the drone moves slower and safer
near the side walls.

Run AFTER mavrospy.launch.py is running:
  python3 arena_web_v2.py

Switch RC to GUIDED + ARM to start.
Ctrl+C → RTL
"""

import time
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

TAKEOFF_ALTITUDE_M = 1.5     # meters
WAYPOINT_THRESHOLD = 0.35    # meters — how close = "reached"
WAYPOINT_TIMEOUT_S = 60.0    # seconds — max time per waypoint (6m leg takes ~20s)
SETTLE_TIME_S      = 1.0     # hover at each waypoint
TAKEOFF_WAIT_S     = 4.0     # wait after takeoff

# ── ARENA LAYOUT ──────────────────────────────────────────────────
ARENA_LENGTH_M = 6.0     # forward/back leg length (m)
LEG_SPACING_M  = 1.5     # sideways distance between legs (m)
MAX_LEGS       = 3       # number of forward/back legs

# ── STEP SIZES ────────────────────────────────────────────────────
LEG_STEP_M   = 1.0   # waypoint every N meters along forward/back legs
SHIFT_STEP_M = 0.3   # waypoint every N meters during sideways shift
                     # ← smaller = slower/safer near walls

# ── WALL MARGINS ──────────────────────────────────────────────────
X_MARGIN_M = 0.3     # min distance from forward/back wall (m)
Y_MARGIN_M = 0.4     # min distance from side wall on first shift (m)


# ─────────────────────────────────────────────────────────────────
# AUTO-GENERATE WAYPOINTS
# ─────────────────────────────────────────────────────────────────

def _generate_waypoints():
    """
    Build lawnmower path with:
    - LEG_STEP_M spacing along each forward/back leg
    - SHIFT_STEP_M spacing during each sideways shift (slower)
    - Y_MARGIN_M applied to clamp minimum Y on first shift
    """
    waypoints = []
    direction = 1     # +1 = forward, -1 = backward
    y = 0.0

    leg_steps   = max(1, int(round(ARENA_LENGTH_M / LEG_STEP_M)))
    shift_steps = max(1, int(round(LEG_SPACING_M  / SHIFT_STEP_M)))

    for leg in range(MAX_LEGS):
        # ── forward/back leg ──────────────────────────────────────
        for s in range(1, leg_steps + 1):
            x = s * LEG_STEP_M if direction == 1 \
                else ARENA_LENGTH_M - s * LEG_STEP_M
            x = max(X_MARGIN_M, min(ARENA_LENGTH_M, x))
            waypoints.append((round(x, 3), round(y, 3)))

        # ── sideways shift to next leg ────────────────────────────
        if leg < MAX_LEGS - 1:
            x_hold = waypoints[-1][0]   # hold X during shift
            for s in range(1, shift_steps + 1):
                y_next = y + s * SHIFT_STEP_M
                # clamp first shift so drone stays off side wall
                if leg == 0:
                    y_next = max(Y_MARGIN_M, y_next)
                y_next = min(y_next, (MAX_LEGS - 1) * LEG_SPACING_M)
                waypoints.append((round(x_hold, 3), round(y_next, 3)))
            y = round(y + LEG_SPACING_M, 3)
            direction *= -1

    return waypoints


WAYPOINTS = _generate_waypoints()


# ═══════════════════════════════════════════════════════════════════
# MAVROS BRIDGE
# ═══════════════════════════════════════════════════════════════════

class MAVROSBridge(Node):

    def __init__(self):
        # unique node name avoids conflict on quick restart
        node_name = f'arena_web_v2_{int(time.time()) % 10000}'
        super().__init__(node_name)
        self._lock    = threading.Lock()
        self.mode     = "UNKNOWN"
        self.armed    = False
        self.local_x  = 0.0
        self.local_y  = 0.0
        self.local_z  = 0.0
        self._last_hb = time.monotonic()

        qos_s = QoSProfile(depth=10,
                           reliability=ReliabilityPolicy.BEST_EFFORT,
                           durability=DurabilityPolicy.VOLATILE)
        qos_r = QoSProfile(depth=10,
                           reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.VOLATILE)

        self.create_subscription(State,
            '/mavros/state', self._state_cb, qos_r)
        self.create_subscription(PoseStamped,
            '/mavros/local_position/pose', self._pose_cb, qos_s)

        self._pos_pub = self.create_publisher(
            PoseStamped, '/mavros/setpoint_position/local', qos_r)
        self._vel_pub = self.create_publisher(
            TwistStamped, '/mavros/setpoint_velocity/cmd_vel', qos_r)

        self._mode_client    = self.create_client(SetMode,     '/mavros/set_mode')
        self._arm_client     = self.create_client(CommandBool, '/mavros/cmd/arming')
        self._takeoff_client = self.create_client(CommandTOL,  '/mavros/cmd/takeoff')

        self.get_logger().info(f'MAVROSBridge started as {node_name}')

    def _state_cb(self, msg):
        with self._lock:
            self.mode     = msg.mode
            self.armed    = msg.armed
            self._last_hb = time.monotonic()

    def _pose_cb(self, msg):
        with self._lock:
            self.local_x = msg.pose.position.x
            self.local_y = msg.pose.position.y
            self.local_z = msg.pose.position.z

    def get_position(self):
        with self._lock:
            return self.local_x, self.local_y, self.local_z

    def get_mode(self):
        with self._lock: return self.mode

    def is_armed(self):
        with self._lock: return self.armed

    def hb_age(self):
        with self._lock: return time.monotonic() - self._last_hb

    def set_mode(self, mode_name):
        if not self._mode_client.wait_for_service(timeout_sec=3.0):
            return False
        req = SetMode.Request()
        req.custom_mode = mode_name
        future = self._mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        if future.result():
            print(f"[MAV] Mode → {mode_name} ✓")
            return future.result().mode_sent
        return False

    def arm(self):
        if not self._arm_client.wait_for_service(timeout_sec=3.0):
            return False
        req = CommandBool.Request()
        req.value = True
        future = self._arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        return future.result().success if future.result() else False

    def takeoff(self, alt):
        if not self._takeoff_client.wait_for_service(timeout_sec=3.0):
            return False
        req = CommandTOL.Request()
        req.altitude = float(alt)
        future = self._takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        print(f"[MAV] Takeoff → {alt}m ✓")
        return True

    def go_to(self, x, y, z=None):
        if z is None:
            z = TAKEOFF_ALTITUDE_M
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self._pos_pub.publish(msg)

    def stop(self):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        self._vel_pub.publish(msg)

    def shutdown(self):
        self.destroy_node()


# ═══════════════════════════════════════════════════════════════════
# MISSION RUNNER
# ═══════════════════════════════════════════════════════════════════

class MissionRunner:

    def __init__(self, mav, waypoints):
        self.mav       = mav
        self.waypoints = waypoints
        self.home_x    = 0.0
        self.home_y    = 0.0

    def _wait_mode(self, target, timeout=30.0):
        print(f"[Mission] Waiting for mode: {target}")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.mav.get_mode() == target:
                return True
            time.sleep(0.2)
        print(f"[Mission] Timeout waiting for {target}")
        return False

    def _wait_arm(self, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.mav.is_armed():
                return True
            time.sleep(0.2)
        return False

    def _wait_alt(self, target, tol=0.2, timeout=20.0):
        print(f"[Mission] Climbing to {target}m...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            _, _, z = self.mav.get_position()
            if z >= target - tol:
                print(f"[Mission] Altitude reached: {z:.2f}m ✓")
                return True
            time.sleep(0.2)
        print("[Mission] Altitude timeout")
        return False

    def _fly_to(self, wx, wy, idx, total):
        tx = self.home_x + wx
        ty = self.home_y + wy
        tz = TAKEOFF_ALTITUDE_M

        print(f"[WP {idx+1:2d}/{total}] → "
              f"x={tx:+.2f} y={ty:+.2f} z={tz:.2f}")

        t0 = time.monotonic()
        while time.monotonic() - t0 < WAYPOINT_TIMEOUT_S:
            if self.mav.hb_age() > 5.0:
                print("[Mission] MAVLink lost → RTL")
                self.mav.set_mode("RTL")
                return False

            self.mav.go_to(tx, ty, tz)

            cx, cy, _ = self.mav.get_position()
            dist    = math.sqrt((cx - tx)**2 + (cy - ty)**2)
            elapsed = time.monotonic() - t0
            print(f"\r    dist={dist:.2f}m  "
                  f"pos=({cx:.2f},{cy:.2f})  t={elapsed:.1f}s    ",
                  end="", flush=True)

            if dist <= WAYPOINT_THRESHOLD:
                print(f"\n    ✓ Reached (dist={dist:.2f}m)")
                t_s = time.monotonic()
                while time.monotonic() - t_s < SETTLE_TIME_S:
                    self.mav.go_to(tx, ty, tz)
                    time.sleep(0.05)
                return True

            time.sleep(0.05)

        print(f"\n    ✗ Timeout at WP{idx+1}")
        return False

    def run(self):
        total = len(self.waypoints)
        print("\n" + "="*55)
        print(f"  Arena Mission v2 — {total} waypoints")
        print(f"  Leg step: {LEG_STEP_M}m  Shift step: {SHIFT_STEP_M}m")
        print(f"  Y wall margin: {Y_MARGIN_M}m")
        print("="*55)

        # Wait for MAVROS
        print("[Mission] Waiting for MAVROS...")
        t0 = time.monotonic()
        while self.mav.get_mode() == "UNKNOWN":
            time.sleep(0.2)
            if time.monotonic() - t0 > 15.0:
                print("[Mission] MAVROS not responding")
                return False
        print(f"[Mission] Connected — mode={self.mav.get_mode()}")

        # Wait for EKF
        print("[Mission] Waiting for EKF position...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < 15.0:
            x, y, z = self.mav.get_position()
            if not (x == 0.0 and y == 0.0 and z == 0.0):
                break
            time.sleep(0.3)
        x, y, z = self.mav.get_position()
        print(f"[Mission] EKF: x={x:.2f} y={y:.2f} z={z:.2f}")

        # Wait for GUIDED + ARM
        print("\n[Mission] Switch RC to GUIDED to start...")
        if not self._wait_mode("GUIDED", timeout=120.0):
            return False

        if not self.mav.is_armed():
            print("[Mission] Arming...")
            if not self.mav.arm():
                print("[Mission] Arm failed")
                return False
        if not self._wait_arm(timeout=10.0):
            print("[Mission] Arm timeout")
            return False
        print("[Mission] Armed ✓")

        # Record home
        self.home_x, self.home_y, _ = self.mav.get_position()
        print(f"[Mission] Home: x={self.home_x:.2f} y={self.home_y:.2f}")

        # Takeoff
        self.mav.takeoff(TAKEOFF_ALTITUDE_M)
        time.sleep(2.0)
        if not self._wait_alt(TAKEOFF_ALTITUDE_M, timeout=15.0):
            print("[Mission] Takeoff failed → RTL")
            self.mav.set_mode("RTL")
            return False
        time.sleep(TAKEOFF_WAIT_S)
        print("[Mission] Airborne ✓\n")

        # Fly all waypoints
        for i, (wx, wy) in enumerate(self.waypoints):
            if not self._fly_to(wx, wy, i, total):
                print(f"[Mission] Failed at WP{i+1} → RTL")
                self.mav.set_mode("RTL")
                return False

        print("\n" + "="*55)
        print(f"  Mission complete — all {total} waypoints ✓")
        print("  RTL...")
        print("="*55 + "\n")
        self.mav.set_mode("RTL")
        return True


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*55)
    print("  Arena Web v2 — IRoC-U 2026")
    print("="*55)
    print(f"  Altitude   : {TAKEOFF_ALTITUDE_M}m")
    print(f"  Leg step   : {LEG_STEP_M}m")
    print(f"  Shift step : {SHIFT_STEP_M}m  ← slow & safe near walls")
    print(f"  Y margin   : {Y_MARGIN_M}m  ← min dist from side wall")
    print(f"  Waypoints  : {len(WAYPOINTS)} total")
    print()
    for i, (x, y) in enumerate(WAYPOINTS):
        print(f"    WP{i+1:2d}: x={x:+.2f}  y={y:.2f}")
    print("="*55 + "\n")

    rclpy.init()
    mav = MAVROSBridge()

    ros_thread = threading.Thread(
        target=rclpy.spin, args=(mav,), daemon=True)
    ros_thread.start()

    runner = MissionRunner(mav, WAYPOINTS)

    try:
        runner.run()
    except KeyboardInterrupt:
        print("\n[Ctrl-C] RTL...")
        try: mav.stop()
        except Exception: pass
        mav.set_mode("RTL")
    finally:
        try: mav.stop()
        except Exception: pass
        time.sleep(0.5)
        try: mav.shutdown()
        except Exception: pass
        try: rclpy.shutdown()
        except Exception: pass
        print("[Done]")


if __name__ == "__main__":
    main()
