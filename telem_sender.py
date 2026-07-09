#!/usr/bin/env python3
"""
telem_sender.py — Run on HOST (not inside Docker)
==================================================
Receives telemetry from telem_collector.py (inside Docker) via UDP.
Forwards full telemetry to GCS via ZeroMQ PUB + MessagePack.
Also forwards X,Y,Z to xyz_receiver.py via UDP localhost:5021.

  python3 telem_sender.py
  python3 telem_sender.py --rate 10
"""

from __future__ import annotations

import argparse
import json
import logging
import socket
import threading
import time
from typing import Any, Dict, Optional

from zmq_transport import (
    ROLE_JETSON_TLM,
    ZMQ_PORT_JETSON_TELEMETRY,
    ZMQ_TOPIC_TELEMETRY,
    ZmqPublisher,
    pack_msgpack,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TELEM-SENDER] %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

LISTEN_HOST   = "0.0.0.0"
LISTEN_PORT   = 5022          # receives from telem_collector.py in Docker

XYZ_UDP_HOST  = "127.0.0.1"
XYZ_UDP_PORT  = 5021          # sends to xyz_receiver.py

TELEMETRY_HZ  = 10.0
STALE_TIMEOUT = 3.0           # seconds before data considered stale


# ─────────────────────────────────────────────────────────────────
# RECEIVER
# ─────────────────────────────────────────────────────────────────

class TelemReceiver:
    """Receives telemetry JSON from Docker container via UDP."""

    def __init__(self):
        self._lock      = threading.Lock()
        self._latest    : Optional[Dict[str, Any]] = None
        self._last_rx   : float = 0.0
        self._stop      = threading.Event()

    def start(self):
        t = threading.Thread(target=self._listen, daemon=True, name="UDPRecv")
        t.start()
        log.info("Listening for telemetry on UDP %s:%d",
                 LISTEN_HOST, LISTEN_PORT)

    def stop(self):
        self._stop.set()

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((LISTEN_HOST, LISTEN_PORT))
        sock.settimeout(0.5)
        log.info("UDP receiver bound to %s:%d", LISTEN_HOST, LISTEN_PORT)

        while not self._stop.is_set():
            try:
                raw, _ = sock.recvfrom(65535)
                data   = json.loads(raw.decode("utf-8"))
                with self._lock:
                    self._latest  = data
                    self._last_rx = time.monotonic()
            except socket.timeout:
                continue
            except Exception as e:
                log.warning("UDP receive error: %s", e)

        sock.close()

    def get_latest(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._latest is None:
                return None
            age = time.monotonic() - self._last_rx
            if age > STALE_TIMEOUT:
                log.warning("Telemetry stale (%.1fs old)", age)
                return None
            return dict(self._latest)

    def is_connected(self) -> bool:
        with self._lock:
            return (self._latest is not None and
                    time.monotonic() - self._last_rx < STALE_TIMEOUT)


# ─────────────────────────────────────────────────────────────────
# SENDER
# ─────────────────────────────────────────────────────────────────

class TelemSender:
    """Publishes telemetry to GCS via ZMQ and XYZ to xyz_receiver via UDP."""

    def __init__(self, rate_hz: float = TELEMETRY_HZ):
        self.rate_hz  = rate_hz
        self._stop    = threading.Event()

        self._pub = ZmqPublisher(
            ZMQ_PORT_JETSON_TELEMETRY,
            ZMQ_TOPIC_TELEMETRY,
            discover_role=ROLE_JETSON_TLM,
        )

        self._receiver = TelemReceiver()
        self._xyz_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def start(self):
        self._pub.start()
        self._receiver.start()
        log.info("Publishing to GCS on ZMQ :%d", ZMQ_PORT_JETSON_TELEMETRY)
        log.info("Forwarding XYZ to xyz_receiver on UDP %s:%d",
                 XYZ_UDP_HOST, XYZ_UDP_PORT)

    def stop(self):
        self._stop.set()
        self._receiver.stop()
        self._pub.close()
        self._xyz_sock.close()

    def _send_xyz(self, data: Dict[str, Any]):
        """Forward X,Y,Z to xyz_receiver.py."""
        try:
            xyz = {
                "x":  data.get("position_x", 0.0),
                "y":  data.get("position_y", 0.0),
                "z":  data.get("position_z", 0.0),
                "ts": data.get("timestamp",  ""),
            }
            self._xyz_sock.sendto(
                json.dumps(xyz).encode(),
                (XYZ_UDP_HOST, XYZ_UDP_PORT))
        except Exception as e:
            log.debug("XYZ UDP send error: %s", e)

    def publish_loop(self):
        period    = 1.0 / max(0.1, self.rate_hz)
        next_t    = time.perf_counter()
        last_log  = 0.0
        rx_count  = 0
        miss_count = 0

        while not self._stop.is_set():
            data = self._receiver.get_latest()

            if data is not None:
                # Send full telemetry to GCS via ZMQ
                self._pub.publish_multipart([pack_msgpack(data)])

                # Send XYZ to xyz_receiver.py via UDP
                self._send_xyz(data)

                rx_count += 1
            else:
                miss_count += 1
                if time.monotonic() - last_log > 5.0:
                    if self._receiver.is_connected():
                        log.warning("No telemetry data received yet")
                    else:
                        log.warning(
                            "Docker collector not sending — "
                            "is telem_collector.py running inside Docker?")
                    last_log = time.monotonic()

            # Status log every 10 seconds
            if time.monotonic() - last_log > 10.0:
                log.info("Published %d packets, missed %d",
                         rx_count, miss_count)
                rx_count   = 0
                miss_count = 0
                last_log   = time.monotonic()

            next_t += period
            sleep   = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Telem sender: UDP from Docker → ZMQ to GCS")
    p.add_argument("--rate", type=float, default=TELEMETRY_HZ,
                   help="Publish Hz (default 10)")
    args = p.parse_args()

    log.info("=" * 55)
    log.info("  Telem Sender — Docker UDP → GCS ZMQ")
    log.info("  Listen : UDP %s:%d  (from Docker)",
             LISTEN_HOST, LISTEN_PORT)
    log.info("  GCS    : ZMQ port %d", ZMQ_PORT_JETSON_TELEMETRY)
    log.info("  XYZ    : UDP %s:%d  (to xyz_receiver)",
             XYZ_UDP_HOST, XYZ_UDP_PORT)
    log.info("  Rate   : %.1f Hz", args.rate)
    log.info("=" * 55)

    sender = TelemSender(rate_hz=args.rate)
    sender.start()

    try:
        sender.publish_loop()
    except KeyboardInterrupt:
        log.info("Shutdown")
    finally:
        sender.stop()


if __name__ == "__main__":
    main()