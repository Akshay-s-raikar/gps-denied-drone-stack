#!/usr/bin/env python3
"""
rock_detector_v6.py — IRoC-U 2026  YOLO + DINOv2 + ZMQ sender
===============================================================
Pipeline:
  1. YOLO  → bounding boxes
  2. For each box: crop → rotate N angles → DINOv2 embed batch
     → cosine sim vs all seeds → take MAX across all rotations
  3. Highest-sim seed wins → if sim >= DINO_THRESH:
       a. Save full 1280×720 frame to detections/{seed_name}/
       b. Send (seed reference image + full captured frame + X,Y,Z)
          via ZMQ exactly like post_landing_sender.py
       c. X,Y,Z read live from UDP localhost:5021 (xyz_receiver feed)
  4. 4-second global cooldown after any detection before next fires
  5. R key resets all detections and cooldown

Rotation invariance:
  - Seeds augmented at every SEED_ROT_STEP degrees (default 15° → 24 angles)
  - Inference crops embedded at every INFER_ROT_STEP degrees (default 30° → 12)

ZMQ send (per detection, once, respecting cooldown):
  - pair_id  : unique UUID per detection
  - image 1  : seed reference file from refs/ folder (ROLE_REFERENCE)
  - image 2  : full 1280×720 captured frame saved to disk (ROLE_ACTUAL)
  - x, y, z  : latest position read from UDP:5021

Usage:
  python rock_detector_v6.py --cam 0 --refs refs/
  python rock_detector_v6.py --video test.mp4 --refs refs/ --no-display
"""

import cv2
import numpy as np
import torch
import time
import os
import sys
import json
import uuid
import socket
import argparse
import threading
import platform
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from PIL import Image

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S")
log = logging.getLogger("RockDetectorV6")

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
CFG = {
    "CAM_W":             1280,
    "CAM_H":             720,
    "CAM_FPS":           30,

    "YOLO_MODEL":        "yolov8n.pt",
    "YOLO_CONF":          0.40,
    "YOLO_IOU":          0.45,
    "YOLO_IMG_SIZE":     640,
    "YOLO_MAX_DET":      20,

    "CROP_PAD":          25,
    "CROP_SIZE":         256,

    "DINO_MODEL":        "dinov2_vits14",
    "DINO_THRESH":       0.55,

    # Rotation invariance
    "SEED_ROT_STEP":     15,      # 0..345 → 24 angles
    "INFER_ROT_STEP":    30,      # 0..330 → 12 angles
    "SEED_EXTRA_AUGS":   True,

    "OUTPUT_DIR":        "detections",
    "GLOBAL_COOLDOWN":   5.0,

    # ZMQ sender
    "ZMQ_PORT":          5555,
    "ZMQ_DISCOVER":      True,
    "ZMQ_CONNECT_WAIT":  2.5,    # seconds to wait after publisher start

    # XYZ UDP feed (from xyz_receiver producer)
    "XYZ_UDP_PORT":      5021,

    "SHOW_DISPLAY":      True,
    "FONT":              cv2.FONT_HERSHEY_SIMPLEX,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    log.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    log.warning("CUDA not available — running on CPU (slow)")

# ─────────────────────────────────────────────────────────────────
# XYZ POSITION LISTENER  (UDP :5021, non-blocking)
# Mirrors xyz_receiver.py — reads the latest X,Y,Z from the
# telemetry_data_send.py UDP broadcast.
# ─────────────────────────────────────────────────────────────────

class XYZListener:
    """
    Background thread that listens on UDP :XYZ_UDP_PORT and keeps
    the latest X, Y, Z position. Thread-safe getter.
    """

    def __init__(self, port: int = 5021):
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
        self._ts = ""
        self._lock = threading.Lock()
        self._port = port
        self._stop = False
        self._sock: Optional[socket.socket] = None
        t = threading.Thread(target=self._run, daemon=True)
        t.start()
        log.info(f"[XYZ] Listener started on UDP :{port}")

    def _run(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind(("0.0.0.0", self._port))
            self._sock.settimeout(1.0)
        except Exception as e:
            log.error(f"[XYZ] Cannot bind UDP :{self._port} — {e}")
            return

        while not self._stop:
            try:
                data, _ = self._sock.recvfrom(1024)
                msg = json.loads(data.decode())
                with self._lock:
                    self._x  = float(msg.get("x",  0.0))
                    self._y  = float(msg.get("y",  0.0))
                    self._z  = float(msg.get("z",  0.0))
                    self._ts = str(msg.get("ts", ""))
            except socket.timeout:
                continue
            except Exception as e:
                log.warning(f"[XYZ] Parse error: {e}")

        if self._sock:
            self._sock.close()

    def get(self) -> Tuple[float, float, float]:
        with self._lock:
            return self._x, self._y, self._z

    def stop(self):
        self._stop = True

# ─────────────────────────────────────────────────────────────────
# ZMQ IMAGE SENDER
# Same logic as post_landing_sender.py ImageSender.
# Sends a (reference, actual) image pair with X,Y,Z per detection.
# ─────────────────────────────────────────────────────────────────

_ZMQ_OK = False
try:
    import zmq_transport as _zt
    ROLE_REFERENCE = _zt.ROLE_REFERENCE
    ROLE_ACTUAL    = _zt.ROLE_ACTUAL
    _ZMQ_OK = True
    log.info("zmq_transport found ✓")
except ImportError:
    ROLE_REFERENCE = "reference"
    ROLE_ACTUAL    = "actual"
    log.warning("zmq_transport NOT found — ZMQ sending disabled (detection/save still works)")


class ZMQDetectionSender:
    """
    Wraps zmq_transport.ImagePublisher.
    Lazy-initialised on first send so startup is fast even if ZMQ
    is unavailable.
    Sends are dispatched on a background thread so the main loop
    never blocks.
    """

    def __init__(self, port: int, discover: bool, connect_wait: float):
        self._port         = port
        self._discover     = discover
        self._connect_wait = connect_wait
        self._pub          = None
        self._ready        = False
        self._init_lock    = threading.Lock()
        self._send_queue: List[dict] = []
        self._queue_lock   = threading.Lock()
        self._worker       = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── init publisher (once) ─────────────────────────────────────
    def _ensure_ready(self) -> bool:
        with self._init_lock:
            if self._ready:
                return True
            if not _ZMQ_OK:
                log.error("[ZMQ] zmq_transport not available — send skipped")
                return False
            try:
                self._pub = _zt.ImagePublisher(
                    zmq_port=self._port,
                    discover=self._discover)
                self._pub.start()
                log.info(f"[ZMQ] Publisher started on port {self._port} "
                         f"— waiting {self._connect_wait}s for subscribers...")
                time.sleep(self._connect_wait)
                self._ready = True
                log.info("[ZMQ] Publisher ready ✓")
            except Exception as e:
                log.error(f"[ZMQ] Publisher init failed: {e}")
                return False
        return True

    # ── background worker ─────────────────────────────────────────
    def _worker_loop(self):
        while True:
            item = None
            with self._queue_lock:
                if self._send_queue:
                    item = self._send_queue.pop(0)
            if item is None:
                time.sleep(0.05)
                continue
            self._do_send(item)

    def _do_send(self, item: dict):
        if not self._ensure_ready():
            return
        seed_path   = item["seed_path"]
        actual_path = item["actual_path"]
        x, y, z     = item["x"], item["y"], item["z"]
        label       = item["label"]
        pair_id     = item["pair_id"]

        if not os.path.exists(seed_path):
            log.error(f"[ZMQ] Seed file missing: {seed_path}"); return
        if not os.path.exists(actual_path):
            log.error(f"[ZMQ] Actual file missing: {actual_path}"); return

        try:
            ok1 = self._pub.publish(seed_path,
                                    x=0.0, y=0.0, z=0.0,
                                    role=ROLE_REFERENCE,
                                    pair_id=pair_id)
            time.sleep(0.1)
            ok2 = self._pub.publish(actual_path,
                                    x=x, y=y, z=z,
                                    role=ROLE_ACTUAL,
                                    pair_id=pair_id)
            time.sleep(0.3)
            status = "✓" if (ok1 and ok2) else "PARTIAL/FAIL"
            log.info(f"[ZMQ] Sent {label}  X={x:.3f} Y={y:.3f} Z={z:.3f} "
                     f"pair={pair_id[:8]}  [{status}]")
        except Exception as e:
            log.error(f"[ZMQ] Send error: {e}")

    # ── public API ────────────────────────────────────────────────
    def enqueue(self, seed_path: str, actual_path: str,
                x: float, y: float, z: float, label: str):
        """
        Queue a send (non-blocking). Worker thread handles the actual
        ZMQ publish so the main detection loop never stalls.
        """
        item = dict(
            seed_path   = seed_path,
            actual_path = actual_path,
            x=x, y=y, z=z,
            label       = label,
            pair_id     = str(uuid.uuid4()),
        )
        with self._queue_lock:
            self._send_queue.append(item)
        log.info(f"[ZMQ] Queued send for {label} @ ({x:.3f},{y:.3f},{z:.3f})")

    def close(self):
        if self._pub:
            try: self._pub.close()
            except Exception: pass

# ─────────────────────────────────────────────────────────────────
# ROTATION UTILITIES
# ─────────────────────────────────────────────────────────────────

def rotate_bgr(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if angle_deg == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def rotation_variants(img: np.ndarray, step_deg: int) -> List[np.ndarray]:
    return [rotate_bgr(img, a) for a in range(0, 360, step_deg)]

# ─────────────────────────────────────────────────────────────────
# SEED AUGMENTATION
# ─────────────────────────────────────────────────────────────────

def augment_seed(img_bgr: np.ndarray) -> List[np.ndarray]:
    step      = CFG["SEED_ROT_STEP"]
    rotations = rotation_variants(img_bgr, step)
    augs: List[np.ndarray] = list(rotations)

    if CFG["SEED_EXTRA_AUGS"]:
        for r in rotations:
            augs.append(cv2.flip(r, 1))
        for gamma in (0.7, 1.4):
            lut = np.array(
                [min(255, int((i / 255.0) ** (1.0 / gamma) * 255))
                 for i in range(256)], dtype=np.uint8)
            augs.append(cv2.LUT(img_bgr, lut))
        h, w = img_bgr.shape[:2]
        m = int(min(h, w) * 0.12)
        if h - 2 * m > 10 and w - 2 * m > 10:
            zoom = img_bgr[m:h - m, m:w - m]
            augs.append(cv2.resize(zoom, (w, h)))

    return augs

# ─────────────────────────────────────────────────────────────────
# DINOv2 EMBEDDER
# ─────────────────────────────────────────────────────────────────

class DinoEmbedder:
    def __init__(self, model_name):
        log.info(f"Loading DINOv2 ({model_name})...")
        self.model = torch.hub.load("facebookresearch/dinov2",
                                    model_name,
                                    pretrained=True).to(DEVICE).eval()
        self._mean = torch.tensor([0.485, 0.456, 0.406],
                                  device=DEVICE).view(1, 3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225],
                                  device=DEVICE).view(1, 3, 1, 1)
        log.info("DINOv2 loaded ✓")

    def _to_tensor(self, pil_img: Image.Image) -> torch.Tensor:
        arr = np.asarray(pil_img.resize((224, 224))).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        return (t - self._mean) / self._std

    @torch.no_grad()
    def embed(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        if bgr is None or bgr.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t   = self._to_tensor(Image.fromarray(rgb))
            f   = self.model(t).squeeze(0).float().cpu().numpy()
            n   = np.linalg.norm(f)
            return f / n if n >= 1e-8 else None
        except Exception as e:
            log.warning(f"DINO embed error: {e}")
            return None

    @torch.no_grad()
    def embed_batch(self, bgr_list: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        valid_idx, tensors = [], []
        for i, bgr in enumerate(bgr_list):
            if bgr is None or bgr.size == 0:
                continue
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensors.append(self._to_tensor(Image.fromarray(rgb)).squeeze(0))
                valid_idx.append(i)
            except Exception:
                continue
        out: List[Optional[np.ndarray]] = [None] * len(bgr_list)
        if not tensors:
            return out
        try:
            batch = torch.stack(tensors).to(DEVICE)
            feats = self.model(batch).float().cpu().numpy()
            for k, i in enumerate(valid_idx):
                f = feats[k]; n = np.linalg.norm(f)
                if n >= 1e-8:
                    out[i] = f / n
        except Exception as e:
            log.warning(f"DINO batch error: {e} — fallback per-image")
            for i in valid_idx:
                out[i] = self.embed(bgr_list[i])
        return out

    @torch.no_grad()
    def embed_rotations(self, bgr: np.ndarray,
                        step_deg: int) -> List[Optional[np.ndarray]]:
        return self.embed_batch(rotation_variants(bgr, step_deg))

# ─────────────────────────────────────────────────────────────────
# SEED BANK
# ─────────────────────────────────────────────────────────────────

class SeedBank:
    def __init__(self, refs_dir, embedder: DinoEmbedder):
        self.embedder    = embedder
        self.names:      List[str]        = []
        self.images_lr:  List[np.ndarray] = []   # original 128×128 for display
        self.ref_paths:  List[str]        = []   # original file paths for ZMQ send
        self.banks:      List[np.ndarray] = []
        self._load(Path(refs_dir))

    def _load(self, d: Path):
        if not d.exists():
            raise FileNotFoundError(f"Ref folder not found: {d}")
        files = []
        for pat in ("ref_*.jpg", "ref_*.png", "ref_*.jpeg",
                    "seed_*.jpg", "seed_*.png"):
            files.extend(sorted(d.glob(pat)))
        seen, uniq = set(), []
        for f in files:
            if f.name not in seen:
                seen.add(f.name); uniq.append(f)
        if not uniq:
            raise FileNotFoundError(f"No seed images in {d}")
        log.info(f"Building seed bank from {len(uniq)} image(s)"
                 f"  (rot_step={CFG['SEED_ROT_STEP']}°)")
        for f in uniq:
            img = cv2.imread(str(f))
            if img is None:
                log.warning(f"  cannot read {f.name} — skipped"); continue
            big  = cv2.resize(img, (CFG["CROP_SIZE"], CFG["CROP_SIZE"]))
            augs = augment_seed(big)
            embs = [e for e in self.embedder.embed_batch(augs) if e is not None]
            if not embs:
                log.warning(f"  {f.name}: no embeddings — skipped"); continue
            self.names.append(f.stem)
            self.images_lr.append(img)
            self.ref_paths.append(str(f.resolve()))   # ← kept for ZMQ send
            self.banks.append(np.stack(embs))
            log.info(f"  {f.name}: {len(embs)} augmented embeddings ✓")
        if not self.banks:
            raise RuntimeError("Seed bank empty.")
        log.info(f"Seed bank ready: {len(self.banks)} seed(s) ✓")

    def best_match_multi(self,
                         embs: List[Optional[np.ndarray]]) -> Tuple[int, float]:
        valid = [e for e in embs if e is not None]
        if not valid:
            return -1, -1.0
        E = np.stack(valid)
        best_id, best_sim = -1, -1.0
        for sid, bank in enumerate(self.banks):
            sim = float(np.max(E @ bank.T))
            if sim > best_sim:
                best_sim, best_id = sim, sid
        return best_id, best_sim

    def sim_per_seed_multi(self,
                           embs: List[Optional[np.ndarray]]) -> List[float]:
        valid = [e for e in embs if e is not None]
        if not valid:
            return [0.0] * len(self.banks)
        E = np.stack(valid)
        return [float(np.max(E @ bank.T)) for bank in self.banks]

    def __len__(self):
        return len(self.banks)

# ─────────────────────────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────────────────────────

def load_yolo(model_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed — pip install ultralytics")
        sys.exit(1)
    path = Path(model_path)
    model = YOLO(str(path) if path.exists() else model_path)
    log.info("YOLO loaded ✓")
    return model


def run_yolo(model, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
    try:
        results = model(frame,
                        conf=CFG["YOLO_CONF"], iou=CFG["YOLO_IOU"],
                        imgsz=CFG["YOLO_IMG_SIZE"], max_det=CFG["YOLO_MAX_DET"],
                        verbose=False)
        boxes = []
        if results and results[0].boxes is not None:
            for b in results[0].boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2, float(b.conf[0])))
        return boxes
    except Exception as e:
        log.warning(f"YOLO error: {e}"); return []

# ─────────────────────────────────────────────────────────────────
# OUTPUT MANAGER  (save to disk + trigger ZMQ send)
# ─────────────────────────────────────────────────────────────────

class OutputManager:
    def __init__(self, bank: SeedBank,
                 zmq_sender: ZMQDetectionSender,
                 xyz_listener: XYZListener):
        self.bank          = bank
        self.zmq           = zmq_sender
        self.xyz           = xyz_listener
        self.base_dir      = Path(CFG["OUTPUT_DIR"])
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.seed_dirs: List[Path] = []
        for name in bank.names:
            d = self.base_dir / name
            d.mkdir(parents=True, exist_ok=True)
            self.seed_dirs.append(d)
            log.info(f"  Seed folder: {d.name}/")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path          = self.base_dir / f"detections_{ts}.json"
        self.results:     list  = []
        self.instance_count: dict = {}
        self._last_save_time: float = 0.0

    def global_ready(self) -> bool:
        return (time.time() - self._last_save_time) >= CFG["GLOBAL_COOLDOWN"]

    def cooldown_remaining(self) -> float:
        return max(0.0, CFG["GLOBAL_COOLDOWN"] - (time.time() - self._last_save_time))

    def save_and_send(self, sid: int, bbox: Tuple[int, int, int, int],
                      full_frame: np.ndarray, dino_sim: float):
        """
        1. Save full frame to disk.
        2. Read current X,Y,Z from UDP listener.
        3. Enqueue ZMQ send: (seed ref image, saved full frame, X,Y,Z).
        Respects the 4s global cooldown — caller must check global_ready() first.
        """
        name  = self.bank.names[sid]
        count = self.instance_count.get(sid, 0) + 1
        self.instance_count[sid] = count
        self._last_save_time = time.time()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        saved_path = self.seed_dirs[sid] / f"inst{count:03d}_{ts}.jpg"
        cv2.imwrite(str(saved_path), full_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # ── JSON log ─────────────────────────────────────────────
        x, y, z = self.xyz.get()
        rec = {
            "seed_id":    sid,
            "seed_name":  name,
            "instance":   count,
            "timestamp":  ts,
            "bbox":       list(bbox),
            "coord_x_px": round(cx, 1),
            "coord_y_px": round(cy, 1),
            "dino_sim":   round(dino_sim, 4),
            "drone_x_m":  round(x, 4),
            "drone_y_m":  round(y, 4),
            "drone_z_m":  round(z, 4),
            "file":       str(saved_path),
        }
        self.results.append(rec)
        try:
            with open(self.json_path, "w") as f:
                json.dump(self.results, f, indent=2)
                f.flush(); os.fsync(f.fileno())
        except Exception as e:
            log.warning(f"JSON save error: {e}")

        log.info(f"[SAVED] {name}/inst{count:03d}  bbox={bbox}"
                 f"  dino_sim={dino_sim:.3f}"
                 f"  drone=({x:.3f},{y:.3f},{z:.3f})"
                 f"  → {saved_path.name}")

        # ── ZMQ send ─────────────────────────────────────────────
        # seed ref path comes from SeedBank (original file in refs/)
        # actual path is the full frame we just saved to disk
        self.zmq.enqueue(
            seed_path   = self.bank.ref_paths[sid],
            actual_path = str(saved_path),
            x=x, y=y, z=z,
            label       = f"{name}_inst{count:03d}",
        )

    def reset(self):
        self.instance_count.clear()
        self._last_save_time = 0.0
        log.info("[RESET] Detection counts and cooldown cleared")

    def summary(self):
        log.info("=" * 52)
        log.info(f"  MISSION SUMMARY — {len(self.results)} detection(s)")
        for sid, count in sorted(self.instance_count.items()):
            names = [r["seed_name"] for r in self.results if r["seed_id"] == sid]
            log.info(f"  {names[0] if names else f'seed_{sid}'}/  →  {count} instance(s)")
        log.info(f"  Log: {self.json_path}")
        log.info("=" * 52)

# ─────────────────────────────────────────────────────────────────
# CAMERA / VIDEO SOURCE  (threaded)
# ─────────────────────────────────────────────────────────────────

class FrameSource:
    def __init__(self, source):
        self._lock, self._frame, self._stop = threading.Lock(), None, False
        self.is_file = isinstance(source, str)
        if self.is_file:
            log.info(f"Opening video: {source}")
            self.cap = cv2.VideoCapture(source)
        else:
            log.info(f"Opening camera {source}...")
            self.cap = (cv2.VideoCapture(source, cv2.CAP_V4L2)
                        if platform.system() != "Windows"
                        else cv2.VideoCapture(source))
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        if not self.is_file:
            if platform.system() != "Windows":
                self.cap.set(cv2.CAP_PROP_FOURCC,
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.cap.read()
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CFG["CAM_W"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CFG["CAM_H"])
            self.cap.set(cv2.CAP_PROP_FPS,          CFG["CAM_FPS"])
            for _ in range(5):
                self.cap.read()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Source opened but cannot read frame")
        self._frame = frame
        log.info(f"Source ready: {frame.shape[1]}x{frame.shape[0]} ✓")
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
                time.sleep(0.005); continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return (self._frame is not None,
                    None if self._frame is None else self._frame.copy())

    def release(self):
        self._stop = True
        time.sleep(0.15)
        self.cap.release()

# ─────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────

def draw_seed_panel(bank: SeedBank, instance_counts: dict,
                    live_sims: dict, w=230):
    n    = len(bank)
    cell = 165
    h    = max(720, 40 + n * cell)
    p    = np.full((h, w, 3), (22, 22, 32), dtype=np.uint8)
    F    = CFG["FONT"]
    cv2.putText(p, "SEED BANK", (10, 24), F, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(p, (0, 30), (w, 30), (55, 55, 85), 1)
    for i in range(n):
        y0    = 40 + i * cell
        count = instance_counts.get(i, 0)
        found = count > 0
        col   = (0, 255, 0) if found else (90, 90, 110)
        th    = cv2.resize(bank.images_lr[i], (w - 20, 100))
        cv2.rectangle(p, (8, y0 - 2), (w - 8, y0 + 102), col, 2)
        p[y0:y0 + 100, 10:w - 10] = th
        cv2.putText(p, bank.names[i], (10, y0 + 118), F, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA)
        if found:
            cv2.putText(p, f"FOUND x{count}", (10, y0 + 138), F, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
        else:
            s = live_sims.get(i, 0.0)
            cv2.putText(p, f"sim:{s:.2f}" if s > 0 else "searching",
                        (10, y0 + 138), F, 0.42,
                        (0, 200, 255) if s > 0 else (90, 90, 90),
                        1, cv2.LINE_AA)
    return p


def draw_ctrl_panel(xyz: XYZListener, w=220):
    h = 720
    p = np.full((h, w, 3), (22, 22, 32), dtype=np.uint8)
    F = CFG["FONT"]
    y = 24
    cv2.putText(p, "CONTROLS", (10, y), F, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    y += 8; cv2.line(p, (0, y), (w, y), (55, 55, 85), 1); y += 22

    # Live XYZ readout
    x, yv, z = xyz.get()
    cv2.putText(p, "-- DRONE POS --", (10, y), F, 0.38,
                (150, 150, 200), 1, cv2.LINE_AA); y += 18
    cv2.putText(p, f"X: {x:+.3f} m", (10, y), F, 0.42,
                (0, 220, 120), 1, cv2.LINE_AA); y += 17
    cv2.putText(p, f"Y: {yv:+.3f} m", (10, y), F, 0.42,
                (0, 220, 120), 1, cv2.LINE_AA); y += 17
    cv2.putText(p, f"Z: {z:.3f} m",  (10, y), F, 0.42,
                (0, 220, 120), 1, cv2.LINE_AA); y += 22
    cv2.line(p, (0, y), (w, y), (45, 45, 65), 1); y += 14

    cv2.putText(p, "-- THRESHOLDS --", (10, y), F, 0.38,
                (150, 150, 200), 1, cv2.LINE_AA); y += 18
    n_ir = 360 // CFG["INFER_ROT_STEP"]
    n_sr = 360 // CFG["SEED_ROT_STEP"]
    for txt, keys in (
        (f"YOLO conf: {CFG['YOLO_CONF']:.2f}", "+/-"),
        (f"DINO sim:  {CFG['DINO_THRESH']:.2f}", "N/M"),
        (f"Cooldown:  {CFG['GLOBAL_COOLDOWN']:.0f}s", "fixed"),
        (f"Seed rots: {n_sr} ({CFG['SEED_ROT_STEP']}deg)", "fixed"),
        (f"Infer rots:{n_ir} ({CFG['INFER_ROT_STEP']}deg)", "fixed"),
    ):
        cv2.putText(p, txt,  (10, y), F, 0.38, (210, 210, 110), 1, cv2.LINE_AA); y += 14
        cv2.putText(p, f"   ({keys})", (10, y), F, 0.33,
                    (110, 110, 140), 1, cv2.LINE_AA); y += 18
    y += 4; cv2.line(p, (0, y), (w, y), (45, 45, 65), 1); y += 14
    cv2.putText(p, "-- KEYS --", (10, y), F, 0.38,
                (150, 150, 200), 1, cv2.LINE_AA); y += 18
    for k in ("Q  quit", "S  screenshot", "R  reset detections",
              "P  print thresholds"):
        cv2.putText(p, k, (10, y), F, 0.40, (160, 160, 160), 1, cv2.LINE_AA); y += 18
    return p


def draw_main(frame, dets, instance_counts, fps, bank, frame_n, cooldown_rem):
    out = frame.copy()
    H, W = out.shape[:2]
    F = CFG["FONT"]
    cv2.rectangle(out, (0, 0), (W, 56), (12, 12, 22), -1)
    fps_col = (0, 255, 0) if fps >= 10 else (0, 165, 255)
    cv2.putText(out, "IRoC-U 2026 v6", (10, 26), F, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"FPS:{fps:.0f}", (210, 26), F, 0.6, fps_col,         2, cv2.LINE_AA)
    cv2.putText(out, f"Saved:{sum(instance_counts.values())}", (310, 26),
                F, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"Frame:{frame_n}", (440, 26), F, 0.5,
                (120, 120, 120), 1, cv2.LINE_AA)
    if cooldown_rem > 0:
        cv2.putText(out, f"COOLDOWN: {cooldown_rem:.1f}s", (10, 50),
                    F, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
        bar_w = int((1.0 - cooldown_rem / CFG["GLOBAL_COOLDOWN"]) * 300)
        cv2.rectangle(out, (160, 40), (460, 54), (50, 50, 50), -1)
        cv2.rectangle(out, (160, 40), (160 + bar_w, 54), (0, 200, 255), -1)
    else:
        cv2.putText(out, "READY", (10, 50), F, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        sid = d.get("seed_id", -1)
        if sid >= 0:
            col = (0, 255, 0)
            cnt = instance_counts.get(sid, 0)
            lab = f"{bank.names[sid]}(x{cnt}) sim:{d['dino']:.2f}"
        else:
            col = (0, 165, 255)
            lab = f"low-sim:{d.get('dino', 0):.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        (tw, th2), _ = cv2.getTextSize(lab, F, 0.4, 1)
        ly = max(y1 - 6, th2 + 4)
        cv2.rectangle(out, (x1, ly - th2 - 4), (x1 + tw + 4, ly + 2), col, -1)
        cv2.putText(out, lab, (x1 + 2, ly - 2), F, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def stack_display(main, left, right):
    th = max(main.shape[0], left.shape[0], right.shape[0])
    def pad(img):
        d = th - img.shape[0]
        if d > 0:
            ext = np.full((d, img.shape[1], 3), (22, 22, 32), dtype=np.uint8)
            return np.vstack([img, ext])
        return img[:th]
    return np.hstack([pad(left), pad(main), pad(right)])

# ─────────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────────

def run(args):
    log.info("=" * 64)
    log.info("  IRoC-U 2026 — Rock Detector v6  (YOLO + DINOv2 + ZMQ)")
    log.info(f"  Seed rot {CFG['SEED_ROT_STEP']}°  |  Infer rot {CFG['INFER_ROT_STEP']}°")
    log.info(f"  ZMQ port {CFG['ZMQ_PORT']}  |  XYZ UDP :{CFG['XYZ_UDP_PORT']}")
    log.info("  One detection at a time | 4s global cooldown | R=reset")
    log.info("=" * 64)

    # ── Init subsystems ──────────────────────────────────────────
    xyz_listener = XYZListener(port=CFG["XYZ_UDP_PORT"])
    zmq_sender   = ZMQDetectionSender(
        port         = CFG["ZMQ_PORT"],
        discover     = CFG["ZMQ_DISCOVER"],
        connect_wait = CFG["ZMQ_CONNECT_WAIT"],
    )
    embedder = DinoEmbedder(CFG["DINO_MODEL"])
    bank     = SeedBank(args.refs, embedder)
    out_mgr  = OutputManager(bank, zmq_sender, xyz_listener)
    yolo     = load_yolo(CFG["YOLO_MODEL"])
    src      = FrameSource(args.video if args.video else args.cam)

    WIN = "IRoC-U 2026 v6"
    if CFG["SHOW_DISPLAY"]:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 1560, 720)

    log.info("\n[RUN] Q quit | S screenshot | R reset | P print-thresholds")
    log.info("+/- YOLO conf | N/M DINO thresh\n")

    fps_t:    List[float] = []
    live_sims: dict = {}
    frame_n = ss_n = 0

    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            time.sleep(0.003); continue
        frame_n += 1
        now = time.monotonic()
        fps_t.append(now)
        fps_t = [t for t in fps_t if now - t < 1.0]
        fps   = float(len(fps_t))

        H, W  = frame.shape[:2]
        dets  = []

        # ── STAGE 1: YOLO ────────────────────────────────────────
        boxes = run_yolo(yolo, frame)

        frame_candidates = []

        for (x1, y1, x2, y2, yc) in boxes:

            # ── STAGE 2: crop ────────────────────────────────────
            px1 = max(0, x1 - CFG["CROP_PAD"]); py1 = max(0, y1 - CFG["CROP_PAD"])
            px2 = min(W, x2 + CFG["CROP_PAD"]); py2 = min(H, y2 + CFG["CROP_PAD"])
            if px2 - px1 < 12 or py2 - py1 < 12:
                continue
            dino_input = cv2.resize(frame[py1:py2, px1:px2],
                                    (CFG["CROP_SIZE"], CFG["CROP_SIZE"]))
            if dino_input.size == 0:
                continue

            # ── STAGE 3: DINOv2 rotation-invariant match ─────────
            rot_embs = embedder.embed_rotations(dino_input, CFG["INFER_ROT_STEP"])
            seed_id, dino_sim = bank.best_match_multi(rot_embs)

            for s_i, s_v in enumerate(bank.sim_per_seed_multi(rot_embs)):
                live_sims[s_i] = max(live_sims.get(s_i, 0.0), s_v)

            if dino_sim < CFG["DINO_THRESH"]:
                dets.append({"bbox": (x1, y1, x2, y2),
                             "seed_id": -1, "dino": dino_sim})
                continue

            dets.append({"bbox": (x1, y1, x2, y2),
                         "seed_id": seed_id, "dino": dino_sim})
            frame_candidates.append((dino_sim, seed_id, (x1, y1, x2, y2)))

        # ── Save + Send: single best candidate, once per cooldown ─
        if frame_candidates and out_mgr.global_ready():
            frame_candidates.sort(key=lambda t: t[0], reverse=True)
            best_sim, best_sid, best_bbox = frame_candidates[0]
            inst = out_mgr.instance_count.get(best_sid, 0) + 1
            log.info("\n" + "*" * 52 +
                     f"\n  INSTANCE #{inst}: {bank.names[best_sid]}"
                     f"\n  bbox={best_bbox}  dino_sim={best_sim:.3f}"
                     f"\n  drone xyz={xyz_listener.get()}"
                     f"\n" + "*" * 52)
            # save full 1280×720 frame to disk AND enqueue ZMQ send
            out_mgr.save_and_send(best_sid, best_bbox, frame, best_sim)

        # ── DISPLAY ──────────────────────────────────────────────
        if CFG["SHOW_DISPLAY"]:
            disp = stack_display(
                draw_main(frame, dets, out_mgr.instance_count,
                          fps, bank, frame_n,
                          out_mgr.cooldown_remaining()),
                draw_seed_panel(bank, out_mgr.instance_count, live_sims),
                draw_ctrl_panel(xyz_listener))
            cv2.imshow(WIN, disp)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                fn = f"screenshot_{ss_n:03d}.jpg"
                cv2.imwrite(fn, disp); ss_n += 1
                log.info(f"Screenshot: {fn}")
            elif k == ord('r'):
                out_mgr.reset(); live_sims.clear()
            elif k == ord('p'):
                log.info(f"THRESHOLDS  yolo={CFG['YOLO_CONF']:.2f}"
                         f"  dino={CFG['DINO_THRESH']:.2f}"
                         f"  cooldown={CFG['GLOBAL_COOLDOWN']:.0f}s"
                         f"  seed_rot={CFG['SEED_ROT_STEP']}°"
                         f"  infer_rot={CFG['INFER_ROT_STEP']}°")
            elif k in (ord('+'), ord('=')):
                CFG["YOLO_CONF"] = min(0.95, round(CFG["YOLO_CONF"] + 0.05, 2))
                log.info(f"YOLO conf → {CFG['YOLO_CONF']:.2f}")
            elif k == ord('-'):
                CFG["YOLO_CONF"] = max(0.05, round(CFG["YOLO_CONF"] - 0.05, 2))
                log.info(f"YOLO conf → {CFG['YOLO_CONF']:.2f}")
            elif k == ord('n'):
                CFG["DINO_THRESH"] = min(0.95, round(CFG["DINO_THRESH"] + 0.05, 2))
                log.info(f"DINO thresh → {CFG['DINO_THRESH']:.2f}")
            elif k == ord('m'):
                CFG["DINO_THRESH"] = max(0.10, round(CFG["DINO_THRESH"] - 0.05, 2))
                log.info(f"DINO thresh → {CFG['DINO_THRESH']:.2f}")

    src.release()
    if CFG["SHOW_DISPLAY"]:
        cv2.destroyAllWindows()
    zmq_sender.close()
    xyz_listener.stop()
    out_mgr.summary()
    log.info("[Done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam",            type=int,   default=0)
    ap.add_argument("--video",          type=str,   default=None)
    ap.add_argument("--refs",           type=str,   default="refs")
    ap.add_argument("--yolo",           type=str,   default=CFG["YOLO_MODEL"])
    ap.add_argument("--conf",           type=float, default=CFG["YOLO_CONF"])
    ap.add_argument("--dino-thresh",    type=float, default=CFG["DINO_THRESH"])
    ap.add_argument("--cooldown",       type=float, default=CFG["GLOBAL_COOLDOWN"])
    ap.add_argument("--seed-rot-step",  type=int,   default=CFG["SEED_ROT_STEP"])
    ap.add_argument("--infer-rot-step", type=int,   default=CFG["INFER_ROT_STEP"])
    ap.add_argument("--zmq-port",       type=int,   default=CFG["ZMQ_PORT"])
    ap.add_argument("--xyz-port",       type=int,   default=CFG["XYZ_UDP_PORT"])
    ap.add_argument("--no-display",     action="store_true")
    args = ap.parse_args()

    CFG["YOLO_MODEL"]      = args.yolo
    CFG["YOLO_CONF"]       = args.conf
    CFG["DINO_THRESH"]     = args.dino_thresh
    CFG["GLOBAL_COOLDOWN"] = args.cooldown
    CFG["SEED_ROT_STEP"]   = args.seed_rot_step
    CFG["INFER_ROT_STEP"]  = args.infer_rot_step
    CFG["ZMQ_PORT"]        = args.zmq_port
    CFG["XYZ_UDP_PORT"]    = args.xyz_port
    if args.no_display:
        CFG["SHOW_DISPLAY"] = False
    run(args)