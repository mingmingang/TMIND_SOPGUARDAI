"""
V2.8b SOP - MediaPipe Holistic Pose + Bearing Sequence - T-MIND Project
==========================================================================
Evolusi dari V2.6b: YOLOv8-Pose DIGANTI dengan MediaPipe Holistic
untuk pose estimation yang lebih detail (pose + tangan + wajah).

Referensi MediaPipe: Mediapipe_fromFile_Video.py
  - OpenCL (AMD iGPU) untuk pre-processing (resize + cvtColor)
  - MediaPipe Holistic inference di CPU (Ryzen)
  - Deteksi: Pose landmarks (33 titik) + tangan kiri/kanan (21 titik)

Yang TETAP sama dari V2.6b:
  1. ArUco marker → identifikasi operator
  2. YOLO custom model → deteksi objek (bearing) dalam ROI
  3. Sequence Tracker → SOP urutan perakitan bearing
  4. Work Area graphic: L-shape corner markers + reference lines
  5. ROI Manager (draw, save, load)
  6. OperatorTracker (memory tracking saat ArUco hilang)

Yang BERUBAH:
  - YOLOPoseDetector (YOLO 17 keypoints) → MediaPipe Holistic
    (33 pose landmarks + 21 tangan kiri + 21 tangan kanan)
  - Person detection kini via bounding box dari pose_landmarks MediaPipe
  - Pre-processing menggunakan UMat OpenCL (AMD iGPU)

Kontrol:
  --- ROI & Detection ---
  Klik + Drag  = Gambar zona ROI baru
  ENTER        = Toggle SETUP / DETECT mode
  'c'          = Clear semua ROI
  'd'          = Delete ROI terakhir
  's'          = Save konfigurasi ROI + WorkArea ke JSON
  'l'          = Load konfigurasi ROI + WorkArea dari JSON
  'm'          = Ganti containment mode (center / overlap / full)
  'q' / ESC    = Keluar
  'r'          = Reset Sequence Tracker

  --- Pose Detection ---
  'p'          = Toggle Pose ON/OFF
  'o'          = Toggle Skeleton style (full/minimal/hands)

  --- Work Area ---
  'w'          = Mode taruh L-corner (klik di frame)
  'h'          = Mode taruh garis referensi Horizontal
  'v'          = Mode taruh garis referensi Vertikal
  'z'          = Kembali ke IDLE (ROI draw mode)
  'x'          = Clear semua Work Area graphics
  'n'          = Undo corner / line terakhir
"""

import os
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2
import cv2.aruco as aruco
import numpy as np
import mediapipe as mp
import time
import sys
import json
import threading
import psutil
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# SYSTEM MONITOR (CPU/RAM)
# ============================================================
class SystemMonitor:
    def __init__(self):
        self.cpu_percent = 0.0
        self.ram_percent = 0.0
        self.stopped = False
    
    def start(self):
        threading.Thread(target=self._update_loop, daemon=True).start()
        return self

    def _update_loop(self):
        while not self.stopped:
            self.cpu_percent = psutil.cpu_percent(interval=1.0)
            self.ram_percent = psutil.virtual_memory().percent

    def stop(self):
        self.stopped = True

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# OPTIMASI HARDWARE: AMD iGPU (OpenCL) + CPU (Ryzen)
# ============================================================

# [CPU] Maksimalkan thread untuk OpenCV dan MediaPipe
cv2.setNumThreads(cv2.getNumberOfCPUs())

# [AMD iGPU] Aktifkan OpenCL — OpenCV pakai Radeon Graphics
# untuk operasi resize & cvtColor via UMat
ocl_available = cv2.ocl.haveOpenCL()
cv2.ocl.setUseOpenCL(ocl_available)
if ocl_available:
    _ocl_dev = cv2.ocl.Device.getDefault()
    print(f"[INFO] OpenCL aktif → {_ocl_dev.name()} (iGPU AMD)")
    print(f"[INFO] Pre-processing (resize + cvtColor) akan dikerjakan iGPU")
else:
    print("[INFO] OpenCL tidak tersedia, semua dikerjakan CPU")
print(f"[INFO] MediaPipe inference → CPU ({cv2.getNumberOfCPUs()} thread)")


# ============================================================
# CONFIGURATION
# ============================================================

# -- File Video ---
#CAMERA_INDEX = r"D:\Documents HDD\Kuli-ah\comp\T-MIND\03 Prototype\Video Pokeb\Video Testing Dance.mp4"

# --- Kamera ---
CAMERA_INDEX  = 1          # 0 = default webcam, 1 = external

# --- DroidCam (comment-out, aktifkan jika pakai IP cam / DroidCam) ---
#CAMERA_INDEX  = "http://10.54.50.236:4747/video"

# --- Resolusi (hanya berlaku untuk kamera, video file pakai resolusi asli) ---
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

# --- MediaPipe Holistic config ---
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF  = 0.5
# Lebar frame untuk processing MediaPipe (lebih kecil = lebih cepat)
MP_PROCESS_WIDTH = 640

# --- YOLO Object Detection (bearing, kotak susu, dll) ---
CUSTOM_MODEL  = "Training bearing dummy V3.pt"
COCO_MODEL    = "yolo11n.pt"
DETECT_MODE_MODEL = "custom"
YOLO_CONF     = 0.40
YOLO_CLASSES  = None

# --- Pose Draw Style ---
ENABLE_POSE   = True
POSE_DRAW_STYLE = "full"    # "full" | "minimal" | "hands"

# --- ArUco Detection ---
ARUCO_DICT_TYPE = "4x4_50"

# --- Mapping ID → Nama Operator ---
ID_MAP = {
    0: "Arya Dwi Kusuma",
    1: "Person B",
    2: "Person C",
    3: "Person D",
    4: "Person E",
}
DEFAULT_NAME = "Unknown"

# --- Target Operator ---
TARGET_OPERATOR_ID = None     # None = semua, 0 = hanya ID 0, dst.

# --- ROI ---
ROI_COLOR          = (0,   230, 255)
ROI_ACTIVE_COLOR   = (0,   255, 100)
ROI_FILL_ALPHA     = 0.12
ROI_SAVE_FILE      = "roi_config.json"

# --- Warna ---
YOLO_BBOX_COLOR      = (255, 180, 0)
OPERATOR_BBOX_COLOR  = (0, 255, 100)
ARUCO_COLOR          = (0, 255, 255)
SKELETON_POSE_LM     = (245, 117, 66)
SKELETON_POSE_CN     = (245,  66, 230)
SKELETON_RHAND_LM    = ( 80,  22,  10)
SKELETON_RHAND_CN    = ( 80,  44, 121)
SKELETON_LHAND_LM    = (121,  22,  76)
SKELETON_LHAND_CN    = (121,  44, 250)

# --- Work Area Graphic ---
WORK_AREA_COLOR       = (255, 200,  50)
WORK_AREA_LINE_COLOR  = (200, 100, 255)
WORK_AREA_L_SIZE      = 30
WORK_AREA_L_THICKNESS = 3
WORK_AREA_LINE_THICK  = 1

# ============================================================

# ArUco dictionary mapping
ARUCO_DICT_MAP = {
    "4x4_50":  cv2.aruco.DICT_4X4_50,
    "4x4_100": cv2.aruco.DICT_4X4_100,
    "4x4_250": cv2.aruco.DICT_4X4_250,
    "5x5_50":  cv2.aruco.DICT_5X5_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
    "6x6_50":  cv2.aruco.DICT_6X6_50,
    "6x6_250": cv2.aruco.DICT_6X6_250,
    "7x7_50":  cv2.aruco.DICT_7X7_50,
}

# MediaPipe solutions
mp_drawing  = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh


# ============================================================
# MULTITHREADING CAMERA / VIDEO FILE
# ============================================================
class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        # Jika src adalah index kamera (angka), gunakan cv2.CAP_DSHOW (DirecShow) agar Logitech bisa terbuka di Windows
        if isinstance(src, int) and os.name == 'nt':
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        # Deteksi apakah sumber = file video
        self.is_file = isinstance(src, str) and not src.startswith("http")

        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) if self.is_file else 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.is_file else 0

        src_type = "Video File" if self.is_file else "Kamera"
        print(f"[INFO] {src_type}: {self.actual_w}x{self.actual_h}")
        if self.is_file:
            print(f"[INFO] Video FPS: {self.video_fps:.1f} | Total frames: {self.total_frames}")

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.stopped = True
                return
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.stopped = True
                        return
                    print("\n[INFO] Video selesai, loop dari awal.")
                else:
                    self.stopped = True
                    return
            with self._lock:
                self.ret, self.frame = ret, frame

            if self.is_file and self.video_fps > 0:
                time.sleep(1.0 / self.video_fps)

    def read(self):
        with self._lock:
            if self.ret and self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()


# ============================================================
# MEDIAPIPE HOLISTIC POSE DETECTOR
# (Pengganti YOLOPoseDetector dari V2.6b)
# ============================================================
class MediaPipePoseDetector:
    """
    MediaPipe Holistic: pose landmarks (33 titik) + tangan kiri/kanan (21 titik).
    Pre-processing (resize + cvtColor) menggunakan UMat OpenCL (AMD iGPU).
    Inference dilakukan di CPU (Ryzen).

    Output 'persons' berisi:
      - bbox: (x1, y1, x2, y2) dari pose_landmarks (frame asli)
      - pose_landmarks: hasil MediaPipe holistic.process()
      - right_hand_landmarks
      - left_hand_landmarks
    """

    def __init__(self,
                 min_detection_confidence=MP_MIN_DETECTION_CONF,
                 min_tracking_confidence=MP_MIN_TRACKING_CONF,
                 process_width=MP_PROCESS_WIDTH):
        self.process_width   = process_width
        self.holistic = mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        print(f"[INFO] MediaPipe Holistic init (process_width={process_width})")

    def _preprocess(self, frame):
        """
        Resize + cvtColor menggunakan UMat agar berjalan di AMD iGPU via OpenCL.
        Kembalikan array numpy RGB kecil untuk MediaPipe.
        """
        h, w = frame.shape[:2]
        scale = self.process_width / w
        umat  = cv2.UMat(frame)
        small = cv2.resize(umat,
                           (self.process_width, int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)          # iGPU
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)          # iGPU
        return small_rgb.get(), scale                               # → numpy CPU

    def detect(self, frame):
        """
        Proses satu frame.
        Return: list of person-dict  (bisa 0 atau 1 orang per hasil holistic).
        """
        h, w = frame.shape[:2]
        small_rgb, scale = self._preprocess(frame)

        small_rgb.flags.writeable = False
        results = self.holistic.process(small_rgb)
        small_rgb.flags.writeable = True

        persons = []

        if results.pose_landmarks is None:
            return persons   # Tidak ada orang terdeteksi

        # ── Hitung bounding box dari pose_landmarks di frame ASLI ──
        xs, ys = [], []
        for lm in results.pose_landmarks.landmark:
            xs.append(lm.x * w)
            ys.append(lm.y * h)

        x1 = max(0, int(min(xs)) - 20)
        y1 = max(0, int(min(ys)) - 20)
        x2 = min(w, int(max(xs)) + 20)
        y2 = min(h, int(max(ys)) + 20)

        persons.append({
            'bbox':                  (x1, y1, x2, y2),
            'conf':                  1.0,              # MediaPipe tidak memberi confidence
            'pose_landmarks':        results.pose_landmarks,
            'right_hand_landmarks':  results.right_hand_landmarks,
            'left_hand_landmarks':   results.left_hand_landmarks,
            # keypoints dummy None agar OperatorTracker tidak crash
            'keypoints':             None,
        })

        return persons

    def close(self):
        self.holistic.close()


# ============================================================
# ARUCO DETECTOR
# ============================================================
class ArUcoDetector:
    def __init__(self, dict_type=ARUCO_DICT_TYPE):
        dict_key = ARUCO_DICT_MAP.get(dict_type, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_key)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def detect(self, frame):
        """Detect ArUco markers. Returns list of {id, name, corners, center}."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        results = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                pts = corners[i][0]
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                name = ID_MAP.get(int(marker_id), DEFAULT_NAME)
                results.append({
                    'id': int(marker_id),
                    'name': name,
                    'corners': pts,
                    'center': (cx, cy),
                })
        return results


# ============================================================
# MATCHING & TRACKING: ArUco → Person
# ============================================================
def match_aruco_to_person(aruco_results, person_results):
    """Match setiap ArUco marker ke person bbox terkecil yang mengandung center marker."""
    matches = []
    for ar in aruco_results:
        if TARGET_OPERATOR_ID is not None and ar['id'] != TARGET_OPERATOR_ID:
            continue
        ax, ay = ar['center']
        best_person = None
        best_area = float('inf')
        for p in person_results:
            x1, y1, x2, y2 = p['bbox']
            if x1 <= ax <= x2 and y1 <= ay <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best_area = area
                    best_person = p
        if best_person is not None:
            matches.append({'person': best_person, 'aruco': ar})
    return matches


class OperatorTracker:
    """Tracking memory: jaga identitas operator saat ArUco hilang sementara."""

    def __init__(self, max_lost_frames=15, distance_threshold=250):
        self.max_lost_frames = max_lost_frames
        self.distance_threshold = distance_threshold
        self.active_operators = {}

    def update(self, persons, aruco_results):
        current_matches = match_aruco_to_person(aruco_results, persons)
        matched_aruco_ids = set()
        matched_person_bboxes = set(tuple(m['person']['bbox']) for m in current_matches)

        for match in current_matches:
            aid = match['aruco']['id']
            self.active_operators[aid] = {
                'person': match['person'],
                'aruco': match['aruco'],
                'lost_frames': 0
            }
            matched_aruco_ids.add(aid)

        keys_to_remove = []
        for aid, data in self.active_operators.items():
            if aid in matched_aruco_ids:
                continue
            data['lost_frames'] += 1
            if data['lost_frames'] > self.max_lost_frames:
                keys_to_remove.append(aid)
                continue

            last_bbox = data['person']['bbox']
            last_cx = (last_bbox[0] + last_bbox[2]) // 2
            last_cy = (last_bbox[1] + last_bbox[3]) // 2

            best_person = None
            best_dist = float('inf')
            for p in persons:
                p_bbox = tuple(p['bbox'])
                if p_bbox in matched_person_bboxes:
                    continue
                pcx = (p_bbox[0] + p_bbox[2]) // 2
                pcy = (p_bbox[1] + p_bbox[3]) // 2
                dist = ((pcx - last_cx)**2 + (pcy - last_cy)**2) ** 0.5
                if dist < best_dist and dist < self.distance_threshold:
                    best_dist = dist
                    best_person = p

            if best_person is not None:
                data['person'] = best_person
                matched_person_bboxes.add(tuple(best_person['bbox']))

        for aid in keys_to_remove:
            del self.active_operators[aid]

        output = []
        for aid, data in self.active_operators.items():
            output.append({
                'person': data['person'],
                'aruco': data['aruco'],
                'is_lost': data['lost_frames'] > 0
            })
        return output


# ============================================================
# ROI DATA CLASS
# ============================================================
@dataclass
class ROIZone:
    x1: int
    y1: int
    x2: int
    y2: int
    name: str = ""
    color: Tuple[int, int, int] = field(default_factory=lambda: (0, 230, 255))

    def contains_point(self, px: int, py: int) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2

    def contains_bbox(self, bx1: int, by1: int, bx2: int, by2: int,
                      mode: str = "center") -> bool:
        if mode == "center":
            cx = (bx1 + bx2) // 2
            cy = (by1 + by2) // 2
            return self.contains_point(cx, cy)
        elif mode == "overlap":
            return not (bx2 < self.x1 or bx1 > self.x2 or
                        by2 < self.y1 or by1 > self.y2)
        elif mode == "full":
            return (bx1 >= self.x1 and by1 >= self.y1 and
                    bx2 <= self.x2 and by2 <= self.y2)
        return False

    def to_dict(self) -> dict:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2,
                "name": self.name}

    @classmethod
    def from_dict(cls, d: dict) -> "ROIZone":
        return cls(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                   name=d.get("name", ""))


# ============================================================
# WORK AREA: L-CORNER & REFERENCE LINE
# ============================================================
@dataclass
class LCorner:
    x: int
    y: int
    orientation: str = "TL"

    def draw(self, frame, color=WORK_AREA_COLOR,
             size=WORK_AREA_L_SIZE, thickness=WORK_AREA_L_THICKNESS):
        x, y = self.x, self.y
        s = size
        dirs = {
            "TL": (+s,  0,  0, +s), "TR": (-s,  0,  0, +s),
            "BL": (+s,  0,  0, -s), "BR": (-s,  0,  0, -s),
        }.get(self.orientation, (+s, 0, 0, +s))
        dx1, dy1, dx2, dy2 = dirs
        cv2.line(frame, (x, y), (x + dx1, y + dy1), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x + dx2, y + dy2), color, thickness, cv2.LINE_AA)
        cv2.circle(frame, (x, y), thickness + 1, color, -1, cv2.LINE_AA)

    def to_dict(self):
        return {"x": self.x, "y": self.y, "orientation": self.orientation}

    @classmethod
    def from_dict(cls, d):
        return cls(x=d["x"], y=d["y"], orientation=d.get("orientation", "TL"))


@dataclass
class RefLine:
    axis: str
    pos:  int
    label: str = ""

    def draw(self, frame, color=WORK_AREA_LINE_COLOR,
             thickness=WORK_AREA_LINE_THICK):
        h, w = frame.shape[:2]
        if self.axis == "H":
            cv2.line(frame, (0, self.pos), (w, self.pos), color, thickness, cv2.LINE_AA)
            lbl = self.label if self.label else f"y={self.pos}"
            cv2.putText(frame, lbl, (w - 90, self.pos - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        else:
            cv2.line(frame, (self.pos, 0), (self.pos, h), color, thickness, cv2.LINE_AA)
            lbl = self.label if self.label else f"x={self.pos}"
            cv2.putText(frame, lbl, (self.pos + 4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    def to_dict(self):
        return {"axis": self.axis, "pos": self.pos, "label": self.label}

    @classmethod
    def from_dict(cls, d):
        return cls(axis=d["axis"], pos=d["pos"], label=d.get("label", ""))


# ============================================================
# WORK AREA MANAGER
# ============================================================
_CORNER_CYCLE = ["TL", "TR", "BR", "BL"]

class WorkAreaManager:
    def __init__(self):
        self.corners:   List[LCorner] = []
        self.ref_lines: List[RefLine] = []
        self.sub_mode   = "IDLE"
        self._preview   = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self._preview = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.sub_mode == "PLACE_CORNER":
                orientation = _CORNER_CYCLE[len(self.corners) % 4]
                self.corners.append(LCorner(x=x, y=y, orientation=orientation))
                print(f"[WorkArea] Corner {orientation} @ ({x},{y})")
            elif self.sub_mode == "PLACE_HLINE":
                lbl = f"H{len([l for l in self.ref_lines if l.axis=='H'])+1}"
                self.ref_lines.append(RefLine(axis="H", pos=y, label=lbl))
                print(f"[WorkArea] H-line '{lbl}' @ y={y}")
            elif self.sub_mode == "PLACE_VLINE":
                lbl = f"V{len([l for l in self.ref_lines if l.axis=='V'])+1}"
                self.ref_lines.append(RefLine(axis="V", pos=x, label=lbl))
                print(f"[WorkArea] V-line '{lbl}' @ x={x}")

    def draw(self, frame):
        for line in self.ref_lines:
            line.draw(frame)
        for corner in self.corners:
            corner.draw(frame)
        if self.sub_mode != "IDLE" and self._preview:
            px, py = self._preview
            h, w   = frame.shape[:2]
            ghost_col = (255, 255, 100)
            overlay = frame.copy()
            if self.sub_mode == "PLACE_CORNER":
                orientation = _CORNER_CYCLE[len(self.corners) % 4]
                LCorner(px, py, orientation).draw(overlay, color=ghost_col, thickness=2)
            elif self.sub_mode == "PLACE_HLINE":
                cv2.line(overlay, (0, py), (w, py), ghost_col, 1, cv2.LINE_AA)
            elif self.sub_mode == "PLACE_VLINE":
                cv2.line(overlay, (px, 0), (px, h), ghost_col, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            hint_map = {
                "PLACE_CORNER": f"[WorkArea] Klik: taruh corner {_CORNER_CYCLE[len(self.corners) % 4]}",
                "PLACE_HLINE":  "[WorkArea] Klik: taruh H-line",
                "PLACE_VLINE":  "[WorkArea] Klik: taruh V-line",
            }
            cv2.putText(frame, hint_map.get(self.sub_mode, ""),
                        (10, frame.shape[0] - 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 240, 80), 1, cv2.LINE_AA)

    def delete_last_corner(self):
        if self.corners:
            c = self.corners.pop()
            print(f"[WorkArea] Corner {c.orientation} dihapus.")
        else:
            print("[WorkArea] Tidak ada corner.")

    def delete_last_line(self):
        if self.ref_lines:
            l = self.ref_lines.pop()
            print(f"[WorkArea] {l.axis}-line '{l.label}' dihapus.")
        else:
            print("[WorkArea] Tidak ada reference line.")

    def clear(self):
        self.corners.clear()
        self.ref_lines.clear()
        print("[WorkArea] Semua graphic dihapus.")

    def save_to_dict(self) -> dict:
        return {
            "corners":   [c.to_dict() for c in self.corners],
            "ref_lines": [l.to_dict() for l in self.ref_lines],
        }

    def load_from_dict(self, d: dict):
        self.corners   = [LCorner.from_dict(c) for c in d.get("corners",   [])]
        self.ref_lines = [RefLine.from_dict(l)  for l in d.get("ref_lines", [])]
        print(f"[WorkArea] Loaded {len(self.corners)} corners, {len(self.ref_lines)} ref lines.")


# ============================================================
# PALETTE WARNA ROI
# ============================================================
ROI_PALETTE = [
    (0,   230, 255), (255, 100,  50), (180,  50, 255),
    (50,  255, 150), (255, 200,  30), (50,  150, 255), (255,  50, 150),
]


# ============================================================
# YOLO DUAL-MODEL DETECTOR (untuk objek/bearing — dari V2.4)
# ============================================================
class YOLODetector:
    def __init__(self, detect_mode: str = DETECT_MODE_MODEL,
                 conf: float = YOLO_CONF, classes=YOLO_CLASSES):
        self.detect_mode = detect_mode
        self.conf        = conf
        self.classes     = classes
        self.model_custom = None
        self.model_coco   = None
        self.device_used = "CPU"

        script_dir   = os.path.dirname(os.path.abspath(__file__))
        custom_path  = os.path.join(script_dir, CUSTOM_MODEL)

        if detect_mode in ("custom", "both"):
            if os.path.exists(custom_path):
                self.model_custom = self._load_yolo_model(custom_path)
            else:
                print(f"[WARN] Custom model tidak ditemukan: '{custom_path}'")
                if detect_mode == "custom":
                    print("[WARN] Fallback ke yolov8n.pt (COCO)")
                    self.model_coco = self._load_yolo_model(COCO_MODEL)

        if detect_mode in ("coco", "both") or \
           (detect_mode == "custom" and self.model_custom is None):
            self.model_coco = self._load_yolo_model(COCO_MODEL)

    def _load_yolo_model(self, model_path):
        from ultralytics import YOLO
        import os
        import shutil

        model_name = os.path.splitext(os.path.basename(model_path))[0]
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'DmlExecutionProvider' in providers:
                self.device_used = "iGPU (DirectML)"
                onnx_path = f"{model_name}_amd_opt.onnx"
                if not os.path.exists(onnx_path):
                    print(f"[INFO] Exporting {model_name} ke ONNX (FP16 + Simplify untuk AMD iGPU)...")
                    temp_model = YOLO(model_path)
                    exported_file = temp_model.export(
                        format="onnx", imgsz=640,
                        half=True, simplify=True
                    )
                    if os.path.exists(exported_file):
                        shutil.move(exported_file, onnx_path)
                
                print(f"[INFO] Loading YOLO model via AMD iGPU (DirectML FP16): {onnx_path}")
                return YOLO(onnx_path, task='detect')
        except ImportError:
            pass

        print(f"[INFO] Loading YOLO model via CPU/Standard Backend: {model_path}")
        return YOLO(model_path)

    def _run_model(self, model, frame) -> list:
        results = model(frame, conf=self.conf, classes=self.classes, verbose=False)
        boxes = []
        names = model.names
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                boxes.append({
                    "bbox":       (x1, y1, x2, y2),
                    "conf":       float(box.conf[0]),
                    "class_id":   int(box.cls[0]),
                    "class_name": names.get(int(box.cls[0]), str(int(box.cls[0]))),
                    "source":     "custom" if model is self.model_custom else "coco",
                })
        return boxes

    def detect(self, frame, roi_zones: List[ROIZone],
               iou_mode: str = "center") -> List[dict]:
        if not roi_zones:
            return []
        raw = []
        if self.model_custom:
            raw.extend(self._run_model(self.model_custom, frame))
        if self.model_coco:
            raw.extend(self._run_model(self.model_coco, frame))

        detections = []
        for det in raw:
            x1, y1, x2, y2 = det["bbox"]
            inside_roi = -1
            for i, roi in enumerate(roi_zones):
                if roi.contains_bbox(x1, y1, x2, y2, mode=iou_mode):
                    inside_roi = i
                    break
            if inside_roi >= 0:
                det["roi_idx"] = inside_roi
                detections.append(det)
        return detections


# ============================================================
# SEQUENCE TRACKER (dari V2.4)
# ============================================================
SEQUENCE_STEPS = [
    {"action": "INIT",  "desc": "Siapkan Storage Penuh, Jig Kosong"},
    {"action": "TAKE",  "roi_idx": 0, "desc": "Ambil Storage #1"},
    {"action": "TAKE",  "roi_idx": 2, "desc": "Ambil Storage #3"},
    {"action": "TAKE",  "roi_idx": 4, "desc": "Ambil Storage #5"},
    {"action": "PLACE", "roi_idx": 9, "desc": "Taruh Jig #5 (Kanan)"},
    {"action": "PLACE", "roi_idx": 7, "desc": "Taruh Jig #3 (Tengah)"},
    {"action": "PLACE", "roi_idx": 5, "desc": "Taruh Jig #1 (Kiri)"},
    {"action": "TAKE",  "roi_idx": 1, "desc": "Ambil Storage #2"},
    {"action": "TAKE",  "roi_idx": 3, "desc": "Ambil Storage #4"},
    {"action": "PLACE", "roi_idx": 8, "desc": "Taruh Jig #4"},
    {"action": "PLACE", "roi_idx": 6, "desc": "Taruh Jig #2"},
    {"action": "DONE",  "desc": "Perakitan Selesai!"}
]

def get_roi_name(idx):
    return f"Storage #{idx + 1}" if idx < 5 else f"Jig #{idx - 5 + 1}"

class SequenceTracker:
    def __init__(self, num_rois=10, debounce_frames=5):
        self.num_rois = num_rois
        self.debounce_frames = debounce_frames
        self.roi_history = [0] * num_rois
        self.roi_states = [False] * num_rois
        self.expected_states = [False] * num_rois
        self.current_step = 0
        self.error_msg = ""

    def reset(self):
        self.roi_history = [0] * self.num_rois
        self.roi_states = [False] * self.num_rois
        self.expected_states = [False] * self.num_rois
        self.current_step = 0
        self.error_msg = ""

    def update(self, detected_roi_indices: List[int]):
        for i in range(self.num_rois):
            if i in detected_roi_indices:
                self.roi_history[i] = min(self.roi_history[i] + 1, self.debounce_frames * 2)
            else:
                self.roi_history[i] = max(self.roi_history[i] - 1, 0)
            if self.roi_history[i] >= self.debounce_frames:
                self.roi_states[i] = True
            elif self.roi_history[i] == 0:
                self.roi_states[i] = False

        if self.current_step >= len(SEQUENCE_STEPS):
            self.error_msg = ""
            return

        step = SEQUENCE_STEPS[self.current_step]
        action = step["action"]

        if action == "INIT":
            if all(self.roi_states[0:5]) and not any(self.roi_states[5:10]):
                self.current_step += 1
                self.expected_states = self.roi_states.copy()
                self.error_msg = ""
            else:
                self.error_msg = "Menunggu Storage 1-5 Penuh & Jig 1-5 Kosong"
        elif action == "DONE":
            self.error_msg = ""
        else:
            roi_idx = step.get("roi_idx", -1)
            target_state = (action == "PLACE")
            wrong_actions = []
            for i in range(self.num_rois):
                if i != roi_idx:
                    if self.roi_states[i] != self.expected_states[i]:
                        wrong_actions.append(i)
            if wrong_actions:
                names = [get_roi_name(w) for w in wrong_actions]
                self.error_msg = f"SALAH URUTAN! Cek: {', '.join(names)}"
            else:
                self.error_msg = f"Instruksi: {step['desc']}"
                if self.roi_states[roi_idx] == target_state:
                    self.current_step += 1
                    self.expected_states[roi_idx] = target_state


# ============================================================
# ROI MANAGER
# ============================================================
class ROIManager:
    def __init__(self):
        self.zones: List[ROIZone] = []
        self._drawing    = False
        self._start_pt   = None
        self._current_pt = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing    = True
            self._start_pt   = (x, y)
            self._current_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._drawing:
                self._current_pt = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._drawing and self._start_pt:
                self._drawing = False
                x1 = min(self._start_pt[0], x)
                y1 = min(self._start_pt[1], y)
                x2 = max(self._start_pt[0], x)
                y2 = max(self._start_pt[1], y)
                if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                    idx   = len(self.zones)
                    color = ROI_PALETTE[idx % len(ROI_PALETTE)]
                    name  = f"ROI-{idx + 1}"
                    self.zones.append(ROIZone(x1, y1, x2, y2, name=name, color=color))
                    print(f"[ROI] Ditambah: {name} ({x1},{y1})-({x2},{y2})")
                self._start_pt  = None
                self._current_pt = None

    def draw_in_progress(self, frame):
        if self._drawing and self._start_pt and self._current_pt:
            x1 = min(self._start_pt[0], self._current_pt[0])
            y1 = min(self._start_pt[1], self._current_pt[1])
            x2 = max(self._start_pt[0], self._current_pt[0])
            y2 = max(self._start_pt[1], self._current_pt[1])
            cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_ACTIVE_COLOR, 2)
            w_px = x2 - x1
            h_px = y2 - y1
            cv2.putText(frame, f"{w_px}x{h_px}px",
                        (x1 + 4, y1 - 6 if y1 > 16 else y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROI_ACTIVE_COLOR, 1, cv2.LINE_AA)

    def draw_zones(self, frame):
        for i, roi in enumerate(self.zones):
            color = roi.color
            overlay = frame.copy()
            cv2.rectangle(overlay, (roi.x1, roi.y1), (roi.x2, roi.y2), color, -1)
            cv2.addWeighted(overlay, ROI_FILL_ALPHA, frame, 1 - ROI_FILL_ALPHA, 0, frame)
            cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), color, 2)
            cl = min(20, (roi.x2 - roi.x1) // 5)
            ct = 3
            for cx, cy, dx, dy in [
                (roi.x1, roi.y1,  cl,  cl), (roi.x2, roi.y1, -cl,  cl),
                (roi.x1, roi.y2,  cl, -cl), (roi.x2, roi.y2, -cl, -cl),
            ]:
                cv2.line(frame, (cx, cy), (cx + dx, cy), color, ct)
                cv2.line(frame, (cx, cy), (cx, cy + dy), color, ct)
            label = roi.name if roi.name else f"Zone {i+1}"
            font  = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
            lx, ly = roi.x1 + 6, roi.y1 + th + 6
            cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), color, -1)
            cv2.putText(frame, label, (lx, ly), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    def clear(self):
        self.zones.clear()
        print("[ROI] Semua zona dihapus.")

    def delete_last(self):
        if self.zones:
            print(f"[ROI] Dihapus: {self.zones.pop().name}")
        else:
            print("[ROI] Tidak ada zona untuk dihapus.")

    def save(self, filepath=ROI_SAVE_FILE, work_area_mgr=None):
        data = {"zones": [z.to_dict() for z in self.zones]}
        if work_area_mgr is not None:
            data["work_area"] = work_area_mgr.save_to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[ROI] Disimpan ke '{filepath}' ({len(self.zones)} zona)")

    def load(self, filepath=ROI_SAVE_FILE, work_area_mgr=None):
        if not os.path.exists(filepath):
            print(f"[ROI] File '{filepath}' tidak ditemukan.")
            return
        with open(filepath, "r") as f:
            data = json.load(f)
        self.zones.clear()
        for i, d in enumerate(data.get("zones", [])):
            zone = ROIZone.from_dict(d)
            zone.color = ROI_PALETTE[i % len(ROI_PALETTE)]
            self.zones.append(zone)
        if work_area_mgr is not None and "work_area" in data:
            work_area_mgr.load_from_dict(data["work_area"])
        print(f"[ROI] Loaded {len(self.zones)} zona dari '{filepath}'")


# ============================================================
# DRAWING HELPERS
# ============================================================
COCO_COLORS = [
    (255,  56,  56), (255, 157,  51), (255, 112,  31),
    (255, 178,  29), (207, 210,  49), (72,  249, 100),
    (146, 204,  23), (61,  219, 134), (26,  147,  52),
    (0,   212, 187), (44,  153, 168), (0,   194, 255),
    (52,   69, 147), (100, 115, 255), (0,    24, 236),
    (132,  56, 255), (82,    0, 133), (203,  56, 255),
    (255, 149, 200), (255,  55, 199),
]

def get_class_color(class_id: int, source: str = "coco") -> Tuple[int, int, int]:
    if source == "custom":
        custom_palette = [
            (0,  220, 255), (255, 80,  80), (80, 255,  80),
            (255, 180,  0), (180, 80, 255),
        ]
        return custom_palette[class_id % len(custom_palette)]
    return COCO_COLORS[class_id % len(COCO_COLORS)]


def draw_detection(frame, det: dict):
    """Draw satu deteksi objek + label."""
    x1, y1, x2, y2 = det["bbox"]
    conf       = det["conf"]
    class_name = det["class_name"]
    roi_idx    = det["roi_idx"]
    source     = det.get("source", "coco")
    color      = get_class_color(det["class_id"], source)

    thickness = 3 if source == "custom" else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    src_tag = "★" if source == "custom" else ""
    label   = f"{src_tag}{class_name} {conf:.0%} [Z{roi_idx+1}]"
    font    = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.55, 1)
    ly = y1 - 6 if y1 > th + 10 else y2 + th + 6
    cv2.rectangle(frame, (x1, ly - th - 3), (x1 + tw + 6, ly + 3), color, -1)
    cv2.putText(frame, label, (x1 + 3, ly), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_person_bbox(frame, person, is_operator=False, operator_name=""):
    """Draw person bounding box (dari estimated bbox MediaPipe)."""
    x1, y1, x2, y2 = person['bbox']
    conf = person['conf']

    if is_operator:
        color = OPERATOR_BBOX_COLOR
        thickness = 3
        label = f"{operator_name}"
    else:
        color = YOLO_BBOX_COLOR
        thickness = 1
        label = f"person"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 if is_operator else 0.45
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 2)
    ly = max(y1 - 8, th + 5)
    cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 8, ly + 4), color, -1)
    cv2.putText(frame, label, (x1 + 4, ly), font, font_scale,
                (255, 255, 255) if is_operator else (0, 0, 0),
                2 if is_operator else 1, cv2.LINE_AA)

    if is_operator:
        cl = min(25, (x2 - x1) // 4)
        ct = 4
        for cx, cy, dx, dy in [
            (x1, y1, cl, cl), (x2, y1, -cl, cl),
            (x1, y2, cl, -cl), (x2, y2, -cl, -cl),
        ]:
            cv2.line(frame, (cx, cy), (cx + dx, cy), color, ct)
            cv2.line(frame, (cx, cy), (cx, cy + dy), color, ct)


def draw_mediapipe_skeleton(frame, person: dict, style: str = "full"):
    """
    Draw MediaPipe Holistic skeleton (pose + tangan) di frame asli.
    style:
      "full"    → pose + right_hand + left_hand
      "minimal" → hanya pose (tanpa kepala)
      "hands"   → hanya kedua tangan
    """
    pose_lm  = person.get('pose_landmarks')
    r_hand   = person.get('right_hand_landmarks')
    l_hand   = person.get('left_hand_landmarks')

    # --- Gambar Pose landmarks ---
    if pose_lm is not None and style in ("full", "minimal"):
        if style == "full":
            mp_drawing.draw_landmarks(
                frame, pose_lm, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=SKELETON_POSE_LM, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=SKELETON_POSE_CN, thickness=2, circle_radius=2)
            )
        else:
            # Minimal: hanya gambar koneksi tubuh (skip kepala 0-10)
            BODY_CONNECTIONS = [
                conn for conn in mp_holistic.POSE_CONNECTIONS
                if conn[0] >= 11 and conn[1] >= 11
            ]
            mp_drawing.draw_landmarks(
                frame, pose_lm, frozenset(BODY_CONNECTIONS),
                mp_drawing.DrawingSpec(color=SKELETON_POSE_LM, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=SKELETON_POSE_CN, thickness=2, circle_radius=2)
            )

    # --- Gambar Tangan Kanan ---
    if r_hand is not None and style in ("full", "hands"):
        mp_drawing.draw_landmarks(
            frame, r_hand, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=SKELETON_RHAND_LM, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=SKELETON_RHAND_CN, thickness=2, circle_radius=2)
        )

    # --- Gambar Tangan Kiri ---
    if l_hand is not None and style in ("full", "hands"):
        mp_drawing.draw_landmarks(
            frame, l_hand, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=SKELETON_LHAND_LM, thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=SKELETON_LHAND_CN, thickness=2, circle_radius=2)
        )


def draw_mediapipe_skeleton_dim(frame, person: dict, style: str = "full"):
    """Versi redup untuk non-operator persons."""
    pose_lm = person.get('pose_landmarks')
    r_hand  = person.get('right_hand_landmarks')
    l_hand  = person.get('left_hand_landmarks')

    dim_lm  = mp_drawing.DrawingSpec(color=(60, 50, 40), thickness=1, circle_radius=2)
    dim_cn  = mp_drawing.DrawingSpec(color=(60, 30, 50), thickness=1, circle_radius=1)
    dim_h   = mp_drawing.DrawingSpec(color=(40, 15, 5),  thickness=1, circle_radius=2)
    dim_hc  = mp_drawing.DrawingSpec(color=(40, 30, 80), thickness=1, circle_radius=1)

    if pose_lm is not None and style in ("full", "minimal"):
        mp_drawing.draw_landmarks(frame, pose_lm, mp_holistic.POSE_CONNECTIONS, dim_lm, dim_cn)
    if r_hand is not None and style in ("full", "hands"):
        mp_drawing.draw_landmarks(frame, r_hand, mp_holistic.HAND_CONNECTIONS, dim_h, dim_hc)
    if l_hand is not None and style in ("full", "hands"):
        mp_drawing.draw_landmarks(frame, l_hand, mp_holistic.HAND_CONNECTIONS, dim_h, dim_hc)


def draw_aruco_marker(frame, aruco_result):
    """Draw ArUco marker outline."""
    pts = aruco_result['corners'].astype(int)
    cv2.polylines(frame, [pts], True, ARUCO_COLOR, 2, cv2.LINE_AA)
    cx, cy = aruco_result['center']
    cv2.circle(frame, (cx, cy), 4, ARUCO_COLOR, -1)
    id_text = f"ArUco #{aruco_result['id']}"
    cv2.putText(frame, id_text, (cx - 30, cy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ARUCO_COLOR, 1, cv2.LINE_AA)


def draw_hud(frame, fps, mode, roi_count, det_count, containment_mode,
             detect_mode_model, wa_sub_mode, seq_tracker=None,
             num_persons=0, num_markers=0, operators=None,
             pose_enabled=False, pose_style="full", sys_monitor=None,
             detect_device="CPU"):
    """Draw HUD info panel."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    aa   = cv2.LINE_AA

    if operators is None:
        operators = []

    hud_h = 300 + len(operators) * 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (440, max(300, hud_h)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y = 36
    cv2.putText(frame, "T-MIND | MediaPipe Holistic + Bearing Seq", (20, y),
                font, 0.52, (200, 220, 255), 2, aa)
    y += 26

    fps_col = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 80, 255)
    hw_tag  = "iGPU + CPU" if ocl_available else "CPU Only"
    cv2.putText(frame, f"FPS: {fps:.1f}  [{hw_tag}]", (20, y), font, 0.52, fps_col, 2, aa)
    y += 22

    if sys_monitor:
        sys_col = (50, 255, 100) if sys_monitor.cpu_percent < 80 else (0, 100, 255)
        cv2.putText(frame, f"Host: CPU {sys_monitor.cpu_percent:.1f}% | RAM {sys_monitor.ram_percent:.1f}%", 
                    (20, y), font, 0.40, sys_col, 1, aa)
        y += 18

    mode_col = (0, 255, 100) if mode == "DETECT" else (255, 180, 50)
    cv2.putText(frame, f"Mode: {mode}", (20, y), font, 0.52, mode_col, 2, aa)
    y += 22

    model_label = {
        "custom": f"Obj Model: {CUSTOM_MODEL} ({detect_device})",
        "coco":   f"Obj Model: {COCO_MODEL} ({detect_device})",
        "both":   f"Obj Model: CUSTOM+COCO ({detect_device})",
    }.get(detect_mode_model, "Model: ?")
    cv2.putText(frame, model_label, (20, y), font, 0.38, (200, 200, 80), 1, aa)
    y += 18
    cv2.putText(frame, "Pose Model: MediaPipe Holistic", (20, y), font, 0.38, (200, 200, 80), 1, aa)
    y += 20

    cv2.putText(frame, f"ROI: {roi_count}  |  Obj Det: {det_count}  |  {containment_mode}",
                (20, y), font, 0.38, (0, 230, 255), 1, aa)
    y += 20

    # Pose section
    cv2.putText(frame, "--- Pose (MediaPipe Holistic) ---", (20, y), font, 0.38, (200, 180, 255), 1, aa)
    y += 18
    cv2.putText(frame, f"Persons: {num_persons}  |  ArUco: {num_markers}",
                (20, y), font, 0.38, (255, 200, 150), 1, aa)
    y += 18

    ps = f"ON ({pose_style})" if pose_enabled else "OFF"
    pc = (0, 255, 200) if pose_enabled else (100, 100, 100)
    cv2.putText(frame, f"Pose Skeleton: {ps}  ['p'=toggle 'o'=style]", (20, y),
                font, 0.38, pc, 1, aa)
    y += 20

    # Operator list
    if operators:
        cv2.putText(frame, "Identified Operators:", (20, y), font, 0.40, (180, 255, 180), 1, aa)
        y += 18
        for op in operators:
            is_lost = op.get('is_lost', False)
            tag = " (TRACKING)" if is_lost else ""
            op_text = f"  ID{op['aruco']['id']}: {op['aruco']['name']}{tag}"
            cv2.putText(frame, op_text, (20, y), font, 0.38, (100, 255, 150), 1, aa)
            y += 18
    else:
        cv2.putText(frame, "No operator identified", (20, y), font, 0.38, (100, 100, 100), 1, aa)
        y += 18

    if wa_sub_mode != "IDLE":
        cv2.putText(frame, f"[WorkArea] {wa_sub_mode}", (20, y),
                    font, 0.40, (255, 240, 80), 1, aa)

    # Instructions (bawah)
    instructions = [
        "Drag=ROI  ENTER=Start/Stop  C=Clear  D=Del  S=Save  L=Load",
        "P=Pose  O=Style  Q=Quit  M=Containment  R=Reset Seq",
        "W=Corner  H=Hline  V=Vline  Z=Idle  X=ClearWA  N=Undo",
    ]
    iy = h - len(instructions) * 18 - 8
    for line in instructions:
        cv2.putText(frame, line, (w - 450, iy), font, 0.34, (140, 140, 140), 1, aa)
        iy += 18

    # Sequence Status Panel (Top Right)
    if seq_tracker and roi_count == 10:
        msg = seq_tracker.error_msg
        if "SALAH" in msg:
            color = (50, 50, 255)
        elif "Menunggu" in msg:
            color = (0, 200, 255)
        elif "Perakitan Selesai" in msg:
            color = (50, 255, 50)
        else:
            color = (255, 255, 50)

        box_w, box_h = 450, 70
        bx = w - box_w - 20
        by = 20
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (25, 25, 25), -1)
        cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), color, 2)

        step_text = f"Step {seq_tracker.current_step}/{len(SEQUENCE_STEPS)}" \
                    if seq_tracker.current_step < len(SEQUENCE_STEPS) else "DONE"
        cv2.putText(frame, "SEQUENCE STATUS - " + step_text, (bx + 10, by + 22),
                    font, 0.45, (200, 200, 200), 1, aa)
        (tw, th), _ = cv2.getTextSize(msg, font, 0.55, 2)
        display_msg = msg if tw <= box_w - 20 else msg[:45] + "..."
        cv2.putText(frame, display_msg, (bx + 10, by + 50), font, 0.55, color, 2, aa)


# ============================================================
# MAIN
# ============================================================
def main():
    global POSE_DRAW_STYLE

    print("=" * 68)
    print("  T-MIND V2.8b: MediaPipe Holistic Pose + Bearing Sequence")
    print("  (Pengganti YOLOv8-Pose — lebih detail: 33 pose + 42 tangan)")
    print("=" * 68)
    print()
    print(f"  Pose engine  : MediaPipe Holistic (Pose + 2 Hands)")
    print(f"  Obj detect   : {DETECT_MODE_MODEL} ({CUSTOM_MODEL})")
    print(f"  ArUco Dict   : {ARUCO_DICT_TYPE}")
    print(f"  Target Op    : {TARGET_OPERATOR_ID if TARGET_OPERATOR_ID is not None else 'Semua'}")
    print(f"  Pose         : {'ON' if ENABLE_POSE else 'OFF'} | Style: {POSE_DRAW_STYLE}")
    print(f"  HW Accel     : {'iGPU (OpenCL) + CPU' if ocl_available else 'CPU only'}")
    print()

    # ---- Init System Monitor ----
    sys_monitor = SystemMonitor().start()

    # ---- Init camera ----
    print(f"[INFO] Membuka sumber video: {CAMERA_INDEX}...")
    cap = CameraStream(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if not cap.ret:
        print("[ERROR] Gagal buka sumber video!")
        sys.exit(1)
    cap.start()

    # ---- Init MediaPipe Holistic Pose Detector ----
    pose_detector = MediaPipePoseDetector(
        min_detection_confidence=MP_MIN_DETECTION_CONF,
        min_tracking_confidence=MP_MIN_TRACKING_CONF,
        process_width=MP_PROCESS_WIDTH
    )

    # ---- Init YOLO Object Detector (bearing, dll) ----
    yolo_obj = YOLODetector(detect_mode=DETECT_MODE_MODEL,
                            conf=YOLO_CONF, classes=YOLO_CLASSES)

    # ---- Init ArUco ----
    aruco_detector = ArUcoDetector(ARUCO_DICT_TYPE)

    # ---- Init Operator Tracker ----
    operator_tracker = OperatorTracker(max_lost_frames=15, distance_threshold=250)

    # ---- Init ROI + WorkArea + Sequence ----
    roi_mgr = ROIManager()
    wa_mgr  = WorkAreaManager()
    seq_tracker = SequenceTracker(num_rois=10, debounce_frames=8)

    # ---- Window ----
    WIN = "T-MIND V2.8b | MediaPipe Holistic + Bearing Seq  [ENTER=detect, Q=quit]"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, cap.actual_w, cap.actual_h)

    def combined_mouse_cb(event, x, y, flags, param):
        if wa_mgr.sub_mode != "IDLE":
            wa_mgr.mouse_callback(event, x, y, flags, param)
        else:
            roi_mgr.mouse_callback(event, x, y, flags, param)

    cv2.setMouseCallback(WIN, combined_mouse_cb)

    # ---- State ----
    mode              = "SETUP"
    containment_modes = ["center", "overlap", "full"]
    contain_idx       = 0
    detections        = []
    fps               = 0.0
    frame_count       = 0
    t_start           = time.time()

    pose_enabled = ENABLE_POSE
    pose_styles  = ["full", "minimal", "hands"]
    pose_si      = pose_styles.index(POSE_DRAW_STYLE) if POSE_DRAW_STYLE in pose_styles else 0

    roi_mgr.load(ROI_SAVE_FILE, work_area_mgr=wa_mgr)
    print(f"\n[INFO] SETUP MODE – gambar zona ROI, lalu tekan ENTER")
    print(f"[INFO] Pipeline: MediaPipe(Holistic) + YOLO(ROI Obj) + ArUco → Draw")
    print(f"  'q'/ESC = Keluar | 'p' = Toggle Pose | 'o' = Style (full/minimal/hands)\n")

    try:
        while not cap.stopped:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # ========================================
            # STEP 1: Object Detection dalam ROI (bearing)
            # ========================================
            if mode == "DETECT" and roi_mgr.zones:
                detections = yolo_obj.detect(
                    frame, roi_mgr.zones,
                    iou_mode=containment_modes[contain_idx]
                )
                if len(roi_mgr.zones) == 10:
                    detected_rois = list(set([d["roi_idx"] for d in detections]))
                    seq_tracker.update(detected_rois)
            elif mode == "SETUP":
                detections = []

            # ========================================
            # STEP 2: MediaPipe Holistic Pose Detection
            # (pre-processing iGPU OpenCL, inference CPU)
            # ========================================
            pose_persons = pose_detector.detect(frame)

            # ========================================
            # STEP 3: ArUco Detection + Match → Person
            # ========================================
            aruco_results = aruco_detector.detect(frame)
            operators = operator_tracker.update(pose_persons, aruco_results)

            matched_bboxes = set()
            for op in operators:
                matched_bboxes.add(tuple(op['person']['bbox']))

            # ========================================
            # STEP 4: Draw layers
            # ========================================
            wa_mgr.draw(frame)
            roi_mgr.draw_zones(frame)
            roi_mgr.draw_in_progress(frame)
            for det in detections:
                draw_detection(frame, det)

            # ========================================
            # STEP 5: Draw non-operator persons (redup)
            # ========================================
            if mode == "DETECT":
                for p in pose_persons:
                    is_op = tuple(p['bbox']) in matched_bboxes
                    if not is_op:
                        draw_person_bbox(frame, p, is_operator=False)
                        if pose_enabled:
                            draw_mediapipe_skeleton_dim(frame, p, style=POSE_DRAW_STYLE)

            # ========================================
            # STEP 6: Draw operators (highlighted + skeleton)
            # ========================================
            for op in operators:
                person  = op['person']
                ar      = op['aruco']
                is_lost = op.get('is_lost', False)

                display_name = ar['name'] if not is_lost else f"{ar['name']} (TRACKING)"
                draw_person_bbox(frame, person, is_operator=True,
                                 operator_name=display_name)

                if not is_lost:
                    draw_aruco_marker(frame, ar)

                # Gambar skeleton operator (warna terang)
                if pose_enabled:
                    draw_mediapipe_skeleton(frame, person, style=POSE_DRAW_STYLE)

            # Draw unmatched ArUco markers
            matched_aruco_ids = set(op['aruco']['id'] for op in operators)
            for ar in aruco_results:
                if ar['id'] not in matched_aruco_ids:
                    draw_aruco_marker(frame, ar)

            # ========================================
            # STEP 7: HUD
            # ========================================
            draw_hud(
                frame, fps, mode,
                roi_count=len(roi_mgr.zones),
                det_count=len(detections),
                containment_mode=containment_modes[contain_idx],
                detect_mode_model=DETECT_MODE_MODEL,
                wa_sub_mode=wa_mgr.sub_mode,
                seq_tracker=seq_tracker,
                num_persons=len(pose_persons),
                num_markers=len(aruco_results),
                operators=operators,
                pose_enabled=pose_enabled,
                pose_style=POSE_DRAW_STYLE,
                sys_monitor=sys_monitor,
                detect_device=yolo_obj.device_used,
            )

            # SETUP banner
            if mode == "SETUP":
                h_f, w_f = frame.shape[:2]
                banner = "SETUP MODE: Gambar ROI, lalu tekan ENTER untuk mulai deteksi"
                (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                bx = (w_f - bw) // 2
                by = h_f - 30
                cv2.rectangle(frame, (bx - 8, by - bh - 6),
                              (bx + bw + 8, by + 6), (20, 20, 20), -1)
                cv2.putText(frame, banner, (bx, by),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 230, 255), 2, cv2.LINE_AA)

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 1.0:
                fps         = frame_count / elapsed
                frame_count = 0
                t_start     = time.time()

            cv2.imshow(WIN, frame)

            # Log
            if operators:
                names = ", ".join([f"ID{op['aruco']['id']}={op['aruco']['name']}"
                                   for op in operators])
                print(f"\r[TRACKING] {names}     ", end="", flush=True)

            # ========================================
            # STEP 8: Keyboard
            # ========================================
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                break

            elif key == 13:  # ENTER
                if mode == "SETUP":
                    if not roi_mgr.zones:
                        print("[WARN] Belum ada ROI!")
                    else:
                        mode = "DETECT"
                        if len(roi_mgr.zones) == 10:
                            seq_tracker.reset()
                            print("[INFO] Sequence Tracking aktif (10 ROI).")
                        print(f"[INFO] DETECT MODE – aktif ({len(roi_mgr.zones)} zona + Pose)")
                else:
                    mode = "SETUP"
                    detections = []
                    seq_tracker.reset()
                    print("[INFO] SETUP MODE")

            elif key == ord('r'):
                seq_tracker.reset()
                print("[INFO] Sequence Tracker di-reset.")

            elif key == ord('c'):
                roi_mgr.clear()
                if mode == "DETECT":
                    mode = "SETUP"

            elif key == ord('d'):
                roi_mgr.delete_last()
                if not roi_mgr.zones and mode == "DETECT":
                    mode = "SETUP"

            elif key == ord('s'):
                roi_mgr.save(ROI_SAVE_FILE, work_area_mgr=wa_mgr)

            elif key == ord('l'):
                roi_mgr.load(ROI_SAVE_FILE, work_area_mgr=wa_mgr)

            elif key == ord('m'):
                contain_idx = (contain_idx + 1) % len(containment_modes)
                print(f"[INFO] Containment: '{containment_modes[contain_idx]}'")

            # Pose controls
            elif key == ord('p'):
                pose_enabled = not pose_enabled
                print(f"\n[INFO] Pose: {'ON' if pose_enabled else 'OFF'}")

            elif key == ord('o'):
                pose_si = (pose_si + 1) % len(pose_styles)
                POSE_DRAW_STYLE = pose_styles[pose_si]
                print(f"\n[INFO] Pose Style: {POSE_DRAW_STYLE}")

            # Work Area
            elif key == ord('w'):
                wa_mgr.sub_mode = "PLACE_CORNER"
                print(f"[WorkArea] PLACE_CORNER – next: {_CORNER_CYCLE[len(wa_mgr.corners) % 4]}")
            elif key == ord('h'):
                wa_mgr.sub_mode = "PLACE_HLINE"
                print("[WorkArea] PLACE_HLINE")
            elif key == ord('v'):
                wa_mgr.sub_mode = "PLACE_VLINE"
                print("[WorkArea] PLACE_VLINE")
            elif key == ord('z'):
                wa_mgr.sub_mode = "IDLE"
                print("[WorkArea] IDLE")
            elif key == ord('x'):
                wa_mgr.clear()
            elif key == ord('n'):
                if wa_mgr.corners:
                    wa_mgr.delete_last_corner()
                elif wa_mgr.ref_lines:
                    wa_mgr.delete_last_line()

    finally:
        cap.stop()
        pose_detector.close()
        sys_monitor.stop()
        cv2.destroyAllWindows()
        print("\n\n[INFO] Program selesai.")


# ============================================================
if __name__ == "__main__":
    main()
