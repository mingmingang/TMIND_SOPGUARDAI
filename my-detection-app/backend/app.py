import cv2
import mediapipe as mp
import numpy as np
import time
import os
import sys
from flask import Flask, Response, request, jsonify 
from flask_cors import CORS
import pyodbc

# ==================== SQL Server Setup ====================
DB_CONFIG = (
    "Driver={SQL Server};"
    "Server=AANG;" # Contoh: localhost atau .\SQLEXPRESS
    "Database=TMIND_DB;"
    "Trusted_Connection=yes;"
)

# ============================================================
# KONFIGURASI ASLI KAMU (100% Sesuai Request)
# ============================================================
MODEL_PATH = "efficientdet_lite0.tflite"
MAX_RESULTS = 10           
SCORE_THRESHOLD = 0.35     
CAMERA_INDEX = 0           
FRAME_WIDTH = 1280         
FRAME_HEIGHT = 720         

COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (255, 128, 0),
    (0, 128, 255), (128, 0, 255),
]
TARGET_OBJECTS = []  

app = Flask(__name__)
CORS(app) # Supaya React bisa akses

# Global variable untuk menyimpan hasil deteksi (MediaPipe Async)
latest_detection_result = None

# ============================================================
# FUNGSI-FUNGSI ORIGINAL KAMU
# ============================================================

def download_model():
    if os.path.exists(MODEL_PATH): return True
    print("[INFO] Mengunduh model...")
    model_url = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite"
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, MODEL_PATH)
        return True
    except Exception as e:
        print(f"[ERROR] Gagal unduh model: {e}")
        return False

def get_color(index):
    return COLORS[index % len(COLORS)]

def draw_detection(frame, detection, color):
    bbox = detection.bounding_box
    x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
    thickness = 2
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    corner_len = min(30, w // 4, h // 4)
    corner_thick = 4
    cv2.line(frame, (x, y), (x + corner_len, y), color, corner_thick)
    cv2.line(frame, (x, y), (x, y + corner_len), color, corner_thick)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, corner_thick)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, corner_thick)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, corner_thick)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, corner_thick)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thick)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thick)

    for i, category in enumerate(detection.categories):
        label = f"{category.category_name}: {category.score:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
        label_y = max(y - 10 - (i * (text_h + 10)), text_h + 5)
        cv2.rectangle(frame, (x, label_y - text_h - 5), (x + text_w + 10, label_y + 5), color, -1)
        cv2.putText(frame, label, (x + 5, label_y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

def draw_hud(frame, fps, detection_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    fps_color = (0, 255, 0) if fps >= 20 else (0, 255, 255) if fps >= 10 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Objek Terdeteksi: {detection_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    filter_text = f"Filter: {', '.join(TARGET_OBJECTS[:3])}" if TARGET_OBJECTS else "Filter: Semua Objek"
    cv2.putText(frame, filter_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

def detection_callback(result, output_image, timestamp_ms):
    global latest_detection_result
    latest_detection_result = result

# ============================================================
# LOGIKA GENERATOR STREAMING
# ============================================================

def generate_frames():
    global latest_detection_result
    download_model()
    
    # Inisialisasi MediaPipe
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp.tasks.vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        max_results=MAX_RESULTS,
        score_threshold=SCORE_THRESHOLD,
        result_callback=detection_callback,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    fps = 0
    frame_count = 0
    start_time = time.time()
    category_color_map = {}
    color_index = 0

    with mp.tasks.vision.ObjectDetector.create_from_options(options) as detector:
        timestamp = 0
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            timestamp += 1
            detector.detect_async(mp_image, timestamp)

            detection_count = 0
            if latest_detection_result:
                for detection in latest_detection_result.detections:
                    category_name = detection.categories[0].category_name
                    if TARGET_OBJECTS and category_name not in TARGET_OBJECTS: continue
                    
                    if category_name not in category_color_map:
                        category_color_map[category_name] = get_color(color_index)
                        color_index += 1
                    
                    draw_detection(frame, detection, category_color_map[category_name])
                    detection_count += 1

            frame_count += 1
            if (time.time() - start_time) >= 1.0:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()

            draw_hud(frame, fps, detection_count)

            # Convert frame ke JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Yield dalam format MJPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        # Query mengambil data dari tabel sesuai kolom di SSMS kamu
        query = """
            SELECT username, fullName, division, employeeId, monitoringHours, securityStatus, accessLevel 
            FROM Users 
            WHERE username=? AND password=?
        """
        cursor.execute(query, (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            # Map hasil database (tuple) ke JSON Object
            return jsonify({
                "success": True, 
                "userData": {
                    "username": str(user[0]),
                    "fullName": str(user[1]),
                    "division": str(user[2]),
                    "employeeId": str(user[3]),
                    "monitoringHours": str(user[4]),
                    "securityStatus": str(user[5]),
                    "accessLevel": str(user[6])
                }
            })
        else:
            return jsonify({"success": False, "message": "User tidak ditemukan!"}), 401

    except Exception as e:
        print(f"Error Database: {e}")
        return jsonify({"success": False, "message": "Terjadi kesalahan server"}), 500
    
@app.route('/api/update-profile', methods=['POST'])
def update_profile():
    data = request.json
    username = data.get('username') # Kunci utama
    full_name = data.get('fullName')
    division = data.get('division')
    employee_id = data.get('employeeId')

    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        query = """
            UPDATE Users 
            SET fullName = ?, division = ?, employeeId = ?
            WHERE username = ?
        """
        cursor.execute(query, (full_name, division, employee_id, username))
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": "Profil berhasil diperbarui"})
    except Exception as e:
        print(f"Error Update DB: {e}")
        return jsonify({"success": False, "message": str(e)}), 500

# Route untuk mengambil semua user
@app.route('/api/users', methods=['GET'])
def get_users():
    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT id, fullName, employeeId, division, accessLevel, securityStatus, username FROM Users")
        rows = cursor.fetchall()
        
        users = []
        for r in rows:
            users.append({
                "id": r[0], "fullName": r[1], "employeeId": r[2],
                "division": r[3], "accessLevel": r[4], "status": r[5], "username": r[6]
            })
        conn.close()
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route untuk menambah user baru
@app.route('/api/users', methods=['POST'])
def add_user():
    data = request.json
    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        query = """
            INSERT INTO Users (username, password, fullName, employeeId, division, accessLevel, securityStatus)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            data['username'], data['password'], data['fullName'], 
            data['employeeId'], data['division'], data['accessLevel'], 'OFFLINE'
        ))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/users/update', methods=['POST'])
def update_user_management():
    data = request.json
    try:
        conn = pyodbc.connect(DB_CONFIG)
        cursor = conn.cursor()
        query = """
            UPDATE Users 
            SET fullName = ?, employeeId = ?, division = ?, accessLevel = ?, securityStatus = ?
            WHERE id = ?
        """
        cursor.execute(query, (
            data['fullName'], data['employeeId'], data['division'], 
            data['accessLevel'], data['status'], data['id']
        ))
        conn.commit()
        conn.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
if __name__ == '__main__':
    # Ganti port dari 5000 ke 5001
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)