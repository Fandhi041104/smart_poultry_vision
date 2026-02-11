from flask import Flask, render_template, Response, jsonify, request, redirect
import cv2
import sqlite3
import datetime
import os
from ultralytics import YOLO
import threading
import time
from pathlib import Path
import numpy as np

app = Flask(__name__)

# Konfigurasi
MODEL_PATH = "models/best.pt"  # Model YOLOv11 hasil training
UPLOAD_FOLDER = "uploads"
DATABASE = "database/detections.db"

# Threshold untuk mengurangi false detection
CONFIDENCE_THRESHOLD = 0.75  # Threshold confidence (75%)
DETECTION_COOLDOWN = 5  # Delay 5 detik antar deteksi untuk menghindari spam
MIN_BOX_AREA = 5000  # Minimum area bounding box (pixel^2) - filter objek terlalu kecil
MAX_BOX_AREA = 500000  # Maximum area bounding box - filter objek terlalu besar
IOU_THRESHOLD = 0.5  # IoU threshold untuk NMS

# Optimasi Performance
DETECTION_FRAME_SKIP = 30  # Deteksi setiap 30 frame (1 detik pada 30fps) - hemat resource
RESIZE_WIDTH = 640  # Resize frame untuk processing (lebih kecil = lebih cepat)
JPEG_QUALITY = 75  # Quality streaming (lebih rendah = lebih cepat)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global variables
camera = None
model = None
last_detection_time = 0
latest_detection = {
    'status': 'Tidak Ada Deteksi',
    'confidence': 0,
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'disease_counts': {'Sehat': 0, 'Coccidiosis': 0, 'Newcastle': 0, 'Salmonella': 0}
}

# Mapping class names - SESUAIKAN DENGAN URUTAN TRAINING ANDA
CLASS_NAMES = {
    0: 'Sehat',           # Healthy
    1: 'Coccidiosis',     # Coccidiosis
    2: 'Newcastle',       # Newcastle Disease
    3: 'Salmonella'       # Salmonella
}

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  disease TEXT,
                  confidence REAL,
                  timestamp TEXT,
                  image_path TEXT)''')
    conn.commit()
    conn.close()

init_db()

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print(f"📂 Loading model from: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            
            print(f"✓ YOLOv11 Model loaded successfully!")
            
            # Cek nama kelas dari model
            if hasattr(model, 'names'):
                print(f"📊 Model original classes: {model.names}")
                print(f"📊 Number of classes: {len(model.names)}")
            
            # Override dengan nama kelas yang benar
            print(f"🔧 Using custom class mapping: {CLASS_NAMES}")
            
        else:
            print(f"⚠ ERROR: Model file not found at {MODEL_PATH}")
            print(f"⚠ Please ensure best.pt is in the 'models' folder")
            model = None
            
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        model = None

load_model()

def get_tapo_rtsp_url(ip, username, password):
    return f"rtsp://{username}:{password}@{ip}:554/stream1"

def connect_camera():
    global camera
    try:
        tapo_ip = "192.168.0.12"
        tapo_user = "fandhi"
        tapo_pass = "tugas_akhir"
        
        rtsp_url = get_tapo_rtsp_url(tapo_ip, tapo_user, tapo_pass)
        print(f"🎥 Connecting to Tapo C100...")
        
        camera = cv2.VideoCapture(rtsp_url)
        
        # Optimasi untuk mengurangi latency
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer kecil = latency rendah
        camera.set(cv2.CAP_PROP_FPS, 30)  # Set FPS
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Resolution tidak terlalu besar
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if camera.isOpened():
            print(f"✓ Connected to Tapo C100")
            return True
        else:
            print(f"✗ Failed to connect, using webcam")
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera.isOpened()
            
    except Exception as e:
        print(f"✗ Camera error: {e}")
        return False

def draw_detection_box(frame, box, disease, confidence):
    """Gambar bounding box yang lebih jelas dan persistent"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # Warna berbeda untuk setiap penyakit
    color_map = {
        'Sehat': (0, 255, 0),        # Hijau
        'Coccidiosis': (0, 165, 255), # Orange
        'Newcastle': (0, 0, 255),     # Merah
        'Salmonella': (255, 0, 255)   # Magenta
    }
    color = color_map.get(disease, (255, 255, 255))
    
    # Gambar rectangle yang lebih tebal
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    # Background untuk text
    label = f"{disease}: {confidence:.1f}%"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
    
    # Text label
    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def generate_frames():
    global camera, model, latest_detection, last_detection_time
    
    if camera is None or not camera.isOpened():
        connect_camera()
    
    frame_count = 0
    detected_frame = None  # Simpan frame dengan deteksi
    detection_display_time = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            print("⚠ Failed to read frame, reconnecting...")
            time.sleep(0.1)  # Delay sebentar sebelum reconnect
            connect_camera()
            continue
        
        frame_count += 1
        current_time = time.time()
        
        # Resize frame untuk display (mengurangi beban encoding)
        height, width = frame.shape[:2]
        if width > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        display_frame = frame.copy()
        
        # Deteksi HANYA setiap DETECTION_FRAME_SKIP frame (hemat CPU)
        if frame_count % DETECTION_FRAME_SKIP == 0 and model is not None:
            try:
                # Run YOLOv11 inference dengan parameter optimal
                results = model(frame, 
                              conf=CONFIDENCE_THRESHOLD, 
                              iou=IOU_THRESHOLD,
                              verbose=False,
                              imgsz=640)  # Inference pada resolusi 640 (lebih cepat)
                
                # Cek apakah ada deteksi
                if len(results[0].boxes) > 0:
                    # Filter dengan multiple criteria
                    valid_detections = []
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        
                        # Hitung area bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        box_area = (x2 - x1) * (y2 - y1)
                        
                        # Filter berdasarkan:
                        # 1. Confidence tinggi
                        # 2. Area bounding box wajar (tidak terlalu kecil/besar)
                        if (conf >= CONFIDENCE_THRESHOLD and 
                            MIN_BOX_AREA <= box_area <= MAX_BOX_AREA):
                            valid_detections.append(box)
                            # print(f"🔍 Valid detection - Conf: {conf:.2f}, Area: {box_area:.0f}px²")
                    
                    if len(valid_detections) > 0:
                        # Ambil deteksi dengan confidence tertinggi
                        best_box = max(valid_detections, key=lambda x: float(x.conf[0]))
                        confidence = float(best_box.conf[0])
                        class_id = int(best_box.cls[0])
                        
                        # Gunakan mapping CLASS_NAMES yang sudah didefinisikan
                        disease = CLASS_NAMES.get(class_id, f'Unknown_Class_{class_id}')
                        
                        # Cooldown untuk menghindari spam deteksi
                        if current_time - last_detection_time >= DETECTION_COOLDOWN:
                            # Update latest detection
                            latest_detection = {
                                'status': disease,
                                'confidence': round(confidence * 100, 2),
                                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            # Simpan ke database (async untuk tidak blocking)
                            threading.Thread(target=save_detection, args=(disease, confidence)).start()
                            last_detection_time = current_time
                            
                            print(f"🔍 DETECTED: {disease} - Confidence: {confidence*100:.2f}%")
                        
                        # Gambar bounding box pada frame
                        display_frame = draw_detection_box(display_frame, best_box, disease, confidence * 100)
                        detected_frame = display_frame.copy()
                        detection_display_time = current_time
                
            except Exception as e:
                print(f"❌ Detection error: {e}")
        
        # Tampilkan frame dengan deteksi selama 3 detik
        if detected_frame is not None and (current_time - detection_display_time) < 3.0:
            display_frame = detected_frame
        
        # Info text dengan font lebih kecil (lebih ringan)
        cv2.putText(display_frame, f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if latest_detection['status'] != 'Tidak Ada Deteksi':
            status_text = f"{latest_detection['status']} ({latest_detection['confidence']:.1f}%)"
            cv2.putText(display_frame, status_text, 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Encode dengan quality lebih rendah untuk speed
        ret, buffer = cv2.imencode('.jpg', display_frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def save_detection(disease, confidence):
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        c.execute("INSERT INTO detections (disease, confidence, timestamp, image_path) VALUES (?, ?, ?, ?)",
                  (disease, confidence, timestamp, ''))
        conn.commit()
        conn.close()
        print(f"💾 Saved to database: {disease} ({confidence*100:.2f}%)")
    except Exception as e:
        print(f"❌ Database error: {e}")

# Routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/riwayat')
def riwayat():
    return render_template('riwayat.html')

@app.route('/edukasi')
def edukasi():
    return render_template('edukasi.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/latest_detection')
def get_latest_detection():
    return jsonify(latest_detection)

@app.route('/api/statistics')
def get_statistics():
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) FROM detections")
        total = c.fetchone()[0]
        
        c.execute("SELECT disease, COUNT(*) FROM detections GROUP BY disease")
        by_disease = dict(c.fetchall())
        
        c.execute("SELECT disease, confidence, timestamp FROM detections ORDER BY id DESC LIMIT 10")
        recent = [{'disease': row[0], 'confidence': round(row[1]*100, 2), 'timestamp': row[2]} 
                  for row in c.fetchall()]
        
        conn.close()
        
        return jsonify({
            'total': total,
            'by_disease': by_disease,
            'recent': recent
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def get_history():
    try:
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("SELECT disease, confidence, timestamp FROM detections ORDER BY id DESC LIMIT 50")
        history = [{'disease': row[0], 'confidence': round(row[1]*100, 2), 'timestamp': row[2]} 
                   for row in c.fetchall()]
        conn.close()
        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/camera_status')
def camera_status():
    global camera
    status = 'online' if camera and camera.isOpened() else 'offline'
    return jsonify({'status': status})

@app.route('/api/model_info')
def model_info():
    """Endpoint untuk cek informasi model"""
    global model
    if model is not None:
        info = {
            'loaded': True,
            'model_path': MODEL_PATH,
            'model_classes': model.names if hasattr(model, 'names') else None,
            'custom_mapping': CLASS_NAMES,
            'confidence_threshold': CONFIDENCE_THRESHOLD
        }
    else:
        info = {
            'loaded': False,
            'error': 'Model not loaded'
        }
    return jsonify(info)

@app.route('/api/set_threshold', methods=['POST'])
def set_threshold():
    """Endpoint untuk mengubah confidence threshold"""
    global CONFIDENCE_THRESHOLD
    try:
        data = request.get_json()
        new_threshold = float(data.get('threshold', 0.7))
        if 0.1 <= new_threshold <= 1.0:
            CONFIDENCE_THRESHOLD = new_threshold
            return jsonify({'success': True, 'threshold': CONFIDENCE_THRESHOLD})
        else:
            return jsonify({'success': False, 'error': 'Threshold must be between 0.1 and 1.0'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Starting Broiler Disease Detection System")
    print("=" * 60)
    print(f"📍 Working Directory: {os.getcwd()}")
    print(f"📂 Model Path: {MODEL_PATH}")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"📏 Min Box Area: {MIN_BOX_AREA}px² | Max: {MAX_BOX_AREA}px²")
    print(f"🎯 IoU Threshold: {IOU_THRESHOLD}")
    print(f"⏱️  Detection Cooldown: {DETECTION_COOLDOWN} seconds")
    print(f"⚡ Detection every {DETECTION_FRAME_SKIP} frames")
    print(f"📺 Stream Resolution: {RESIZE_WIDTH}px | Quality: {JPEG_QUALITY}%")
    print(f"🏷️  Class Mapping: {CLASS_NAMES}")
    print("=" * 60)
    
    connect_camera()
    
    if model is None:
        print("\n⚠ WARNING: Model not loaded! Detection will not work.")
        print("⚠ Please ensure best.pt is in the 'models' folder\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)