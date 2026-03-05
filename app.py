from flask import Flask, render_template, Response, jsonify, request, redirect, send_file
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

MODEL_PATH = "models/best.pt" 
UPLOAD_FOLDER = "uploads"
SNAPSHOT_FOLDER = "static/snapshots"
DATABASE = "database/detections.db"

CONFIDENCE_THRESHOLD = 0.45
DETECTION_COOLDOWN = 2  
MIN_BOX_AREA = 1500  
MAX_BOX_AREA = 800000  
IOU_THRESHOLD = 0.3  

DETECTION_FRAME_SKIP = 8
RESIZE_WIDTH = 720  
JPEG_QUALITY = 78  
INFERENCE_SIZE = 640

SNAPSHOT_SIZE = 640
MAX_SNAPSHOTS = 6

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("models", exist_ok=True)

camera = None
model = None
last_detection_time = 0

latest_detection = {
    'status': 'Tidak Ada Deteksi',
    'confidence': 0,
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'disease_counts': {'Sehat': 0, 'Coccidiosis': 0, 'Newcastle': 0, 'Salmonella': 0},
    'total_detected': 0
}

detection_snapshots = []
snapshot_lock = threading.Lock()

CLASS_NAMES = {
    0: 'Sehat',         
    1: 'Coccidiosis',    
    2: 'Newcastle',      
    3: 'Salmonella'       
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
            print(f"Loading model from {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model(dummy, verbose=False)
            print("Model loaded successfully")
        else:
            print("Model file not found")
            model = None
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

load_model()

def get_tapo_rtsp_url(ip, username, password):
    return f"rtsp://{username}:{password}@{ip}:554/stream1"

def connect_camera():
    global camera
    try:
        tapo_ip = "192.168.0.14"
        tapo_user = "fandhi"
        tapo_pass = "fandhi_ta"
        
        rtsp_url = get_tapo_rtsp_url(tapo_ip, tapo_user, tapo_pass)
        print("Connecting to camera...")
        
        camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if camera.isOpened():
            print("Camera connected")
            return True
        else:
            print("Failed to connect, using webcam")
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera.isOpened()
    except Exception as e:
        print(f"Camera error: {e}")
        return False

def create_detection_snapshot(frame, detections_data):
    if len(detections_data) == 0:
        return None
    
    snapshot = frame.copy()
    
    for det in detections_data:
        box = det['box']
        disease = det['disease']
        confidence = det['confidence']
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        color_map = {
            'Sehat': (0, 255, 0),
            'Coccidiosis': (0, 165, 255),
            'Newcastle': (0, 0, 255),
            'Salmonella': (255, 0, 255)
        }
        color = color_map.get(disease, (255, 255, 255))
        
        cv2.rectangle(snapshot, (x1, y1), (x2, y2), color, 4)
        
        label = f"{disease}: {confidence:.1f}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(snapshot, (x1, y1 - h - 15), (x1 + w + 10, y1), color, -1)
        cv2.putText(snapshot, label, (x1 + 5, y1 - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    height, width = snapshot.shape[:2]
    if width > SNAPSHOT_SIZE:
        scale = SNAPSHOT_SIZE / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        snapshot = cv2.resize(snapshot, (new_width, new_height))
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    num_detected = len(detections_data)
    info_text = f"{timestamp} | Detected: {num_detected} object(s)"
    
    cv2.putText(snapshot, info_text, (10, snapshot.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    filename = f"detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SNAPSHOT_FOLDER, filename)
    cv2.imwrite(filepath, snapshot, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return filename

def generate_frames():
    global camera, model, latest_detection, last_detection_time, detection_snapshots
    
    if camera is None or not camera.isOpened():
        connect_camera()
    
    frame_count = 0
    
    print("Stream started")
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Reconnecting camera...")
            time.sleep(0.1)
            connect_camera()
            continue
        
        frame_count += 1
        current_time = time.time()
        original_frame = frame.copy()
        
        if frame_count % DETECTION_FRAME_SKIP == 0 and model is not None:
            try:
                results = model(original_frame, 
                              conf=CONFIDENCE_THRESHOLD, 
                              iou=IOU_THRESHOLD,
                              verbose=False,
                              imgsz=INFERENCE_SIZE,
                              max_det=20)
                
                if len(results[0].boxes) > 0:
                    valid_detections = []
                    detections_data = []
                    disease_counts = {'Sehat': 0, 'Coccidiosis': 0, 'Newcastle': 0, 'Salmonella': 0}
                    
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0]
                        box_area = (x2 - x1) * (y2 - y1)
                        
                        if (conf >= CONFIDENCE_THRESHOLD and 
                            MIN_BOX_AREA <= box_area <= MAX_BOX_AREA):
                            
                            class_id = int(box.cls[0])
                            disease = CLASS_NAMES.get(class_id, f'Unknown_{class_id}')
                            
                            valid_detections.append(box)
                            detections_data.append({
                                'box': box,
                                'disease': disease,
                                'confidence': conf * 100
                            })
                            
                            disease_counts[disease] = disease_counts.get(disease, 0) + 1
                    
                    if len(valid_detections) > 0:
                        print(f"Detected {len(valid_detections)} objects: {disease_counts}")
                        
                        if current_time - last_detection_time >= DETECTION_COOLDOWN:
                            snapshot_filename = create_detection_snapshot(original_frame, detections_data)
                            
                            if snapshot_filename:
                                with snapshot_lock:
                                    detection_snapshots.insert(0, snapshot_filename)
                                    if len(detection_snapshots) > MAX_SNAPSHOTS:
                                        old_snapshot = detection_snapshots.pop()
                                        old_path = os.path.join(SNAPSHOT_FOLDER, old_snapshot)
                                        if os.path.exists(old_path):
                                            os.remove(old_path)
                            
                            total = len(detections_data)
                            main_disease = max(disease_counts, key=disease_counts.get)
                            main_count = disease_counts[main_disease]
                            
                            latest_detection = {
                                'status': f"{main_disease} ({main_count}x)" if total > 1 else main_disease,
                                'confidence': round(max([d['confidence'] for d in detections_data]), 2),
                                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'disease_counts': disease_counts,
                                'total_detected': total,
                                'snapshot': snapshot_filename
                            }
                            
                            for det in detections_data:
                                threading.Thread(target=save_detection, 
                                              args=(det['disease'], det['confidence']/100), 
                                              daemon=True).start()
                            
                            last_detection_time = current_time
                            print(f"Saved: {total} objects - {disease_counts}")
                
            except Exception as e:
                print(f"Detection error: {e}")
        
        height, width = frame.shape[:2]
        if width > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        
        cv2.putText(frame, f"Confidence: {CONFIDENCE_THRESHOLD*100:.0f}%", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if latest_detection.get('total_detected', 0) > 0:
            cv2.putText(frame, f"{latest_detection['status']}", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        
        if ret:
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
    except Exception as e:
        print(f"Database error: {e}")

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

@app.route('/api/detection_snapshots')
def get_detection_snapshots():
    with snapshot_lock:
        snapshots = [{'filename': s, 'url': f'/static/snapshots/{s}'} 
                    for s in detection_snapshots]
    return jsonify({'snapshots': snapshots})

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
        return jsonify({'total': total, 'by_disease': by_disease, 'recent': recent})
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
    return jsonify({'status': 'online' if camera and camera.isOpened() else 'offline'})

@app.route('/api/model_info')
def model_info():
    global model
    if model:
        return jsonify({'loaded': True, 'confidence': CONFIDENCE_THRESHOLD, 'inference_size': INFERENCE_SIZE})
    return jsonify({'loaded': False})

@app.route('/api/set_threshold', methods=['POST'])
def set_threshold():
    global CONFIDENCE_THRESHOLD
    try:
        data = request.get_json()
        new_threshold = float(data.get('threshold', 0.45))
        if 0.1 <= new_threshold <= 1.0:
            CONFIDENCE_THRESHOLD = new_threshold
            return jsonify({'success': True, 'threshold': CONFIDENCE_THRESHOLD})
        return jsonify({'success': False, 'error': 'Must be 0.1-1.0'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("Broiler Disease Detection System")
    print("=" * 70)
    print(f"Inference Size: {INFERENCE_SIZE}px")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"Detection Interval: Every {DETECTION_FRAME_SKIP} frames")
    print(f"Stream Resolution: {RESIZE_WIDTH}px")
    print(f"Multi-object Detection: Enabled (max 20 objects)")
    print(f"Live Video: Clean (no bounding boxes)")
    print(f"Snapshots: Bounding boxes displayed")
    print("=" * 70)
    
    connect_camera()
    
    if model is None:
        print("WARNING: Model not loaded")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)