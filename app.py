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
MODEL_PATH = "models/best.pt" 
UPLOAD_FOLDER = "uploads"
DATABASE = "database/detections.db"

# Threshold 
CONFIDENCE_THRESHOLD = 0.75  
DETECTION_COOLDOWN = 5  
MIN_BOX_AREA = 5000  
MAX_BOX_AREA = 500000 
IOU_THRESHOLD = 0.5  

# Optimasi
DETECTION_FRAME_SKIP = 30 
RESIZE_WIDTH = 640  # Resize frame 
JPEG_QUALITY = 75  

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("database", exist_ok=True)
os.makedirs("models", exist_ok=True)

camera = None
model = None
last_detection_time = 0
latest_detection = {
    'status': 'Tidak Ada Deteksi',
    'confidence': 0,
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'disease_counts': {'Sehat': 0, 'Coccidiosis': 0, 'Newcastle': 0, 'Salmonella': 0}
}


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
            print(f"Loading model from: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            
            print(f"✓ YOLOv11 Model loaded successfully!")
            

            if hasattr(model, 'names'):
                print(f"Model original classes: {model.names}")
                print(f"Number of classes: {len(model.names)}")
            

            print(f"Using custom class mapping: {CLASS_NAMES}")
            
        else:
            print(f"ERROR: Model file tidak ditemukan{MODEL_PATH}")
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
        tapo_ip = "192.168.137.71"
        tapo_user = "fandhi"
        tapo_pass = "fandhi_ta"
        
        rtsp_url = get_tapo_rtsp_url(tapo_ip, tapo_user, tapo_pass)
        print(f"Mengkoneksikan ke Ip Camera....")
        
        camera = cv2.VideoCapture(rtsp_url)
        
        # Optimasi untuk mengurangi latency
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        camera.set(cv2.CAP_PROP_FPS, 30)  
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Resolution 
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if camera.isOpened():
            print(f"Connected to Tapo C100")
            return True
        else:
            print(f"Failed to connect")
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera.isOpened()
            
    except Exception as e:
        print(f" Camera error: {e}")
        return False

def draw_detection_box(frame, box, disease, confidence):
    """Gambar bounding box yang lebih jelas"""
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    

    color_map = {
        'Sehat': (0, 255, 0),        # Hijau
        'Coccidiosis': (0, 165, 255), # Orange
        'Newcastle': (0, 0, 255),     # Merah
        'Salmonella': (255, 0, 255)   # Magenta
    }
    color = color_map.get(disease, (255, 255, 255))
    

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    

    label = f"{disease}: {confidence:.1f}%"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), color, -1)
    
  
    cv2.putText(frame, label, (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return frame

def generate_frames():
    global camera, model, latest_detection, last_detection_time
    
    if camera is None or not camera.isOpened():
        connect_camera()
    
    frame_count = 0
    detected_frame = None  
    detection_display_time = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame, reconnecting...")
            time.sleep(0.1)  
            connect_camera()
            continue
        
        frame_count += 1
        current_time = time.time()
        
        # Resize frame untuk display
        height, width = frame.shape[:2]
        if width > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
        display_frame = frame.copy()
        
        
        if frame_count % DETECTION_FRAME_SKIP == 0 and model is not None:
            try:
              
                results = model(frame, 
                              conf=CONFIDENCE_THRESHOLD, 
                              iou=IOU_THRESHOLD,
                              verbose=False,
                              imgsz=640) 
                
               
                if len(results[0].boxes) > 0:
                    valid_detections = []
                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        
                       
                        x1, y1, x2, y2 = box.xyxy[0]
                        box_area = (x2 - x1) * (y2 - y1)
                        
                    
                        if (conf >= CONFIDENCE_THRESHOLD and 
                            MIN_BOX_AREA <= box_area <= MAX_BOX_AREA):
                            valid_detections.append(box)
                    
                    if len(valid_detections) > 0:
                        # Mengambil deteksi dengan confidence tertinggi
                        best_box = max(valid_detections, key=lambda x: float(x.conf[0]))
                        confidence = float(best_box.conf[0])
                        class_id = int(best_box.cls[0])
                        
                        disease = CLASS_NAMES.get(class_id, f'Unknown_Class_{class_id}')
                        
                        # Cooldown
                        if current_time - last_detection_time >= DETECTION_COOLDOWN:
                            latest_detection = {
                                'status': disease,
                                'confidence': round(confidence * 100, 2),
                                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            
                            threading.Thread(target=save_detection, args=(disease, confidence)).start()
                            last_detection_time = current_time
                            
                            print(f"DETECTED: {disease} - Confidence: {confidence*100:.2f}%")
                        
                        display_frame = draw_detection_box(display_frame, best_box, disease, confidence * 100)
                        detected_frame = display_frame.copy()
                        detection_display_time = current_time
                
            except Exception as e:
                print(f"Detection error: {e}")
        
        if detected_frame is not None and (current_time - detection_display_time) < 3.0:
            display_frame = detected_frame
        
        cv2.putText(display_frame, f"Threshold: {CONFIDENCE_THRESHOLD*100:.0f}%", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if latest_detection['status'] != 'Tidak Ada Deteksi':
            status_text = f"{latest_detection['status']} ({latest_detection['confidence']:.1f}%)"
            cv2.putText(display_frame, status_text, 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Encode
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
        print(f"Saved to database: {disease} ({confidence*100:.2f}%)")
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
    """Endpoint cek informasi model"""
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
    
    connect_camera()
    
    if model is None:
        print("\nWARNING: Model belum di upload.")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)