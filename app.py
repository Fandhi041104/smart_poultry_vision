from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import mysql.connector
from mysql.connector import Error
import datetime
import os
from ultralytics import YOLO
import threading
import time
from pathlib import Path
import numpy as np

app = Flask(__name__)

MODEL_PATH = "models/best.pt"
SNAPSHOT_FOLDER = "static/snapshots"

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

os.makedirs(SNAPSHOT_FOLDER, exist_ok=True)
os.makedirs("models", exist_ok=True)

camera = None
model = None
last_detection_time = 0

latest_detection = {
    'status': 'Tidak Ada Deteksi',
    'confidence': 0,
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'disease_counts': {'Sehat': 0,'Coccidiosis': 0, 'Newcastle': 0, 'Salmonella': 0},
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

MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'poultry_detection',
    'port': 3306,
    'autocommit': True 
}


def get_db_connection():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"Error koneksi MySQL: {e}")
        return None


def init_db():
    conn = get_db_connection()
    if conn:
        print("Terhubung ke MySQL")
        conn.close()
    else:
        print("Gagal terhubung ke MySQL")


init_db()


def save_detection(disease, confidence):
    try:
        conn = get_db_connection()
        if not conn:
            print("Koneksi database gagal")
            return
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = "INSERT INTO detections (disease, confidence, timestamp, image_path) VALUES (%s, %s, %s, %s)"
        cursor.execute(query, (disease, confidence, timestamp, ''))
        cursor.close()
        conn.close()
        print(f"Tersimpan: {disease} - {confidence*100:.2f}%")
    except Error as e:
        print(f"MySQL Error: {e}")


def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Memuat model dari {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model(dummy, verbose=False)
            print("Model berhasil dimuat")
        else:
            print("File model tidak ditemukan")
            model = None
    except Exception as e:
        print(f"Error memuat model: {e}")
        model = None


load_model()


def get_tapo_rtsp_url(ip, username, password):
    return f"rtsp://{username}:{password}@{ip}:554/stream1"


def connect_camera():
    global camera
    try:
        tapo_ip = "10.220.129.99"
        tapo_user = "fandhi"
        tapo_pass = "tugas_akhir"

        rtsp_url = get_tapo_rtsp_url(tapo_ip, tapo_user, tapo_pass)
        print("Menghubungkan ke kamera...")

        camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if camera.isOpened():
            print("Kamera terhubung")
            return True
        else:
            print("Gagal terhubung, menggunakan webcam")
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return camera.isOpened()
    except Exception as e:
        print(f"Error kamera: {e}")
        return False


def create_detection_snapshot(frame, detections_data):
    if len(detections_data) == 0:
        return None

    snapshot = frame.copy()

    color_map = {
        'Sehat': (0, 255, 0),
        'Coccidiosis': (0, 165, 255),
        'Newcastle': (0, 0, 255),
        'Salmonella': (255, 0, 255)
    }

    for det in detections_data:
        box = det['box']
        disease = det['disease']
        confidence = det['confidence']

        x1, y1, x2, y2 = map(int, box.xyxy[0])
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
        snapshot = cv2.resize(snapshot, (int(width * scale), int(height * scale)))

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info_text = f"{timestamp} | Terdeteksi: {len(detections_data)} objek"
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
    print("Stream dimulai")

    while True:
        success, frame = camera.read()
        if not success:
            print("Menghubungkan ulang kamera...")
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

                        if conf >= CONFIDENCE_THRESHOLD and MIN_BOX_AREA <= box_area <= MAX_BOX_AREA:
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
                        print(f"Terdeteksi {len(valid_detections)} objek: {disease_counts}")

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
                                threading.Thread(
                                    target=save_detection,
                                    args=(det['disease'], det['confidence'] / 100),
                                    daemon=True
                                ).start()

                            last_detection_time = current_time
                            print(f"Disimpan: {total} objek - {disease_counts}")

            except Exception as e:
                print(f"Error deteksi: {e}")

        height, width = frame.shape[:2]
        if width > RESIZE_WIDTH:
            scale = RESIZE_WIDTH / width
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        cv2.putText(frame, f"Confidence: {CONFIDENCE_THRESHOLD*100:.0f}%",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if latest_detection.get('total_detected', 0) > 0:
            cv2.putText(frame, f"{latest_detection['status']}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ===== ROUTES =====

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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# API 

@app.route('/api/latest_detection')
def get_latest_detection():
    return jsonify(latest_detection)


@app.route('/api/detection_snapshots')
def get_detection_snapshots():
    with snapshot_lock:
        snapshots = [{'filename': s, 'url': f'/static/snapshots/{s}'}
                     for s in detection_snapshots]
    return jsonify({'snapshots': snapshots})


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


@app.route('/api/statistics')
def get_statistics():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Koneksi database gagal'}), 500

        cursor = conn.cursor(dictionary=True)

        cursor.execute("SELECT COUNT(*) as total FROM detections")
        total = cursor.fetchone()['total']

        cursor.execute("SELECT disease, COUNT(*) as count FROM detections GROUP BY disease")
        by_disease = {row['disease']: row['count'] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT disease, confidence, timestamp
            FROM detections
            ORDER BY id DESC
            LIMIT 10
        """)
        recent = [
            {
                'disease': row['disease'],
                'confidence': round(row['confidence'] * 100, 2),
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for row in cursor.fetchall()
        ]

        cursor.close()
        conn.close()

        return jsonify({'total': total, 'by_disease': by_disease, 'recent': recent})
    except Error as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics_today')
def get_statistics_today():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Koneksi database gagal'}), 500

        cursor = conn.cursor(dictionary=True)
        today = datetime.date.today().strftime('%Y-%m-%d')

        cursor.execute("""
            SELECT disease, COUNT(*) as count
            FROM detections
            WHERE DATE(timestamp) = %s
            GROUP BY disease
        """, (today,))
        today_data = {row['disease']: row['count'] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(*) as total FROM detections")
        total_all = cursor.fetchone()['total']

        cursor.close()
        conn.close()

        return jsonify({'today': today_data, 'total_all': total_all, 'date': today})
    except Error as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/history_all')
def get_history_all():
    try:
        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Koneksi database gagal'}), 500

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT disease, confidence, timestamp FROM detections ORDER BY id DESC")
        history = [
            {
                'disease': row['disease'],
                'confidence': round(row['confidence'] * 100, 2),
                'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            }
            for row in cursor.fetchall()
        ]

        cursor.close()
        conn.close()
        return jsonify(history)
    except Error as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_history')
def export_history():
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from io import BytesIO

        # Ambil parameter filter dari URL
        disease   = request.args.get('disease', '')
        from_date = request.args.get('from', '')
        to_date   = request.args.get('to', '')

        conn = get_db_connection()
        if not conn:
            return jsonify({'error': 'Koneksi database gagal'}), 500

        cursor = conn.cursor(dictionary=True)

        query  = "SELECT disease, confidence, timestamp FROM detections WHERE 1=1"
        params = []

        if disease:
            query += " AND disease = %s"
            params.append(disease)
        if from_date:
            query += " AND DATE(timestamp) >= %s"
            params.append(from_date)
        if to_date:
            query += " AND DATE(timestamp) <= %s"
            params.append(to_date)

        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Buat file Excel
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Riwayat Deteksi"

        headers = ['No', 'Penyakit', 'Tingkat Keyakinan (%)', 'Tanggal', 'Waktu']
        header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)

        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal='center')

        for idx, row in enumerate(rows, 1):
            ts = row['timestamp']
            if hasattr(ts, 'strftime'):
                date_str = ts.strftime('%Y-%m-%d')
                time_str = ts.strftime('%H:%M:%S')
            else:
                parts = str(ts).split(' ')
                date_str = parts[0] if len(parts) > 0 else ''
                time_str = parts[1] if len(parts) > 1 else ''

            ws.cell(row=idx+1, column=1, value=idx)
            ws.cell(row=idx+1, column=2, value=row['disease'])
            ws.cell(row=idx+1, column=3, value=round(row['confidence'] * 100, 2))
            ws.cell(row=idx+1, column=4, value=date_str)
            ws.cell(row=idx+1, column=5, value=time_str)

        ws.column_dimensions['A'].width = 6
        ws.column_dimensions['B'].width = 16
        ws.column_dimensions['C'].width = 22
        ws.column_dimensions['D'].width = 14
        ws.column_dimensions['E'].width = 12

        # Nama file menyertakan info filter
        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if disease:
            suffix = f"{disease}_{suffix}"

        output = BytesIO()
        wb.save(output)
        output.seek(0)

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'riwayat_deteksi_{suffix}.xlsx'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/set_threshold', methods=['POST'])
def set_threshold():
    global CONFIDENCE_THRESHOLD
    try:
        data = request.get_json()
        new_threshold = float(data.get('threshold', 0.45))
        if 0.1 <= new_threshold <= 1.0:
            CONFIDENCE_THRESHOLD = new_threshold
            return jsonify({'success': True, 'threshold': CONFIDENCE_THRESHOLD})
        return jsonify({'success': False, 'error': 'Nilai harus antara 0.1 - 1.0'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("Sistem Deteksi Penyakit Ayam Broiler")
    print("=" * 70)
    print(f"Inference Size    : {INFERENCE_SIZE}px")
    print(f"Confidence        : {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"Deteksi setiap    : {DETECTION_FRAME_SKIP} frame")
    print(f"Resolusi stream   : {RESIZE_WIDTH}px")
    print(f"Database          : MySQL (autocommit aktif)")
    print("=" * 70)

    connect_camera()

    if model is None:
        print("PERINGATAN: Model tidak berhasil dimuat")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)