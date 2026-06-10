from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import mysql.connector
from mysql.connector import Error
import datetime
import os
from ultralytics import YOLO
import threading
import time
import numpy as np

app = Flask(__name__)

MODEL_PATH       = "models/best.pt"
SNAPSHOT_FOLDER  = "static/snapshots"
RECORDING_FOLDER = "static/recordings"

CONFIDENCE_THRESHOLD = 0.45
DETECTION_COOLDOWN   = 2
MIN_BOX_AREA         = 1500
MAX_BOX_AREA         = 800000
IOU_THRESHOLD        = 0.3
DETECTION_FRAME_SKIP = 8
RESIZE_WIDTH         = 720
JPEG_QUALITY         = 78
INFERENCE_SIZE       = 640
SNAPSHOT_SIZE        = 640
MAX_SNAPSHOTS        = 6

os.makedirs(SNAPSHOT_FOLDER,  exist_ok=True)
os.makedirs(RECORDING_FOLDER, exist_ok=True)
os.makedirs("models",         exist_ok=True)

# ── Global state ──────────────────────────────────────────────
camera             = None
model              = None
last_detection_time = 0

latest_detection = {
    'status': 'Tidak Ada Deteksi',
    'confidence': 0,
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'disease_counts': {'Sehat': 0, 'Coccidiosis': 0, 'Newcastle': 0, 'Salmonella': 0},
    'total_detected': 0
}

detection_snapshots = []
snapshot_lock       = threading.Lock()

# ── Recording state ───────────────────────────────────────────
is_recording       = False
video_writer       = None
recording_filename = ''
recording_start    = None
recording_lock     = threading.Lock()
display_detections = []
display_lock       = threading.Lock()
last_frame_wh      = (RESIZE_WIDTH, 405)
recording_frame_w  = RESIZE_WIDTH
recording_frame_h  = 0        # ditentukan dari frame pertama

CLASS_NAMES = {0: 'Sehat', 1: 'Coccidiosis', 2: 'Newcastle', 3: 'Salmonella'}

MYSQL_CONFIG = {
    'host':        'localhost',
    'user':        'root',
    'password':    '',
    'database':    'poultry_detection',
    'port':        3306,
    'autocommit':  True
}


# ── DB helpers ────────────────────────────────────────────────
def get_db_connection():
    try:
        return mysql.connector.connect(**MYSQL_CONFIG)
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


def save_detection(disease, confidence, snapshot_filename=''):
    try:
        conn = get_db_connection()
        if not conn:
            return
        cursor = conn.cursor()
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        query = """INSERT INTO detections (disease, confidence, timestamp, image_path)
                   VALUES (%s, %s, %s, %s)"""
        cursor.execute(query, (disease, confidence, timestamp, snapshot_filename))
        cursor.close()
        conn.close()
        print(f"Tersimpan: {disease} - {confidence*100:.2f}% - {snapshot_filename}")
    except Error as e:
        print(f"MySQL Error: {e}")


# ── Model ──────────────────────────────────────────────────────
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


# ── Camera ────────────────────────────────────────────────────
def get_tapo_rtsp_url(ip, user, pwd):
    return f"rtsp://{user}:{pwd}@{ip}:554/stream1"


def connect_camera():
    global camera
    try:
        tapo_ip   = "10.145.218.99"
        tapo_user = "fandhi"
        tapo_pass = "tugas_akhir"

        rtsp_url = get_tapo_rtsp_url(tapo_ip, tapo_user, tapo_pass)
        print("Menghubungkan ke kamera...")
        camera = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if camera.isOpened(): 
            print("Kamera terhubung")
            return True

        print("Gagal terhubung ke IP Camera, mencoba webcam...")
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if camera.isOpened():
            print("Webcam terhubung")
            return True

        print("Webcam juga gagal")
        return False

    except Exception as e:
        print(f"Error kamera: {e}")
        # Coba webcam sebagai fallback terakhir
        try:
            camera = cv2.VideoCapture(0)
            if camera.isOpened():
                print("Fallback ke webcam berhasil")
                return True
        except:
            pass
        return False


# ── Snapshot ──────────────────────────────────────────────────
def create_detection_snapshot(frame, detections_data):
    if not detections_data:
        return None

    snapshot  = frame.copy()
    color_map = {
        'Sehat':       (0, 255, 0),
        'Coccidiosis': (0, 165, 255),
        'Newcastle':   (0, 0, 255),
        'Salmonella':  (255, 0, 255)
    }

    for det in detections_data:
        box        = det['box']
        disease    = det['disease']
        confidence = det['confidence']
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = color_map.get(disease, (255, 255, 255))

        cv2.rectangle(snapshot, (x1, y1), (x2, y2), color, 4)
        label = f"{disease}: {confidence:.1f}%"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        cv2.rectangle(snapshot, (x1, y1-h-15), (x1+w+10, y1), color, -1)
        cv2.putText(snapshot, label, (x1+5, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    ht, wd = snapshot.shape[:2]
    if wd > SNAPSHOT_SIZE:
        scale    = SNAPSHOT_SIZE / wd
        snapshot = cv2.resize(snapshot, (int(wd*scale), int(ht*scale)))

    info = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {len(detections_data)} objek"
    cv2.putText(snapshot, info, (10, snapshot.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    filename = f"detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(os.path.join(SNAPSHOT_FOLDER, filename), snapshot,
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    return filename


# ── Recording helpers ─────────────────────────────────────────
def start_video_writer():
    global video_writer, recording_filename, recording_start
    w, h = last_frame_wh
    ts               = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    recording_filename = f"rekaman_{ts}.mp4"
    filepath         = os.path.join(RECORDING_FOLDER, recording_filename)
    fourcc           = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer     = cv2.VideoWriter(filepath, fourcc, 15.0, (w, h))
    recording_start  = datetime.datetime.now()
    print(f"Rekaman dimulai: {filepath}")


def stop_video_writer():
    global video_writer, is_recording
    with recording_lock:
        if video_writer:
            video_writer.release()
            video_writer = None
        is_recording = False
    print(f"Rekaman dihentikan: {recording_filename}")


# ── Main frame generator ──────────────────────────────────────
def generate_frames():
    global camera, model, latest_detection, last_detection_time
    global detection_snapshots, is_recording, video_writer
    global display_detections, last_frame_wh

    if camera is None or not camera.isOpened():
        connect_camera()

    frame_count = 0
    color_map   = {
        'Sehat':       (0, 255, 0),
        'Coccidiosis': (0, 165, 255),
        'Newcastle':   (0, 0, 255),
        'Salmonella':  (255, 0, 255)
    }
    print("Stream dimulai")

    while True:
        success, frame = camera.read()
        if not success:
            print("Menghubungkan ulang kamera...")
            time.sleep(0.1)
            connect_camera()
            continue

        frame_count    += 1
        current_time    = time.time()
        original_frame  = frame.copy()

        # Hitung scale dari resolusi asli ke resolusi display
        h_orig, w_orig  = frame.shape[:2]
        scale_display   = RESIZE_WIDTH / w_orig if w_orig > RESIZE_WIDTH else 1.0

        # ── Deteksi YOLO ──────────────────────────────────
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
                    detections_data  = []
                    disease_counts   = {'Sehat': 0, 'Coccidiosis': 0,
                                        'Newcastle': 0, 'Salmonella': 0}

                    for box in results[0].boxes:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0]
                        box_area = (x2 - x1) * (y2 - y1)

                        if conf >= CONFIDENCE_THRESHOLD and MIN_BOX_AREA <= box_area <= MAX_BOX_AREA:
                            class_id = int(box.cls[0])
                            disease  = CLASS_NAMES.get(class_id, f'Unknown_{class_id}')
                            valid_detections.append(box)
                            detections_data.append({
                                'box': box, 'disease': disease,
                                'confidence': conf * 100
                            })
                            disease_counts[disease] = disease_counts.get(disease, 0) + 1

                    if valid_detections:
                        # Simpan koordinat bbox yang sudah di-scale ke display size
                        with display_lock:
                            display_detections = []
                            for det in detections_data:
                                x1, y1, x2, y2 = map(float, det['box'].xyxy[0])
                                display_detections.append({
                                    'x1':        int(x1 * scale_display),
                                    'y1':        int(y1 * scale_display),
                                    'x2':        int(x2 * scale_display),
                                    'y2':        int(y2 * scale_display),
                                    'disease':    det['disease'],
                                    'confidence': det['confidence'],
                                    'color':      color_map.get(det['disease'], (255,255,255))
                                })

                        if current_time - last_detection_time >= DETECTION_COOLDOWN:
                            snapshot_filename = create_detection_snapshot(original_frame, detections_data)

                            if snapshot_filename:
                                with snapshot_lock:
                                    detection_snapshots.insert(0, snapshot_filename)
                                    if len(detection_snapshots) > MAX_SNAPSHOTS:
                                        old = detection_snapshots.pop()
                                        old_path = os.path.join(SNAPSHOT_FOLDER, old)
                                        if os.path.exists(old_path):
                                            os.remove(old_path)

                            total        = len(detections_data)
                            main_disease = max(disease_counts, key=disease_counts.get)
                            main_count   = disease_counts[main_disease]

                            latest_detection = {
                                'status':         f"{main_disease} ({main_count}x)" if total > 1 else main_disease,
                                'confidence':     round(max(d['confidence'] for d in detections_data), 2),
                                'timestamp':      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'disease_counts':  disease_counts,
                                'total_detected':  total,
                                'snapshot':        snapshot_filename
                            }

                            for det in detections_data:
                                threading.Thread(
                                    target=save_detection,
                                    args=(det['disease'], det['confidence']/100,
                                          snapshot_filename or ''),
                                    daemon=True
                                ).start()

                            last_detection_time = current_time
                            print(f"Disimpan: {total} objek - {disease_counts}")

            except Exception as e:
                print(f"Error deteksi: {e}")

        # ── Resize frame untuk display ──────────────────────
        if w_orig > RESIZE_WIDTH:
            frame = cv2.resize(frame, None, fx=scale_display, fy=scale_display)

        # Simpan ukuran frame aktual untuk start_recording
        h_disp, w_disp = frame.shape[:2]
        last_frame_wh  = (w_disp, h_disp)

        # ── Teks overlay stream ─────────────────────────────
        cv2.putText(frame, f"Confidence: {CONFIDENCE_THRESHOLD*100:.0f}%",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        if latest_detection.get('total_detected', 0) > 0:
            cv2.putText(frame, latest_detection['status'],
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ── Tulis ke video rekaman (dengan bounding box) ────
        if is_recording:
            elapsed = ''
            if recording_start:
                secs    = int((datetime.datetime.now() - recording_start).total_seconds())
                elapsed = f"  {secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"

            cv2.circle(frame, (frame.shape[1]-25, 20), 8, (0,0,255), -1)
            cv2.putText(frame, f"REC{elapsed}",
                        (frame.shape[1]-130, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # Buat frame rekaman dengan bounding box
            rec_frame = frame.copy()
            with display_lock:
                for det in display_detections:
                    color = det['color']
                    cv2.rectangle(rec_frame,
                                  (det['x1'], det['y1']),
                                  (det['x2'], det['y2']), color, 3)
                    label = f"{det['disease']}: {det['confidence']:.1f}%"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                    cv2.rectangle(rec_frame,
                                  (det['x1'], det['y1']-th-12),
                                  (det['x1']+tw+8, det['y1']), color, -1)
                    cv2.putText(rec_frame, label,
                                (det['x1']+4, det['y1']-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            with recording_lock:
                if video_writer and video_writer.isOpened():
                    video_writer.write(rec_frame)

        # ── Encode JPEG untuk stream browser ───────────────
        ret, buffer = cv2.imencode('.jpg', frame,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# ════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════
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


# ════════════════════════════════════════════
#  API
# ════════════════════════════════════════════
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
    return jsonify({'status': 'online' if camera and camera.isOpened() else 'offline'})


@app.route('/api/model_info')
def model_info():
    if model:
        return jsonify({'loaded': True, 'confidence': CONFIDENCE_THRESHOLD,
                        'inference_size': INFERENCE_SIZE})
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
        by_disease = {r['disease']: r['count'] for r in cursor.fetchall()}

        cursor.execute("""SELECT disease, confidence, timestamp
                          FROM detections ORDER BY id DESC LIMIT 10""")
        recent = [{'disease':    r['disease'],
                   'confidence': round(r['confidence']*100, 2),
                   'timestamp':  r['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
                  for r in cursor.fetchall()]

        cursor.close(); conn.close()
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
        today  = datetime.date.today().strftime('%Y-%m-%d')

        cursor.execute("""SELECT disease, COUNT(*) as count FROM detections
                          WHERE DATE(timestamp)=%s GROUP BY disease""", (today,))
        today_data = {r['disease']: r['count'] for r in cursor.fetchall()}

        cursor.execute("SELECT COUNT(*) as total FROM detections")
        total_all = cursor.fetchone()['total']

        cursor.close(); conn.close()
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
        cursor.execute("""SELECT disease, confidence, timestamp, image_path
                          FROM detections ORDER BY id DESC""")
        history = []
        for r in cursor.fetchall():
            snap_url = f"/static/snapshots/{r['image_path']}" if r['image_path'] else ''
            history.append({
                'disease':      r['disease'],
                'confidence':   round(r['confidence']*100, 2),
                'timestamp':    r['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'snapshot_url': snap_url
            })
        cursor.close(); conn.close()
        return jsonify(history)
    except Error as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export_history')
def export_history():
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from io import BytesIO

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
            query += " AND disease=%s"; params.append(disease)
        if from_date:
            query += " AND DATE(timestamp)>=%s"; params.append(from_date)
        if to_date:
            query += " AND DATE(timestamp)<=%s"; params.append(to_date)
        query += " ORDER BY id DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        cursor.close(); conn.close()

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Riwayat Deteksi"

        headers     = ['No','Penyakit','Tingkat Keyakinan (%)','Tanggal','Waktu']
        hfill       = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
        hfont       = Font(color="FFFFFF", bold=True)
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.fill = hfill; cell.font = hfont
            cell.alignment = Alignment(horizontal='center')

        for idx, r in enumerate(rows, 1):
            ts   = r['timestamp']
            date = ts.strftime('%Y-%m-%d') if hasattr(ts,'strftime') else str(ts).split(' ')[0]
            time_ = ts.strftime('%H:%M:%S') if hasattr(ts,'strftime') else str(ts).split(' ')[1]
            ws.cell(row=idx+1, column=1, value=idx)
            ws.cell(row=idx+1, column=2, value=r['disease'])
            ws.cell(row=idx+1, column=3, value=round(r['confidence']*100, 2))
            ws.cell(row=idx+1, column=4, value=date)
            ws.cell(row=idx+1, column=5, value=time_)

        for col, w in zip(['A','B','C','D','E'], [6,16,22,14,12]):
            ws.column_dimensions[col].width = w

        suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if disease: suffix = f"{disease}_{suffix}"

        output = BytesIO()
        wb.save(output); output.seek(0)
        return send_file(output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=f'riwayat_deteksi_{suffix}.xlsx')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ════════════════════════════════════════════
#  RECORDING API
# ════════════════════════════════════════════
@app.route('/api/recording/start', methods=['POST'])
def start_recording():
    global is_recording

    if is_recording:
        return jsonify({'success': False, 'message': 'Rekaman sudah berjalan'})
    if camera is None or not camera.isOpened():
        return jsonify({'success': False, 'message': 'Kamera tidak terhubung'})

    with recording_lock:
        start_video_writer()   # tidak lagi butuh camera.read()
        is_recording = True

    return jsonify({
        'success':  True,
        'message':  'Rekaman dimulai',
        'filename': recording_filename
    })


@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    global is_recording, recording_filename

    if not is_recording:
        return jsonify({'success': False, 'message': 'Tidak ada rekaman yang berjalan'})

    saved_filename = recording_filename
    stop_video_writer()

    return jsonify({
        'success':      True,
        'message':      'Rekaman berhasil disimpan',
        'filename':     saved_filename,
        'download_url': f'/api/recording/download/{saved_filename}'
    })


@app.route('/api/recording/status')
def recording_status():
    elapsed = 0
    if is_recording and recording_start:
        elapsed = int((datetime.datetime.now() - recording_start).total_seconds())
    return jsonify({
        'is_recording': is_recording,
        'filename':     recording_filename,
        'elapsed':      elapsed
    })


@app.route('/api/recording/download/<filename>')
def download_recording(filename):
    """Download file rekaman yang sudah selesai."""
    # Keamanan: hanya izinkan nama file yang valid
    safe_name = os.path.basename(filename)
    filepath  = os.path.join(RECORDING_FOLDER, safe_name)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File tidak ditemukan'}), 404
    return send_file(filepath, as_attachment=True, download_name=safe_name,
                     mimetype='video/mp4')


@app.route('/api/recording/list')
def list_recordings():
    """Daftar semua file rekaman yang tersimpan."""
    files = []
    for f in sorted(os.listdir(RECORDING_FOLDER), reverse=True):
        if f.endswith('.mp4'):
            fpath = os.path.join(RECORDING_FOLDER, f)
            size  = os.path.getsize(fpath)
            files.append({
                'filename':     f,
                'size_mb':      round(size / (1024*1024), 2),
                'download_url': f'/api/recording/download/{f}'
            })
    return jsonify({'recordings': files})

@app.route('/api/snapshots/all')
def get_all_snapshots():
    """Baca semua file snapshot langsung dari folder, bukan dari database."""
    try:
        page     = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 12))
        files    = []

        if os.path.exists(SNAPSHOT_FOLDER):
            for f in sorted(os.listdir(SNAPSHOT_FOLDER), reverse=True):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    timestamp = ''
                    try:
                        # Parse dari nama: detection_YYYYMMDD_HHMMSS.jpg
                        name  = f.replace('detection_', '').replace('.jpg','')
                        parts = name.split('_')
                        if len(parts) >= 2:
                            d = parts[0]  # YYYYMMDD
                            t = parts[1]  # HHMMSS
                            timestamp = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
                    except:
                        pass
                    files.append({
                        'filename':  f,
                        'url':       f'/static/snapshots/{f}',
                        'timestamp': timestamp
                    })

        total      = len(files)
        total_pages = (total + per_page - 1) // per_page
        start      = (page - 1) * per_page
        paged      = files[start:start + per_page]

        return jsonify({
            'snapshots':    paged,
            'total':        total,
            'page':         page,
            'total_pages':  total_pages,
            'per_page':     per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/set_threshold', methods=['POST'])
def set_threshold():
    global CONFIDENCE_THRESHOLD
    try:
        data = request.get_json()
        val  = float(data.get('threshold', 0.45))
        if 0.1 <= val <= 1.0:
            CONFIDENCE_THRESHOLD = val
            return jsonify({'success': True, 'threshold': CONFIDENCE_THRESHOLD})
        return jsonify({'success': False, 'error': 'Nilai harus 0.1–1.0'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("Sistem Deteksi Penyakit Ayam Broiler")
    print("=" * 70)
    print(f"Inference Size   : {INFERENCE_SIZE}px")
    print(f"Confidence       : {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"Deteksi setiap   : {DETECTION_FRAME_SKIP} frame")
    print(f"Rekaman disimpan : {RECORDING_FOLDER}/")
    print(f"Database         : MySQL (autocommit aktif)")
    print("=" * 70)

    connect_camera()
    if model is None:
        print("PERINGATAN: Model tidak berhasil dimuat")

    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)