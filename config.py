# config.py - Konfigurasi RTSP dan Aplikasi

# ===== RTSP CAMERA CONFIGURATION =====
# Cara mendapatkan RTSP URL Tapo C100:
# 1. Buka aplikasi Tapo di HP
# 2. Pilih kamera Tapo C100
# 3. Cek IP kamera di Settings > Device Info
# 4. Username default: admin
# 5. Password: password yang Anda buat saat setup kamera

# Format RTSP URL untuk Tapo C100:
# rtsp://username:password@ip_address:554/stream1  (High Quality)
# rtsp://username:password@ip_address:554/stream2  (Low Quality - hemat bandwidth)

CAMERA_CONFIG = {
    'ip': '192.168.0.13',          # Ganti dengan IP kamera Anda
    'username': 'fandhi',             # Default Tapo username
    'password': 'fandhi_044011',     # Password kamera Anda
    'rtsp_port': 554,
    'stream': 'stream1'              # stream1 (HD) atau stream2 (SD)
}

# Generate RTSP URL
def get_rtsp_url():
    return f"rtsp://{CAMERA_CONFIG['username']}:{CAMERA_CONFIG['password']}@{CAMERA_CONFIG['ip']}:{CAMERA_CONFIG['rtsp_port']}/{CAMERA_CONFIG['stream']}"

# ===== YOLO MODEL CONFIGURATION =====
YOLO_CONFIG = {
    'model_path': 'models/yolov8n.pt',
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'classes': ['Sehat', 'Coccidiosis', 'Newcastle', 'Salmonella']
}

# ===== DATABASE CONFIGURATION =====
DATABASE_CONFIG = {
    'path': 'database/detections.db'
}

# ===== APP CONFIGURATION =====
APP_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'upload_folder': 'uploads',
    'allowed_extensions': {'png', 'jpg', 'jpeg'}
}

# ===== DETECTION SETTINGS =====
DETECTION_CONFIG = {
    'process_every_n_frames': 30,  # Proses setiap 30 frame (hemat resource)
    'save_detections': True,
    'save_images': False,          # Set True untuk simpan gambar hasil deteksi
    'max_detections_db': 1000      # Maksimal record di database
}

# Print RTSP URL untuk testing
if __name__ == '__main__':
    print("=" * 50)
    print("RTSP URL Configuration")
    print("=" * 50)
    print(f"RTSP URL: {get_rtsp_url()}")
    print()
    print("Test koneksi dengan VLC Media Player:")
    print("1. Buka VLC")
    print("2. Media > Open Network Stream")
    print("3. Paste RTSP URL di atas")
    print("4. Play")
    print("=" * 50)