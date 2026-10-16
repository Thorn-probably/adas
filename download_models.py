"""
Model downloader utility for MediaPipe Tasks
"""

import os
import urllib.request
from config import DMSConfig

def ensure_models_exist():
    os.makedirs(DMSConfig.MODELS_DIR, exist_ok=True)
    
    models = [
        ("Face Landmarker", DMSConfig.FACE_MODEL_PATH, DMSConfig.FACE_MODEL_URL),
        ("Pose Landmarker", DMSConfig.POSE_MODEL_PATH, DMSConfig.POSE_MODEL_URL)
    ]
    
    for name, path, url in models:
        if not os.path.exists(path) or os.path.getsize(path) < 1000:
            print(f"[DMS] Downloading {name} model from {url}...")
            urllib.request.urlretrieve(url, path)
            print(f"[DMS] Saved {name} ({os.path.getsize(path) / 1024 / 1024:.2f} MB) to {path}")
        else:
            print(f"[DMS] {name} model verified ({os.path.getsize(path) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    ensure_models_exist()
