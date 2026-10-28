# ADAS Driver Monitoring System (DMS) Configuration
# Optimized for Raspberry Pi 4B (ARM Cortex-A72) and Desktop Environments

import os
import platform
from dataclasses import dataclass

@dataclass
class DMSConfig:
    # --- RASPBERRY PI 4B HARDWARE & RESOLUTION ---
    # 320x240 provides guaranteed 30+ FPS on Raspberry Pi 4B with <25% CPU per core and zero thermal throttling.
    # Can also be set to 480x360 or 640x480 if desired.
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 320
    FRAME_HEIGHT: int = 240
    TARGET_FPS: int = 30
    
    # Camera backend: "opencv" (V4L2 USB camera) or "picamera2" (Raspberry Pi Camera Module v2/v3)
    CAMERA_TYPE: str = "opencv"
    
    # Headless mode: True runs without OpenCV GUI display (for embedded Pi deployment in a vehicle)
    HEADLESS: bool = False
    
    # Hardware GPIO buzzer/LED on Raspberry Pi (Pin 18 = GPIO 18, standard PWM/Buzzer pin)
    ENABLE_GPIO: bool = False
    GPIO_BUZZER_PIN: int = 18
    
    # Engine type: "mediapipe" (TFLite with XNNPACK) or "custom" (TinyDMSNet lightweight model)
    ENGINE_TYPE: str = "mediapipe"
    CUSTOM_MODEL_PATH: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "tiny_dms_rpi.pt")
    
    # Model assets
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    FACE_MODEL_PATH: str = os.path.join(MODELS_DIR, "face_landmarker.task")
    FACE_MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    # ==========================================
    # (A) DROWSINESS & SLEEP DETECTION
    # ==========================================
    # Neural blendshape eyeBlink score (0.0 = wide open, 1.0 = fully closed)
    BLINK_CLOSED_THRESHOLD: float = 0.50
    MICROSLEEP_FRAMES: int = 38             # ~1.2s of continuous closed eyes -> Critical Alert
    PERCLOS_WINDOW_FRAMES: int = 120        # ~4-second rolling window
    PERCLOS_TIRED_THRESHOLD: float = 0.20   # 20% closure -> Mild fatigue
    PERCLOS_DROWSY_THRESHOLD: float = 0.35  # 35% closure -> High drowsiness
    
    # Yawn detection via neural jawOpen blendshape (0.0 to 1.0)
    YAWN_JAW_THRESHOLD: float = 0.55        # Sustained open mouth
    YAWN_CONSEC_FRAMES: int = 20            # ~0.7s to avoid counting speech/smiling
    YAWN_WINDOW_SECONDS: float = 90.0
    
    # ==========================================
    # (B) GAZE & DISTRACTION DETECTION
    # ==========================================
    # Neural blendshape pupil gaze thresholds
    GAZE_LOOK_DOWN_THRESHOLD: float = 0.48  # Eyes looking down at phone/lap
    GAZE_LOOK_SIDE_THRESHOLD: float = 0.48  # Eyes looking left/right
    
    # Head Pose rotation limits (degrees)
    HEAD_YAW_LEFT: float = -20.0
    HEAD_YAW_RIGHT: float = 20.0
    HEAD_PITCH_DOWN: float = -14.0
    HEAD_PITCH_UP: float = 18.0
    DISTRACTION_CONSEC_FRAMES: int = 42     # ~1.4s looking away -> Alert
    
    # ==========================================
    # (C) MULTI-FACTOR SOBRIETY / IMPAIRMENT
    # ==========================================
    SOBRIETY_RESTING_YAW_MAX: float = 14.0  # Only evaluate resting sway when looking forward
    SOBRIETY_RESTING_PITCH_MAX: float = 14.0
    SOBRIETY_WINDOW_FRAMES: int = 120
    SOBRIETY_SWAY_STD_THRESHOLD: float = 8.0 # High resting tremor threshold
    SOBRIETY_MIN_SLOW_BLINKS: int = 3       # Requires co-occurring sluggish blinks
    SOBRIETY_CONSEC_FRAMES: int = 40
    
    # ==========================================
    # AUDIO & ALARM SETTINGS
    # ==========================================
    AUDIO_ALERTS_ENABLED: bool = True
    AUDIO_MIN_INTERVAL_SEC: float = 1.0
