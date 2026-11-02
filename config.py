# ADAS Driver Monitoring System (DMS) Configuration
# Pure Eye State & Drowsiness Detection (Optimized for Raspberry Pi 4B & Desktop)

import os
from dataclasses import dataclass

@dataclass
class DMSConfig:
    # Camera resolution: 320x240 or 480x360 ensures 30+ FPS on Raspberry Pi 4B with zero thermal throttling
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 320
    FRAME_HEIGHT: int = 240
    TARGET_FPS: int = 30
    
    # Camera backend: "opencv" (V4L2 USB camera) or "picamera2" (Raspberry Pi Camera Module v2/v3)
    CAMERA_TYPE: str = "opencv"
    
    # Headless mode for vehicle integration without display
    HEADLESS: bool = False
    
    # Raspberry Pi hardware GPIO buzzer pin (GPIO 18)
    ENABLE_GPIO: bool = False
    GPIO_BUZZER_PIN: int = 18
    
    # Model assets (MediaPipe Face Landmarker with Neural Blendshapes)
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    FACE_MODEL_PATH: str = os.path.join(MODELS_DIR, "face_landmarker.task")
    FACE_MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    
    # ==========================================
    # EYE STATE & DROWSINESS THRESHOLDS (Time-based & framerate-agnostic)
    # ==========================================
    # Neural eyeBlink blendshape: 0.0 = wide open, 1.0 = fully closed
    BLINK_CLOSED_THRESHOLD: float = 0.50
    
    # Microsleep: continuous closed eyes for >= 1.2s -> Immediate Alarm
    MICROSLEEP_DURATION_SEC: float = 1.2
    
    # PERCLOS (Percentage of Eye Closure over rolling 4.0-second window)
    PERCLOS_WINDOW_SEC: float = 4.0
    PERCLOS_TIRED_THRESHOLD: float = 0.20   # 20% eye closure -> Mild fatigue
    PERCLOS_DROWSY_THRESHOLD: float = 0.35  # 35% eye closure -> Drowsiness warning
    
    # Yawn detection via neural jawOpen blendshape (0.0 = closed, 1.0 = wide open yawn)
    YAWN_JAW_THRESHOLD: float = 0.55        # Mouth opening threshold
    YAWN_MIN_DURATION_SEC: float = 0.7      # Sustained >= 0.7s (filters out talking and chewing)
    YAWN_WINDOW_SECONDS: float = 90.0       # Tracks yawns across rolling 90 seconds
    
    # Driver Unresponsive / Slump / Face Lost threshold
    # If a driver slumps down, passes out, or turns away for >= 2.0s -> Alarm
    FACE_LOST_TIMEOUT_SEC: float = 2.0
    
    # Audio alerts
    AUDIO_ALERTS_ENABLED: bool = True
    AUDIO_MIN_INTERVAL_SEC: float = 1.0
