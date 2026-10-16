# ADAS Driver Monitoring System (DMS) Configuration

import os
import torch
from dataclasses import dataclass

@dataclass
class DMSConfig:
    # Camera resolution: 640x480 standard for 30 FPS without USB hardware latency
    CAMERA_INDEX: int = 0
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 480
    TARGET_FPS: int = 30
    
    # GPU Device
    USE_GPU: bool = torch.cuda.is_available()
    GPU_DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Driver position: "right" for RHD (India, UK) or "left" for LHD (USA, Europe)
    DRIVER_SEAT_SIDE: str = "right"  # Toggle with 'S' key
    
    # Neural model assets
    MODELS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    FACE_MODEL_PATH: str = os.path.join(MODELS_DIR, "face_landmarker.task")
    POSE_MODEL_PATH: str = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")
    
    FACE_MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    POSE_MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    
    # ==========================================
    # (A) NEURAL BLENDSHAPE SLEEP & EYE THRESHOLDS
    # ==========================================
    # Google neural blendshape eyeBlink score (0.0 = wide open, 1.0 = fully closed)
    BLINK_CLOSED_THRESHOLD: float = 0.50
    MICROSLEEP_FRAMES: int = 40             # ~1.3s of continuous closed eyes -> Critical Alert
    PERCLOS_WINDOW_FRAMES: int = 150        # 5-second sliding window
    PERCLOS_TIRED_THRESHOLD: float = 0.20
    PERCLOS_DROWSY_THRESHOLD: float = 0.35
    
    # Yawn detection via neural jawOpen blendshape (0.0 to 1.0)
    YAWN_JAW_THRESHOLD: float = 0.55        # Sustained open mouth
    YAWN_CONSEC_FRAMES: int = 22            # Must hold >= 0.7s to ignore talking/smiling
    YAWN_WINDOW_SECONDS: float = 90.0
    
    # ==========================================
    # (B) NEURAL GAZE & DISTRACTION THRESHOLDS
    # ==========================================
    # Blendshape pupil gaze thresholds
    GAZE_LOOK_DOWN_THRESHOLD: float = 0.48  # Eyes looking down at phone/lap
    GAZE_LOOK_SIDE_THRESHOLD: float = 0.48  # Eyes looking away
    
    # Head Pose rotation limits (degrees)
    HEAD_YAW_LEFT: float = -20.0
    HEAD_YAW_RIGHT: float = 20.0
    HEAD_PITCH_DOWN: float = -14.0
    HEAD_PITCH_UP: float = 18.0
    DISTRACTION_CONSEC_FRAMES: int = 45     # ~1.5s looking away -> Alert
    
    # ==========================================
    # (C) MULTI-FACTOR SOBRIETY / IMPAIRMENT
    # ==========================================
    SOBRIETY_RESTING_YAW_MAX: float = 14.0  # Only evaluate resting sway when looking forward
    SOBRIETY_RESTING_PITCH_MAX: float = 14.0
    SOBRIETY_WINDOW_FRAMES: int = 150
    SOBRIETY_SWAY_STD_THRESHOLD: float = 8.0 # High resting tremor threshold
    SOBRIETY_MIN_SLOW_BLINKS: int = 3       # Requires co-occurring sluggish blinks
    SOBRIETY_CONSEC_FRAMES: int = 45
    
    # ==========================================
    # (D) SEATBELT
    # ==========================================
    SEATBELT_SIMULATION_OVERRIDE: bool = False # Can be toggled with 'T' for desk testing
    SEATBELT_UNFASTENED_DEBOUNCE: int = 25
    
    # AUDIO WARNINGS
    AUDIO_ALERTS_ENABLED: bool = True
    AUDIO_MIN_INTERVAL_SEC: float = 1.2
