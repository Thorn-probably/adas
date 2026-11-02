"""
ADAS Driver Monitoring System (DMS) Engine
Specialized exclusively for:
1. Eye Detection & Openness Tracking (eyeBlinkLeft, eyeBlinkRight)
2. Drowsiness, Fatigue & Microsleep Detection (PERCLOS, Yawn Accumulator, Microsleep)

Runs at 35+ FPS on Raspberry Pi 4B CPU with MediaPipe TFLite + XNNPACK.
"""

import time
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import DMSConfig


@dataclass
class DMSFrameResult:
    face_detected: bool = False
    face_lost_duration: float = 0.0   # Seconds elapsed since face was lost
    
    # Eye Metrics
    blink_score: float = 0.0          # 0.0 = wide open, 1.0 = fully closed
    eye_openness: float = 1.0         # 1.0 = 100% open, 0.0 = 0% open
    is_eye_closed: bool = False
    eye_closed_duration: float = 0.0  # Continuous eye closure time in seconds
    eye_closed_frames: int = 0
    perclos: float = 0.0              # 0.0 to 1.0 (% time eyes closed in rolling window)
    
    # Yawn / Fatigue Metrics
    jaw_open: float = 0.0             # 0.0 = shut, 1.0 = wide open yawn
    is_yawning: bool = False
    yawn_count_window: int = 0        # Number of yawns in the last 90 seconds
    
    # Drowsiness State: "ALERT", "TIRED", "DROWSY", "MICROSLEEP", "SEARCHING", "UNRESPONSIVE"
    drowsiness_state: str = "ALERT"
    
    # Active alarms
    active_alarms: List[str] = field(default_factory=list)


class DMSEngine:
    def __init__(self, config: Optional[DMSConfig] = None):
        self.config = config or DMSConfig()
        self.frame_idx = 0
        
        # Initialize Google MediaPipe Face Landmarker with Neural Blendshapes
        face_base = python.BaseOptions(model_asset_path=self.config.FACE_MODEL_PATH)
        face_opts = vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
        
        # Temporal buffers (timestamp-based for framerate invariance)
        self.blink_time_history = deque()
        self.recent_yawns = deque(maxlen=10)
        
        # Timestamps for duration tracking
        self.eye_closed_start_time: Optional[float] = None
        self.face_lost_start_time: Optional[float] = None
        self.yawn_start_time: Optional[float] = None
        
        # State counters
        self.eye_closed_counter = 0

    def process_frame(self, frame_bgr: np.ndarray) -> DMSFrameResult:
        self.frame_idx += 1
        res = DMSFrameResult()
        now = time.time()
        
        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Single ultra-fast neural inference (<10ms)
        face_res = self.face_landmarker.detect(mp_img)
        
        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            res.face_detected = True
            self.face_lost_start_time = None
            res.face_lost_duration = 0.0
            
            # --- 1. EXTRACT NEURAL BLENDSHAPES ---
            bs_dict: Dict[str, float] = {}
            if face_res.face_blendshapes and len(face_res.face_blendshapes) > 0:
                bs_dict = {cat.category_name: cat.score for cat in face_res.face_blendshapes[0]}

            # --- 2. EYE DETECTION & OPENNESS TRACKING ---
            blink_l = bs_dict.get("eyeBlinkLeft", 0.0)
            blink_r = bs_dict.get("eyeBlinkRight", 0.0)
            res.blink_score = float((blink_l + blink_r) / 2.0)
            res.eye_openness = float(max(0.0, min(1.0, 1.0 - res.blink_score)))
            
            # Eye closed boolean flag & duration
            res.is_eye_closed = res.blink_score >= self.config.BLINK_CLOSED_THRESHOLD
            if res.is_eye_closed:
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = now
                res.eye_closed_duration = now - self.eye_closed_start_time
                self.eye_closed_counter += 1
            else:
                self.eye_closed_start_time = None
                self.eye_closed_counter = 0
                res.eye_closed_duration = 0.0
            res.eye_closed_frames = self.eye_closed_counter

            # Framerate-agnostic PERCLOS (% time eyes closed over rolling window)
            self.blink_time_history.append((now, 1.0 if res.is_eye_closed else 0.0))
            cutoff = now - self.config.PERCLOS_WINDOW_SEC
            while self.blink_time_history and self.blink_time_history[0][0] < cutoff:
                self.blink_time_history.popleft()
                
            if len(self.blink_time_history) >= 5:
                res.perclos = float(np.mean([val for _, val in self.blink_time_history]))

            # --- 3. YAWN DETECTION (jawOpen) ---
            res.jaw_open = float(bs_dict.get("jawOpen", 0.0))
            if res.jaw_open >= self.config.YAWN_JAW_THRESHOLD:
                if self.yawn_start_time is None:
                    self.yawn_start_time = now
                yawn_dur = now - self.yawn_start_time
                # Must sustain open mouth >= YAWN_MIN_DURATION_SEC (~0.7s) to filter out speech/smiling
                if yawn_dur >= self.config.YAWN_MIN_DURATION_SEC:
                    res.is_yawning = True
                    # Record timestamp if not already logged in last 6 seconds
                    if len(self.recent_yawns) == 0 or (now - self.recent_yawns[-1] > 6.0):
                        self.recent_yawns.append(now)
            else:
                self.yawn_start_time = None

            # Prune old yawns (> 90 seconds)
            valid_yawns = [t for t in self.recent_yawns if (now - t) <= self.config.YAWN_WINDOW_SECONDS]
            self.recent_yawns = deque(valid_yawns, maxlen=10)
            res.yawn_count_window = len(self.recent_yawns)

            # --- 4. DROWSINESS STATE ARBITRATION ---
            if res.eye_closed_duration >= self.config.MICROSLEEP_DURATION_SEC:
                res.drowsiness_state = "MICROSLEEP"
                res.active_alarms.append("MICROSLEEP")
            elif res.perclos >= self.config.PERCLOS_DROWSY_THRESHOLD or res.yawn_count_window >= 2:
                res.drowsiness_state = f"DROWSY ({res.yawn_count_window} YAWNS)" if res.yawn_count_window >= 2 else "DROWSY"
                res.active_alarms.append("DROWSY")
            elif res.perclos >= self.config.PERCLOS_TIRED_THRESHOLD or res.yawn_count_window == 1:
                res.drowsiness_state = "TIRED (1 YAWN)" if res.yawn_count_window == 1 else "MILD FATIGUE"
            else:
                res.drowsiness_state = "ALERT"

        else:
            # Face not detected: handle driver slump / look away / passed out
            res.face_detected = False
            self.eye_closed_start_time = None
            self.eye_closed_counter = 0
            self.yawn_start_time = None
            
            if self.face_lost_start_time is None:
                self.face_lost_start_time = now
            res.face_lost_duration = now - self.face_lost_start_time
            
            # If face is lost for longer than timeout (e.g. 2.0s), trigger emergency alarm
            if res.face_lost_duration >= self.config.FACE_LOST_TIMEOUT_SEC:
                res.drowsiness_state = "UNRESPONSIVE"
                res.active_alarms.append("UNRESPONSIVE")
            else:
                res.drowsiness_state = "SEARCHING"

        return res
