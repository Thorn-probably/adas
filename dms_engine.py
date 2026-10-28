"""
ADAS Driver Monitoring System (DMS) Engine
Optimized for Raspberry Pi 4B (ARM Cortex-A72) and Desktop.

Tracks:
1. Drowsiness: Neural Blendshape Eyelid Closure, PERCLOS, Yawn Accumulator, Microsleep
2. Distraction: Neural Pupil Gaze Vector (Down/Left/Right) + 3D Head Pose (Pitch, Yaw, Roll)
3. Sobriety: Resting Postural Tremor & Sluggish Blink Multi-Factor AND-Gate
"""

import time
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import DMSConfig


@dataclass
class DMSFrameResult:
    face_detected: bool = False
    
    # 1. Drowsiness Metrics
    blink_score: float = 0.0          # 0.0 = open, 1.0 = fully closed
    is_eye_closed: bool = False
    eye_closed_frames: int = 0
    perclos: float = 0.0
    jaw_open: float = 0.0             # 0.0 = shut, 1.0 = yawn
    is_yawning: bool = False
    yawn_count_window: int = 0
    drowsiness_state: str = "ALERT"   # "ALERT", "TIRED", "DROWSY", "MICROSLEEP"
    
    # 2. Attention & Gaze Metrics
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze_down: float = 0.0
    gaze_left: float = 0.0
    gaze_right: float = 0.0
    head_direction: str = "ROAD"
    is_distracted: bool = False
    
    # 3. Sobriety Metrics
    resting_sway_std: float = 0.0
    sobriety_state: str = "SOBER"     # "SOBER" or "IMPAIRED"
    
    # Active alarms
    active_alarms: List[str] = field(default_factory=list)


class DMSEngine:
    def __init__(self, config: Optional[DMSConfig] = None):
        self.config = config or DMSConfig()
        self.frame_idx = 0
        
        # Load MediaPipe Face Landmarker (TFLite with XNNPACK on ARM)
        self._init_mediapipe_face()
        
        # Optional Custom-trained TinyDMSNet model
        self.custom_model = None
        self._init_custom_model()
        
        # Posture zero-calibration
        self.calib_pitch = 0.0
        self.calib_yaw = 0.0
        self.calib_roll = 0.0
        
        # Temporal buffers
        self.blink_history = deque(maxlen=self.config.PERCLOS_WINDOW_FRAMES)
        self.resting_sway_history = deque(maxlen=self.config.SOBRIETY_WINDOW_FRAMES)
        self.recent_yawns = deque(maxlen=10)
        self.blink_durations = deque(maxlen=30)
        
        # State counters
        self.eye_closed_counter = 0
        self.current_blink_duration = 0
        self.yawn_frame_counter = 0
        self.distraction_counter = 0
        self.sobriety_alarm_counter = 0

    def _init_mediapipe_face(self):
        face_base = python.BaseOptions(model_asset_path=self.config.FACE_MODEL_PATH)
        face_opts = vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

    def _init_custom_model(self):
        if self.config.ENGINE_TYPE == "custom" and os.path.isfile(self.config.CUSTOM_MODEL_PATH):
            try:
                import torch
                from train_dms_model import TinyDMSNet
                self.custom_model = TinyDMSNet(num_classes=4)
                self.custom_model.load_state_dict(torch.load(self.config.CUSTOM_MODEL_PATH, map_location="cpu"))
                self.custom_model.eval()
                print(f"[DMS] Loaded custom TinyDMSNet model from {self.config.CUSTOM_MODEL_PATH}")
            except Exception as e:
                print(f"[DMS] Could not load custom model ({e}), using MediaPipe default.")
                self.custom_model = None

    def calibrate_baseline(self):
        recent = list(self.resting_sway_history)[-12:]
        if len(recent) > 0:
            self.calib_pitch = float(np.mean([p[0] for p in recent]))
            self.calib_yaw = float(np.mean([p[1] for p in recent]))
            self.calib_roll = float(np.mean([p[2] for p in recent]))

    def process_frame(self, frame_bgr: np.ndarray) -> DMSFrameResult:
        self.frame_idx += 1
        res = DMSFrameResult()
        h, w = frame_bgr.shape[:2]
        now = time.time()
        
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # Single lightweight neural face detection per frame
        face_res = self.face_landmarker.detect(mp_img)
        
        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            res.face_detected = True
            
            # --- EXTRACT NEURAL BLENDSHAPES ---
            bs_dict: Dict[str, float] = {}
            if face_res.face_blendshapes and len(face_res.face_blendshapes) > 0:
                bs_dict = {cat.category_name: cat.score for cat in face_res.face_blendshapes[0]}

            # (1) DROWSINESS & EYE CLOSURE
            blink_l = bs_dict.get("eyeBlinkLeft", 0.0)
            blink_r = bs_dict.get("eyeBlinkRight", 0.0)
            res.blink_score = (blink_l + blink_r) / 2.0
            
            res.is_eye_closed = res.blink_score >= self.config.BLINK_CLOSED_THRESHOLD
            self.blink_history.append(1.0 if res.is_eye_closed else 0.0)
            
            if res.is_eye_closed:
                self.eye_closed_counter += 1
                self.current_blink_duration += 1
            else:
                if self.current_blink_duration > 0:
                    self.blink_durations.append(self.current_blink_duration)
                    self.current_blink_duration = 0
                self.eye_closed_counter = 0
            res.eye_closed_frames = self.eye_closed_counter

            # PERCLOS over rolling window
            if len(self.blink_history) > 10:
                res.perclos = float(np.mean(self.blink_history))

            # (2) YAWN DETECTION (jawOpen)
            res.jaw_open = bs_dict.get("jawOpen", 0.0)
            if res.jaw_open >= self.config.YAWN_JAW_THRESHOLD:
                self.yawn_frame_counter += 1
                if self.yawn_frame_counter >= self.config.YAWN_CONSEC_FRAMES:
                    res.is_yawning = True
                    if len(self.recent_yawns) == 0 or (now - self.recent_yawns[-1] > 6.0):
                        self.recent_yawns.append(now)
            else:
                self.yawn_frame_counter = 0

            # Prune old yawns (> 90s)
            valid_yawns = [t for t in self.recent_yawns if (now - t) <= self.config.YAWN_WINDOW_SECONDS]
            self.recent_yawns = deque(valid_yawns, maxlen=10)
            res.yawn_count_window = len(self.recent_yawns)

            # (3) 3D HEAD POSE (from 4x4 rigid transformation matrix)
            if face_res.facial_transformation_matrixes and len(face_res.facial_transformation_matrixes) > 0:
                T = np.array(face_res.facial_transformation_matrixes[0])
                R = T[:3, :3]
                raw_pitch = float(np.degrees(np.arctan2(-R[1, 2], R[2, 2])))
                raw_yaw = float(np.degrees(np.arctan2(R[0, 2], np.sqrt(R[1, 2]**2 + R[2, 2]**2))))
                raw_roll = float(np.degrees(np.arctan2(-R[0, 1], R[0, 0])))
            else:
                raw_pitch, raw_yaw, raw_roll = 0.0, 0.0, 0.0

            res.pitch = raw_pitch - self.calib_pitch
            res.yaw = raw_yaw - self.calib_yaw
            res.roll = raw_roll - self.calib_roll

            # (4) NEURAL PUPIL GAZE VECTOR
            res.gaze_down = (bs_dict.get("eyeLookDownLeft", 0.0) + bs_dict.get("eyeLookDownRight", 0.0)) / 2.0
            res.gaze_left = (bs_dict.get("eyeLookOutLeft", 0.0) + bs_dict.get("eyeLookInRight", 0.0)) / 2.0
            res.gaze_right = (bs_dict.get("eyeLookInLeft", 0.0) + bs_dict.get("eyeLookOutRight", 0.0)) / 2.0

            direction, distracted = self._classify_attention(
                res.pitch, res.yaw, res.gaze_down, res.gaze_left, res.gaze_right
            )
            res.head_direction = direction

            if distracted:
                self.distraction_counter += 1
            else:
                self.distraction_counter = max(0, self.distraction_counter - 2)
            res.is_distracted = self.distraction_counter >= self.config.DISTRACTION_CONSEC_FRAMES

            # (5) MULTI-FACTOR DROWSINESS STATE
            if res.eye_closed_frames >= self.config.MICROSLEEP_FRAMES:
                res.drowsiness_state = "MICROSLEEP"
            elif res.perclos >= self.config.PERCLOS_DROWSY_THRESHOLD or res.yawn_count_window >= 2:
                res.drowsiness_state = f"DROWSY ({res.yawn_count_window} YAWNS)" if res.yawn_count_window >= 2 else "DROWSY"
            elif res.perclos >= self.config.PERCLOS_TIRED_THRESHOLD or res.yawn_count_window == 1:
                res.drowsiness_state = "TIRED (1 YAWN)" if res.yawn_count_window == 1 else "MILD FATIGUE"
            else:
                res.drowsiness_state = "ALERT"

            # (6) MULTI-FACTOR SOBRIETY (AND-gated resting tremor + slow blinks)
            is_facing_road = (abs(res.yaw) <= self.config.SOBRIETY_RESTING_YAW_MAX and 
                              abs(res.pitch) <= self.config.SOBRIETY_RESTING_PITCH_MAX)
            if is_facing_road:
                self.resting_sway_history.append((res.pitch, res.yaw, res.roll))

            if len(self.resting_sway_history) >= 40:
                hist = np.array(self.resting_sway_history)
                resting_sway = float((np.std(hist[:, 0]) + np.std(hist[:, 1]) + np.std(hist[:, 2])) / 3.0)
                res.resting_sway_std = resting_sway

                slow_blinks = len([b for b in self.blink_durations if b >= 10])
                is_sway_bad = resting_sway >= self.config.SOBRIETY_SWAY_STD_THRESHOLD
                has_sluggish = slow_blinks >= self.config.SOBRIETY_MIN_SLOW_BLINKS

                if is_sway_bad and has_sluggish:
                    self.sobriety_alarm_counter += 1
                else:
                    self.sobriety_alarm_counter = max(0, self.sobriety_alarm_counter - 1)

                res.sobriety_state = "IMPAIRED" if self.sobriety_alarm_counter >= self.config.SOBRIETY_CONSEC_FRAMES else "SOBER"
            else:
                res.sobriety_state = "SOBER"

        # ====================================================
        # ALARM ARBITRATION
        # ====================================================
        alarms = []
        if res.drowsiness_state == "MICROSLEEP":
            alarms.append("MICROSLEEP")
        elif res.is_distracted:
            alarms.append("DISTRACTION")
        elif "DROWSY" in res.drowsiness_state:
            alarms.append("DROWSY")

        if res.sobriety_state == "IMPAIRED":
            alarms.append("IMPAIRMENT")

        res.active_alarms = alarms
        return res

    def _classify_attention(self, pitch: float, yaw: float, down: float, left: float, right: float) -> Tuple[str, bool]:
        if down >= self.config.GAZE_LOOK_DOWN_THRESHOLD or pitch < self.config.HEAD_PITCH_DOWN:
            return "DOWN (PHONE)", True
        elif left >= self.config.GAZE_LOOK_SIDE_THRESHOLD or yaw < self.config.HEAD_YAW_LEFT:
            return "LOOKING LEFT", True
        elif right >= self.config.GAZE_LOOK_SIDE_THRESHOLD or yaw > self.config.HEAD_YAW_RIGHT:
            return "LOOKING RIGHT", True
        elif pitch > self.config.HEAD_PITCH_UP:
            return "LOOKING UP", True
        return "ROAD", False
