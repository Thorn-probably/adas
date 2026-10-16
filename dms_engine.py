"""
ADAS Driver Monitoring System (DMS) Engine
Powered by Google MediaPipe Neural Face Blendshapes & 3D Rigid Transformation Matrices.

Directly extracts:
- Invariant Eyelid Closure: eyeBlinkLeft, eyeBlinkRight (0.0 to 1.0)
- Invariant Pupil Gaze: eyeLookDown, eyeLookLeft, eyeLookRight (0.0 to 1.0)
- Invariant Yawn Detection: jawOpen (0.0 to 1.0)
- Invariant 3D Head Pose: Rigid transformation matrix R|T
- Seatbelt Detection with Hough line filtering, strap pairs, contrast & Desk-Test override
"""

import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config import DMSConfig
from gpu_seatbelt import GPUSatbeltDetector


@dataclass
class DMSFrameResult:
    face_detected: bool = False
    pose_detected: bool = False
    
    # 1. Neural Blendshape Eye Metrics
    blink_score: float = 0.0          # 0.0 = wide open, 1.0 = fully closed
    is_eye_closed: bool = False
    eye_closed_frames: int = 0
    perclos: float = 0.0
    
    # 2. Neural Blendshape Mouth / Yawn
    jaw_open: float = 0.0             # 0.0 = shut, 1.0 = yawn
    is_yawning: bool = False
    yawn_count_window: int = 0
    drowsiness_state: str = "ALERT"   # "ALERT", "TIRED", "DROWSY", "MICROSLEEP"
    
    # 3. Neural Blendshape Gaze & Head Pose
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze_down: float = 0.0
    gaze_left: float = 0.0
    gaze_right: float = 0.0
    head_direction: str = "ROAD"
    is_distracted: bool = False
    
    # 4. Sobriety multi-metrics
    resting_sway_std: float = 0.0
    sobriety_state: str = "SOBER"     # "SOBER" or "IMPAIRED"
    
    # 5. Seatbelt
    seatbelt_fastened: bool = True
    seatbelt_confidence: float = 0.0
    seatbelt_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    is_simulation_mode: bool = False
    
    active_alarms: List[str] = field(default_factory=list)


class DMSEngine:
    def __init__(self, config: Optional[DMSConfig] = None):
        self.config = config or DMSConfig()
        
        self.gpu_seatbelt = GPUSatbeltDetector(device=self.config.GPU_DEVICE)
        self._init_mediapipe_models()
        
        self.frame_idx = 0
        self.cached_pose_landmarks = None
        
        # Posture zero-calibration
        self.calib_pitch = 0.0
        self.calib_yaw = 0.0
        self.calib_roll = 0.0
        
        # Simulation override flag for desk testing
        self.seatbelt_sim_override = False
        self.seatbelt_sim_state = True
        
        # Temporal buffers
        self.blink_history = deque(maxlen=self.config.PERCLOS_WINDOW_FRAMES)
        self.resting_sway_history = deque(maxlen=self.config.SOBRIETY_WINDOW_FRAMES)
        self.recent_yawns = deque(maxlen=10)
        self.blink_durations = deque(maxlen=30)
        
        # Counters
        self.eye_closed_counter = 0
        self.current_blink_duration = 0
        self.yawn_frame_counter = 0
        self.distraction_counter = 0
        self.seatbelt_unfastened_counter = 0
        self.sobriety_alarm_counter = 0

    def _init_mediapipe_models(self):
        face_base = python.BaseOptions(model_asset_path=self.config.FACE_MODEL_PATH)
        face_opts = vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=vision.RunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)
        
        pose_base = python.BaseOptions(model_asset_path=self.config.POSE_MODEL_PATH)
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    def calibrate_baseline(self):
        recent = list(self.resting_sway_history)[-12:]
        if len(recent) > 0:
            self.calib_pitch = float(np.mean([p[0] for p in recent]))
            self.calib_yaw = float(np.mean([p[1] for p in recent]))
            self.calib_roll = float(np.mean([p[2] for p in recent]))

    def calibrate_seatbelt(self):
        self.gpu_seatbelt.calibrate_baseline()

    def adjust_seatbelt_sensitivity(self, delta: float):
        self.gpu_seatbelt.set_sensitivity(self.gpu_seatbelt.sensitivity + delta)
        return self.gpu_seatbelt.sensitivity

    def toggle_seat_side(self):
        self.config.DRIVER_SEAT_SIDE = "left" if self.config.DRIVER_SEAT_SIDE == "right" else "right"
        self.seatbelt_unfastened_counter = 0
        return self.config.DRIVER_SEAT_SIDE

    def toggle_seatbelt_simulation(self):
        """Allows desk testing of buckled / unbuckled state without a car seatbelt"""
        self.seatbelt_sim_override = True
        self.seatbelt_sim_state = not self.seatbelt_sim_state
        return self.seatbelt_sim_state

    def process_frame(self, frame_bgr: np.ndarray) -> DMSFrameResult:
        self.frame_idx += 1
        res = DMSFrameResult()
        h, w = frame_bgr.shape[:2]
        now = time.time()
        
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        # 1. Face landmarks with Blendshapes and 3D Transformation
        face_res = self.face_landmarker.detect(mp_img)
        
        # 2. Pose landmarks every 2 frames
        if self.frame_idx % 2 == 1 or self.cached_pose_landmarks is None:
            pose_res = self.pose_landmarker.detect(mp_img)
            if pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0:
                self.cached_pose_landmarks = pose_res.pose_landmarks[0]
            else:
                self.cached_pose_landmarks = None

        chin_pt = None
        face_size = 80.0

        # ====================================================
        # 1. NEURAL BLENDSHAPE ANALYSIS (Sleep, Gaze, Yawn)
        # ====================================================
        if face_res.face_landmarks and len(face_res.face_landmarks) > 0:
            res.face_detected = True
            raw_face = face_res.face_landmarks[0]
            pts_2d = np.array([(lm.x * w, lm.y * h) for lm in raw_face], dtype=np.float32)
            chin_pt = pts_2d[152]
            nose_pt = pts_2d[1]
            face_size = float(np.linalg.norm(chin_pt - nose_pt) * 2.2)

            # --- EXTRACT NEURAL BLENDSHAPES ---
            bs_dict: Dict[str, float] = {}
            if face_res.face_blendshapes and len(face_res.face_blendshapes) > 0:
                bs_dict = {cat.category_name: cat.score for cat in face_res.face_blendshapes[0]}

            # (A) SLEEP & EYELID CLOSURE
            # Use neural blendshape eyeBlinkLeft and eyeBlinkRight (0.0 to 1.0)
            blink_l = bs_dict.get("eyeBlinkLeft", 0.0)
            blink_r = bs_dict.get("eyeBlinkRight", 0.0)
            res.blink_score = (blink_l + blink_r) / 2.0
            
            # Eyelid closed threshold
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

            # (B) YAWN DETECTION (Neural jawOpen)
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

            # (C) 3D HEAD POSE (from 4x4 rigid transformation matrix)
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

            # (D) PUPIL GAZE TRACKING (Blendshapes)
            res.gaze_down = (bs_dict.get("eyeLookDownLeft", 0.0) + bs_dict.get("eyeLookDownRight", 0.0)) / 2.0
            res.gaze_left = (bs_dict.get("eyeLookOutLeft", 0.0) + bs_dict.get("eyeLookInRight", 0.0)) / 2.0
            res.gaze_right = (bs_dict.get("eyeLookInLeft", 0.0) + bs_dict.get("eyeLookOutRight", 0.0)) / 2.0

            # Classify gaze orientation
            direction, distracted = self._classify_attention(
                res.pitch, res.yaw, res.gaze_down, res.gaze_left, res.gaze_right
            )
            res.head_direction = direction

            if distracted:
                self.distraction_counter += 1
            else:
                self.distraction_counter = max(0, self.distraction_counter - 2)
            res.is_distracted = self.distraction_counter >= self.config.DISTRACTION_CONSEC_FRAMES

            # (E) MULTI-FACTOR DROWSINESS STATE
            if res.eye_closed_frames >= self.config.MICROSLEEP_FRAMES:
                res.drowsiness_state = "MICROSLEEP"
            elif res.perclos >= self.config.PERCLOS_DROWSY_THRESHOLD or res.yawn_count_window >= 2:
                res.drowsiness_state = f"DROWSY ({res.yawn_count_window} YAWNS)" if res.yawn_count_window >= 2 else "DROWSY"
            elif res.perclos >= self.config.PERCLOS_TIRED_THRESHOLD or res.yawn_count_window == 1:
                res.drowsiness_state = "TIRED (1 YAWN)" if res.yawn_count_window == 1 else "MILD FATIGUE"
            else:
                res.drowsiness_state = "ALERT"

            # (F) MULTI-FACTOR SOBRIETY (AND-gated resting tremor + slow blinks)
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
        # 2. SEATBELT DETECTION & DESK SIMULATION OVERRIDE
        # ====================================================
        res.is_simulation_mode = self.seatbelt_sim_override
        if self.seatbelt_sim_override:
            # Desk test override mode enabled by user
            res.seatbelt_fastened = self.seatbelt_sim_state
            res.seatbelt_confidence = 0.95 if self.seatbelt_sim_state else 0.05
        else:
            # Automatic computer vision seatbelt detector
            is_rhd = (self.config.DRIVER_SEAT_SIDE == "right")
            p_start, p_end = None, None
            
            if self.cached_pose_landmarks is not None:
                res.pose_detected = True
                pose_2d = np.array([(lm.x * w, lm.y * h) for lm in self.cached_pose_landmarks], dtype=np.float32)
                l_sh, r_sh = pose_2d[11], pose_2d[12]
                l_hip, r_hip = pose_2d[23], pose_2d[24]
                sh_dist = float(np.linalg.norm(l_sh - r_sh))
                if is_rhd:
                    p_start = r_sh
                    p_end = l_hip if (l_hip[1] > r_sh[1] + 30 and l_hip[1] < h) else np.array([l_sh[0], r_sh[1] + sh_dist * 1.5], dtype=np.float32)
                else:
                    p_start = l_sh
                    p_end = r_hip if (r_hip[1] > l_sh[1] + 30 and r_hip[1] < h) else np.array([r_sh[0], l_sh[1] + sh_dist * 1.5], dtype=np.float32)
            elif chin_pt is not None:
                res.pose_detected = True
                chest_top = chin_pt[1] + face_size * 0.25
                if is_rhd:
                    p_start = np.array([chin_pt[0] - face_size * 1.0, chest_top], dtype=np.float32)
                    p_end = np.array([chin_pt[0] + face_size * 0.9, chest_top + face_size * 1.8], dtype=np.float32)
                else:
                    p_start = np.array([chin_pt[0] + face_size * 1.0, chest_top], dtype=np.float32)
                    p_end = np.array([chin_pt[0] - face_size * 0.9, chest_top + face_size * 1.8], dtype=np.float32)

            if p_start is not None and p_end is not None:
                fastened, conf, line = self.gpu_seatbelt.detect_seatbelt(
                    frame_bgr, p_start, p_end, seat_side=self.config.DRIVER_SEAT_SIDE
                )
                if not fastened:
                    self.seatbelt_unfastened_counter += 1
                else:
                    self.seatbelt_unfastened_counter = max(0, self.seatbelt_unfastened_counter - 2)

                res.seatbelt_fastened = fastened
                res.seatbelt_confidence = conf
                res.seatbelt_line = line
            else:
                res.seatbelt_fastened = True

        # ====================================================
        # 3. ALARM ARBITRATION
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

        if not res.seatbelt_fastened and (self.seatbelt_unfastened_counter >= self.config.SEATBELT_UNFASTENED_DEBOUNCE or self.seatbelt_sim_override):
            alarms.append("SEATBELT")

        res.active_alarms = alarms
        return res

    def _classify_attention(self, pitch: float, yaw: float, down: float, left: float, right: float) -> Tuple[str, bool]:
        """Combines head pose rotation AND neural pupil gaze"""
        if down >= self.config.GAZE_LOOK_DOWN_THRESHOLD or pitch < self.config.HEAD_PITCH_DOWN:
            return "DOWN (PHONE)", True
        elif left >= self.config.GAZE_LOOK_SIDE_THRESHOLD or yaw < self.config.HEAD_YAW_LEFT:
            return "LOOKING LEFT", True
        elif right >= self.config.GAZE_LOOK_SIDE_THRESHOLD or yaw > self.config.HEAD_YAW_RIGHT:
            return "LOOKING RIGHT", True
        elif pitch > self.config.HEAD_PITCH_UP:
            return "LOOKING UP", True
        return "ROAD", False
