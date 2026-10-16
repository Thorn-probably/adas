"""
Clean, Executive-Grade Automotive HUD Overlay (Tesla / Mobileye Style).
Displays Google Neural Blendshapes:
- Eyelid Closure score (0-100%)
- Pupil Gaze Direction (Road / Down Phone / Left / Right)
- Jaw Opening / Yawn Accumulator
- Sobriety Status
- Seatbelt Status with Live Override [T] key
"""

import cv2
import numpy as np
from typing import Tuple
from dms_engine import DMSFrameResult
from config import DMSConfig


class DMSHUDOverlay:
    def __init__(self, config: DMSConfig):
        self.config = config

    def render(self, frame: np.ndarray, res: DMSFrameResult, fps: float, sound_enabled: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Subtle seatbelt guideline across chest
        if res.seatbelt_line is not None:
            p1, p2 = res.seatbelt_line
            if res.seatbelt_fastened:
                cv2.line(canvas, p1, p2, (45, 200, 85), 3, cv2.LINE_AA)
            else:
                cv2.line(canvas, p1, p2, (40, 70, 230), 3, cv2.LINE_AA)

        # 2. Sleek Top Status Bar
        self._draw_top_bar(canvas, res, w, fps)

        # 3. Live Neural Tracking Metrics (Subtle & clean)
        self._draw_live_metrics(canvas, res, w, h)

        # 4. Dedicated Alert Card (only appears if there is an active alarm)
        self._draw_alert_card(canvas, res.active_alarms, w, h)

        # 5. Minimalist Footer Hotkey Bar
        self._draw_footer(canvas, w, h, sound_enabled, res.is_simulation_mode)

        return canvas

    def _draw_top_bar(self, canvas: np.ndarray, res: DMSFrameResult, w: int, fps: float):
        bar_h = 44
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 20, 24), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (45, 50, 58), 1)

        # Title
        cv2.putText(canvas, "ADAS DMS", (16, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.58, (0, 215, 255), 1, cv2.LINE_AA)

        badges = []

        # 1. Multi-factor Drowsiness Badge
        if res.drowsiness_state == "MICROSLEEP":
            badges.append(("MICROSLEEP", (35, 35, 220), (255, 255, 255)))
        elif "DROWSY" in res.drowsiness_state:
            badges.append((res.drowsiness_state, (30, 130, 225), (255, 255, 255)))
        elif "TIRED" in res.drowsiness_state or "FATIGUE" in res.drowsiness_state:
            badges.append((res.drowsiness_state, (25, 110, 200), (255, 255, 255)))
        else:
            badges.append(("EYES: ALERT", (35, 155, 60), (255, 255, 255)))

        # 2. Attention / Gaze Badge
        if res.is_distracted:
            badges.append((f"GAZE: {res.head_direction}", (30, 130, 225), (255, 255, 255)))
        else:
            badges.append(("GAZE: ROAD", (35, 155, 60), (255, 255, 255)))

        # 3. Sobriety Badge
        if res.sobriety_state == "IMPAIRED":
            badges.append(("SOBRIETY: WARN", (135, 35, 155), (255, 255, 255)))
        else:
            badges.append(("SOBRIETY: SOBER", (35, 155, 60), (255, 255, 255)))

        # 4. Seatbelt Badge
        sim_tag = " (SIM)" if res.is_simulation_mode else ""
        if res.seatbelt_fastened:
            badges.append((f"BELT: BUCKLED{sim_tag}", (35, 155, 60), (255, 255, 255)))
        else:
            badges.append((f"BELT: UNFASTENED{sim_tag}", (35, 35, 215), (255, 255, 255)))

        # Draw Badges side by side
        badge_x = 125
        for text, bg_col, txt_col in badges:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
            bw = text_size[0] + 16
            bh = 24
            by = 10

            cv2.rectangle(canvas, (badge_x, by), (badge_x + bw, by + bh), bg_col, -1)
            cv2.rectangle(canvas, (badge_x, by), (badge_x + bw, by + bh), (255, 255, 255), 1)
            cv2.putText(canvas, text, (badge_x + 8, by + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)
            badge_x += bw + 8

        # FPS on far right
        fps_str = f"{fps:3.0f} FPS"
        cv2.putText(canvas, fps_str, (w - 70, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (155, 160, 170), 1, cv2.LINE_AA)

    def _draw_live_metrics(self, canvas: np.ndarray, res: DMSFrameResult, w: int, h: int):
        # Subtle real-time neural readout card in upper-left under top bar
        card_w = 205
        card_h = 42
        cx, cy = 12, 52
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (50, 60, 75), 1)

        # Line 1: Eye closure & Mouth Opening
        cv2.putText(canvas, f"Eye Closure: {res.blink_score*100:3.0f}% | Mouth: {res.jaw_open*100:3.0f}%",
                    (cx + 8, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 230), 1, cv2.LINE_AA)
        
        # Line 2: Gaze down & Pitch
        gaze_str = f"Down Gaze: {res.gaze_down*100:3.0f}% | Pitch: {res.pitch:+.0f}"
        cv2.putText(canvas, gaze_str, (cx + 8, cy + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 180, 195), 1, cv2.LINE_AA)

    def _draw_alert_card(self, canvas: np.ndarray, alarms: list, w: int, h: int):
        if not alarms:
            return

        alarm_map = {
            "MICROSLEEP": ("CRITICAL: WAKE UP - MICROSLEEP DETECTED", (30, 30, 220)),
            "DISTRACTION": ("ATTENTION: KEEP EYES ON THE ROAD", (25, 120, 225)),
            "DROWSY": ("FATIGUE ALERT: TIME FOR A BREAK", (25, 140, 215)),
            "IMPAIRMENT": ("DRIVER IMPAIRMENT WARNING DETECTED", (130, 30, 150)),
            "SEATBELT": ("SAFETY ALERT: PLEASE FASTEN SEATBELT", (35, 35, 210))
        }

        top_alarm = alarms[0]
        if top_alarm not in alarm_map:
            return

        text, bg_col = alarm_map[top_alarm]
        card_w = min(480, int(w * 0.85))
        card_h = 38
        cx = int((w - card_w) / 2)
        cy = 100

        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), bg_col, -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (255, 255, 255), 1)

        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)[0]
        tx = cx + int((card_w - t_size[0]) / 2)
        ty = cy + 25
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_footer(self, canvas: np.ndarray, w: int, h: int, sound_enabled: bool, is_sim: bool):
        bar_h = 24
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (14, 16, 20), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, h - bar_h), (w, h - bar_h), (35, 40, 48), 1)

        seat_str = f"Seat:{self.config.DRIVER_SEAT_SIDE.upper()}"
        audio_str = f"Audio:{'ON' if sound_enabled else 'MUTE'}"
        sim_str = "Sim:ON" if is_sim else "Sim:OFF"
        help_str = f"[C] Posture   [S] {seat_str}   [T] {sim_str}   [B] Calib   [M] {audio_str}   [Q] Quit"
        cv2.putText(canvas, help_str, (12, h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 155, 165), 1, cv2.LINE_AA)
