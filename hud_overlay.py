"""
Clean, Executive-Grade Automotive HUD Overlay (Tesla / Mobileye Style).
Optimized for Raspberry Pi 4B (320x240 / 480x360 / 640x480).
- Focus on the 3 core pillars: Drowsiness, Distraction, Sobriety
- Zero flashing / strobe lights
- Clean, calm translucent status capsules
- Dedicated OEM alert card only when genuine safety events trigger
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

        # 1. Sleek Top Status Bar
        self._draw_top_bar(canvas, res, w, fps)

        # 2. Live Neural Metrics Card
        self._draw_live_metrics(canvas, res, w, h)

        # 3. Dedicated Alert Card (only appears if there is an active alarm)
        self._draw_alert_card(canvas, res.active_alarms, w, h)

        # 4. Minimalist Footer Hotkey Bar
        self._draw_footer(canvas, w, h, sound_enabled)

        return canvas

    def _draw_top_bar(self, canvas: np.ndarray, res: DMSFrameResult, w: int, fps: float):
        bar_h = 38 if w <= 480 else 44
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 20, 24), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (45, 50, 58), 1)

        # Title
        font_scale = 0.45 if w <= 480 else 0.58
        cv2.putText(canvas, "ADAS DMS", (12, bar_h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 215, 255), 1, cv2.LINE_AA)

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

        # Draw Badges side by side
        badge_x = 90 if w <= 480 else 125
        badge_scale = 0.32 if w <= 480 else 0.40
        for text, bg_col, txt_col in badges:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, badge_scale, 1)[0]
            bw = text_size[0] + 12
            bh = 22 if w <= 480 else 24
            by = 8 if w <= 480 else 10

            cv2.rectangle(canvas, (badge_x, by), (badge_x + bw, by + bh), bg_col, -1)
            cv2.rectangle(canvas, (badge_x, by), (badge_x + bw, by + bh), (255, 255, 255), 1)
            cv2.putText(canvas, text, (badge_x + 6, by + bh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, badge_scale, txt_col, 1, cv2.LINE_AA)
            badge_x += bw + 6

        # FPS on far right
        fps_str = f"{fps:2.0f}FPS"
        fps_scale = 0.38 if w <= 480 else 0.46
        cv2.putText(canvas, fps_str, (w - (55 if w <= 480 else 68), bar_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_scale, (155, 160, 170), 1, cv2.LINE_AA)

    def _draw_live_metrics(self, canvas: np.ndarray, res: DMSFrameResult, w: int, h: int):
        card_w = 175 if w <= 480 else 205
        card_h = 36 if w <= 480 else 42
        cx, cy = 8, 44 if w <= 480 else 52
        
        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (50, 60, 75), 1)

        scale = 0.32 if w <= 480 else 0.38
        cv2.putText(canvas, f"Eyes: {res.blink_score*100:2.0f}% | Mouth: {res.jaw_open*100:2.0f}%",
                    (cx + 6, cy + (14 if w <= 480 else 18)), cv2.FONT_HERSHEY_SIMPLEX, scale, (210, 220, 230), 1, cv2.LINE_AA)
        
        gaze_str = f"Down Gaze: {res.gaze_down*100:2.0f}% | Pitch: {res.pitch:+.0f}"
        cv2.putText(canvas, gaze_str, (cx + 6, cy + (28 if w <= 480 else 34)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (170, 180, 195), 1, cv2.LINE_AA)

    def _draw_alert_card(self, canvas: np.ndarray, alarms: list, w: int, h: int):
        if not alarms:
            return

        alarm_map = {
            "MICROSLEEP": ("CRITICAL: WAKE UP - MICROSLEEP DETECTED", (30, 30, 220)),
            "DISTRACTION": ("ATTENTION: KEEP EYES ON THE ROAD", (25, 120, 225)),
            "DROWSY": ("FATIGUE ALERT: TIME FOR A BREAK", (25, 140, 215)),
            "IMPAIRMENT": ("DRIVER IMPAIRMENT WARNING DETECTED", (130, 30, 150))
        }

        top_alarm = alarms[0]
        if top_alarm not in alarm_map:
            return

        text, bg_col = alarm_map[top_alarm]
        card_w = min(420, int(w * 0.90))
        card_h = 32 if w <= 480 else 38
        cx = int((w - card_w) / 2)
        cy = 90 if w <= 480 else 100

        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), bg_col, -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (255, 255, 255), 1)

        scale = 0.40 if w <= 480 else 0.50
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0]
        tx = cx + int((card_w - t_size[0]) / 2)
        ty = cy + (22 if w <= 480 else 25)
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_footer(self, canvas: np.ndarray, w: int, h: int, sound_enabled: bool):
        bar_h = 20 if w <= 480 else 24
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (14, 16, 20), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, h - bar_h), (w, h - bar_h), (35, 40, 48), 1)

        audio_str = f"Audio:{'ON' if sound_enabled else 'MUTE'}"
        help_str = f"[C] Posture   [M] {audio_str}   [Q] Quit"
        scale = 0.32 if w <= 480 else 0.38
        cv2.putText(canvas, help_str, (10, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (150, 155, 165), 1, cv2.LINE_AA)
