"""
Clean Automotive HUD Overlay (Focused Exclusively on Eyes & Drowsiness).
Optimized for Raspberry Pi 4B & Desktop:
- Live Eye Openness & Closure Tracking
- Drowsiness & Yawn State Capsule
- Clear Microsleep Emergency Card
- Zero clutter, zero flashing lights
"""

import cv2
import numpy as np
from dms_engine import DMSFrameResult
from config import DMSConfig


class DMSHUDOverlay:
    def __init__(self, config: DMSConfig):
        self.config = config

    def render(self, frame: np.ndarray, res: DMSFrameResult, fps: float, sound_enabled: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # 1. Top Status Capsule Bar
        self._draw_top_bar(canvas, res, w, fps)

        # 2. Live Eye & Drowsiness Telemetry Card
        self._draw_metrics_card(canvas, res, w, h)

        # 3. Emergency Alert Card (only appears when drowsy or asleep)
        self._draw_alert_card(canvas, res.active_alarms, w, h)

        # 4. Minimalist Footer Bar
        self._draw_footer(canvas, w, h, sound_enabled)

        return canvas

    def _draw_top_bar(self, canvas: np.ndarray, res: DMSFrameResult, w: int, fps: float):
        bar_h = 38 if w <= 480 else 44
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 20, 24), -1)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.line(canvas, (0, bar_h), (w, bar_h), (45, 50, 58), 1)

        # System Title
        font_scale = 0.44 if w <= 480 else 0.56
        cv2.putText(canvas, "EYE DMS", (12, bar_h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 215, 255), 1, cv2.LINE_AA)

        badges = []

        # Badge 1: Eye State (OPEN vs CLOSED or UNRESPONSIVE)
        if not res.face_detected:
            if res.drowsiness_state == "UNRESPONSIVE":
                badges.append(("DRIVER: UNRESPONSIVE", (30, 30, 220), (255, 255, 255)))
            else:
                badges.append(("FACE: SEARCHING", (70, 75, 85), (200, 205, 215)))
        elif res.is_eye_closed:
            badges.append((f"EYES: CLOSED ({res.eye_closed_duration:0.1f}s)", (35, 35, 220), (255, 255, 255)))
        else:
            badges.append(("EYES: OPEN", (35, 155, 60), (255, 255, 255)))

        # Badge 2: Drowsiness State
        if res.drowsiness_state == "MICROSLEEP":
            badges.append(("MICROSLEEP", (30, 30, 220), (255, 255, 255)))
        elif res.drowsiness_state == "UNRESPONSIVE":
            badges.append(("NO FACE DETECTED", (30, 30, 220), (255, 255, 255)))
        elif "DROWSY" in res.drowsiness_state:
            badges.append((res.drowsiness_state, (30, 130, 225), (255, 255, 255)))
        elif "TIRED" in res.drowsiness_state or "FATIGUE" in res.drowsiness_state:
            badges.append((res.drowsiness_state, (25, 110, 205), (255, 255, 255)))
        elif res.drowsiness_state == "SEARCHING":
            badges.append(("SEARCHING", (70, 75, 85), (200, 205, 215)))
        else:
            badges.append(("ALERT", (35, 155, 60), (255, 255, 255)))

        # Render Badges
        badge_x = 95 if w <= 480 else 125
        badge_scale = 0.34 if w <= 480 else 0.40
        for text, bg_col, txt_col in badges:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, badge_scale, 1)[0]
            bw = text_size[0] + 14
            bh = 22 if w <= 480 else 24
            by = 8 if w <= 480 else 10

            cv2.rectangle(canvas, (badge_x, by), (badge_x + bw, by + bh), bg_col, -1)
            cv2.rectangle(canvas, (badge_x, by), (badge_x + bw, by + bh), (255, 255, 255), 1)
            cv2.putText(canvas, text, (badge_x + 7, by + bh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, badge_scale, txt_col, 1, cv2.LINE_AA)
            badge_x += bw + 8

        # FPS
        fps_scale = 0.36 if w <= 480 else 0.44
        cv2.putText(canvas, f"{fps:2.0f}FPS", (w - (55 if w <= 480 else 65), bar_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, fps_scale, (155, 160, 170), 1, cv2.LINE_AA)

    def _draw_metrics_card(self, canvas: np.ndarray, res: DMSFrameResult, w: int, h: int):
        card_w = 185 if w <= 480 else 225
        card_h = 44 if w <= 480 else 50
        cx, cy = 8, 44 if w <= 480 else 52

        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (50, 60, 75), 1)

        scale = 0.32 if w <= 480 else 0.38
        y1 = cy + (14 if w <= 480 else 18)
        y2 = cy + (28 if w <= 480 else 34)

        if not res.face_detected:
            lost_col = (40, 40, 240) if res.face_lost_duration >= 2.0 else (50, 180, 240)
            cv2.putText(canvas, f"Face Lost: {res.face_lost_duration:0.1f}s",
                        (cx + 6, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, lost_col, 1, cv2.LINE_AA)
            cv2.putText(canvas, "Status: Monitoring Road",
                        (cx + 6, y2), cv2.FONT_HERSHEY_SIMPLEX, scale, (160, 170, 185), 1, cv2.LINE_AA)
        else:
            # Line 1: Eye Openness % and Closed status
            openness_col = (100, 240, 120) if res.eye_openness > 0.50 else (60, 80, 240)
            eye_text = f"Eye Openness: {res.eye_openness*100:3.0f}%"
            if res.is_eye_closed:
                eye_text += f" ({res.eye_closed_duration:0.1f}s)"
            cv2.putText(canvas, eye_text,
                        (cx + 6, y1), cv2.FONT_HERSHEY_SIMPLEX, scale, openness_col, 1, cv2.LINE_AA)

            # Line 2: PERCLOS & Yawn metrics
            yawn_str = f"PERCLOS: {res.perclos*100:2.0f}% | Yawns: {res.yawn_count_window}"
            cv2.putText(canvas, yawn_str,
                        (cx + 6, y2), cv2.FONT_HERSHEY_SIMPLEX, scale, (180, 190, 205), 1, cv2.LINE_AA)

    def _draw_alert_card(self, canvas: np.ndarray, alarms: list, w: int, h: int):
        if not alarms:
            return

        alarm_map = {
            "MICROSLEEP": ("CRITICAL: WAKE UP - MICROSLEEP DETECTED", (30, 30, 220)),
            "UNRESPONSIVE": ("WARNING: DRIVER UNRESPONSIVE / FACE LOST", (30, 30, 220)),
            "DISTRACTION": ("WARNING: DRIVER DISTRACTED", (30, 30, 220)),
            "DROWSY": ("FATIGUE ALERT: TIME FOR A BREAK", (25, 140, 215))
        }

        top_alarm = alarms[0]
        if top_alarm not in alarm_map:
            return

        text, bg_col = alarm_map[top_alarm]
        card_w = min(440, int(w * 0.92))
        card_h = 32 if w <= 480 else 38
        cx = int((w - card_w) / 2)
        cy = 95 if w <= 480 else 105

        overlay = canvas.copy()
        cv2.rectangle(overlay, (cx, cy), (cx + card_w, cy + card_h), bg_col, -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), (255, 255, 255), 1)

        scale = 0.38 if w <= 480 else 0.48
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
        help_str = f"[M] {audio_str}   [Q] Quit"
        scale = 0.32 if w <= 480 else 0.38
        cv2.putText(canvas, help_str, (10, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (150, 155, 165), 1, cv2.LINE_AA)
