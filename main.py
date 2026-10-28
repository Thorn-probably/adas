"""
Main entry point for ADAS Driver Monitoring System (DMS).
Optimized for Raspberry Pi 4B (ARM Cortex-A72) and Desktop.

Specialized exclusively for:
1. Eye Detection & Openness Tracking (Eyelid closure, Eye Openness %, Blink Tracking)
2. Drowsiness, Fatigue & Microsleep Detection (PERCLOS, Yawn Accumulation, Microsleep Alarm)
"""

import sys
import os
import time
import argparse
import cv2

from config import DMSConfig
from download_models import ensure_models_exist
from dms_engine import DMSEngine
from hud_overlay import DMSHUDOverlay
from audio_alert import AudioAlertManager


def parse_args():
    parser = argparse.ArgumentParser(description="ADAS Driver Monitoring System (DMS) - RPi 4B & Desktop")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--camera-type", type=str, choices=["opencv", "picamera2"], default="opencv",
                        help="Camera backend: 'opencv' (V4L2 USB camera) or 'picamera2' (Raspberry Pi Camera Module)")
    parser.add_argument("--video", type=str, default=None, help="Optional path to a recorded video file")
    parser.add_argument("--width", type=int, default=320, help="Frame width (default: 320 for Raspberry Pi 4B)")
    parser.add_argument("--height", type=int, default=240, help="Frame height (default: 240 for Raspberry Pi 4B)")
    parser.add_argument("--model", type=str, choices=["mediapipe", "custom"], default="mediapipe",
                        help="Model engine: 'mediapipe' (TFLite+XNNPACK) or 'custom' (TinyDMSNet)")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode without GUI window (ideal for embedded Raspberry Pi)")
    parser.add_argument("--gpio-buzzer", action="store_true",
                        help="Enable hardware buzzer on Raspberry Pi GPIO Pin 18")
    parser.add_argument("--no-audio", action="store_true", help="Disable audible alarm beeps")
    return parser.parse_args()


def init_camera(args, config):
    """Initializes OpenCV V4L2 camera or native Raspberry Pi Picamera2"""
    if args.video is not None:
        print(f"[DMS] Opening video: {args.video}")
        return cv2.VideoCapture(args.video), "opencv"

    if args.camera_type == "picamera2":
        try:
            from picamera2 import Picamera2
            picam = Picamera2()
            picam.configure(picam.create_preview_configuration(main={"size": (config.FRAME_WIDTH, config.FRAME_HEIGHT), "format": "RGB888"}))
            picam.start()
            print(f"[DMS] Initialized native Raspberry Pi Picamera2 ({config.FRAME_WIDTH}x{config.FRAME_HEIGHT})")
            return picam, "picamera2"
        except Exception as e:
            print(f"[DMS] Could not initialize Picamera2 ({e}), falling back to OpenCV V4L2...")

    # Standard OpenCV V4L2 USB camera
    print(f"[DMS] Opening camera {args.camera} ({config.FRAME_WIDTH}x{config.FRAME_HEIGHT})...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        sys.exit(1)

    return cap, "opencv"


def main():
    args = parse_args()

    # 1. Ensure model assets exist
    if args.model == "mediapipe":
        ensure_models_exist()

    # 2. Configuration
    config = DMSConfig()
    config.CAMERA_INDEX = args.camera
    config.FRAME_WIDTH = args.width
    config.FRAME_HEIGHT = args.height
    config.CAMERA_TYPE = args.camera_type
    config.ENGINE_TYPE = args.model
    config.HEADLESS = args.headless
    config.ENABLE_GPIO = args.gpio_buzzer
    config.AUDIO_ALERTS_ENABLED = not args.no_audio

    print(f"\n[DMS] Initializing System for Raspberry Pi 4B & Embedded ADAS")
    print(f"[DMS] Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT} | Engine: {config.ENGINE_TYPE.upper()}")
    print(f"[DMS] Mode: {'HEADLESS (No Display)' if config.HEADLESS else 'GUI Window Active'}")

    # 3. Initialize Engine, HUD, Audio
    engine = DMSEngine(config)
    hud = DMSHUDOverlay(config) if not config.HEADLESS else None
    audio = AudioAlertManager(
        enabled=config.AUDIO_ALERTS_ENABLED,
        min_interval=config.AUDIO_MIN_INTERVAL_SEC,
        gpio_pin=config.GPIO_BUZZER_PIN,
        enable_gpio=config.ENABLE_GPIO
    )

    # 4. Open Camera Source
    cap, cam_type = init_camera(args, config)

    print("\n" + "="*55)
    print(" ADAS DRIVER MONITORING SYSTEM (DMS) ACTIVE ")
    print("="*55)
    print(" Active Features:")
    print("   [1] Eye Detection & Openness Tracking (eyeBlinkLeft/Right)")
    print("   [2] Drowsiness & Fatigue (PERCLOS rolling average)")
    print("   [3] Yawn Accumulator (jawOpen sustained tracking)")
    print("   [4] Microsleep Critical Wake-Up Alarm")
    print("\n Controls:")
    print("   'M' -> Mute / Unmute audio alerts")
    print("   'Q' or ESC -> Exit")
    print("="*55 + "\n")

    window_name = f"ADAS DMS ({config.FRAME_WIDTH}x{config.FRAME_HEIGHT})"
    if not config.HEADLESS:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, max(640, config.FRAME_WIDTH * 2), max(480, config.FRAME_HEIGHT * 2))

    prev_time = time.time()
    fps = 30.0

    try:
        while True:
            # Capture frame
            if cam_type == "picamera2":
                frame_rgb = cap.capture_array()
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                ret = True
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if args.video is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        time.sleep(0.01)
                        continue

            # FPS calculation
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Process frame with optimized DMS engine
            result = engine.process_frame(frame)

            # Trigger Audio / GPIO Buzzer
            if result.active_alarms:
                audio.trigger(result.active_alarms[0])

            # Headless telemetry output or GUI render
            if config.HEADLESS:
                if engine.frame_idx % 15 == 0:
                    eye_str = "CLOSED" if result.is_eye_closed else f"OPEN ({result.eye_openness*100:2.0f}%)"
                    status_str = f"[{fps:2.0f} FPS] EYES: {eye_str} | PERCLOS: {result.perclos*100:2.0f}% | STATE: {result.drowsiness_state}"
                    if result.active_alarms:
                        status_str += f" | ALARMS: {result.active_alarms}"
                    print(status_str)
            else:
                annotated_frame = hud.render(frame, result, fps, audio.enabled)
                cv2.imshow(window_name, annotated_frame)

                # Keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:
                    print("[DMS] Exiting...")
                    break
                elif key == ord('m') or key == ord('M'):
                    audio.set_enabled(not audio.enabled)
                    print(f"[DMS] Audio: {'ENABLED' if audio.enabled else 'MUTED'}")

    finally:
        audio.stop()
        if cam_type == "picamera2":
            cap.stop()
        else:
            cap.release()
        if not config.HEADLESS:
            cv2.destroyAllWindows()
        print("[DMS] Clean shutdown complete.")


if __name__ == "__main__":
    main()
