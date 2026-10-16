"""
Main entry point for ADAS Driver Monitoring System (DMS).
Clean, fast, non-distracting automotive DMS pipeline.
"""

import sys
import time
import argparse
import cv2
import torch

from config import DMSConfig
from download_models import ensure_models_exist
from dms_engine import DMSEngine
from hud_overlay import DMSHUDOverlay
from audio_alert import AudioAlertManager


def parse_args():
    parser = argparse.ArgumentParser(description="ADAS Driver Monitoring System (DMS)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional path to a recorded video file")
    parser.add_argument("--seat-side", type=str, choices=["left", "right"], default="right",
                        help="Driver seating position: 'right' (RHD) or 'left' (LHD)")
    parser.add_argument("--no-audio", action="store_true", help="Disable audible alarms")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Ensure required neural network model files exist
    ensure_models_exist()

    # 2. Configuration
    config = DMSConfig()
    config.CAMERA_INDEX = args.camera
    config.FRAME_WIDTH = args.width
    config.FRAME_HEIGHT = args.height
    config.DRIVER_SEAT_SIDE = args.seat_side
    config.AUDIO_ALERTS_ENABLED = not args.no_audio

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"\n[DMS] Device: {gpu_name}")
    print(f"[DMS] Driver Position: {config.DRIVER_SEAT_SIDE.upper()}-Hand Drive")

    # 3. Initialize Engine, HUD, Audio
    engine = DMSEngine(config)
    hud = DMSHUDOverlay(config)
    audio = AudioAlertManager(enabled=config.AUDIO_ALERTS_ENABLED, min_interval=config.AUDIO_MIN_INTERVAL_SEC)

    # 4. Open Video Source
    if args.video is not None:
        print(f"[DMS] Opening video: {args.video}")
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"[DMS] Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {args.camera}")
        sys.exit(1)

    print("\n" + "="*50)
    print(" ADAS DRIVER MONITORING SYSTEM (DMS) ACTIVE ")
    print("="*50)
    print(" Controls:")
    print("   'C' -> Calibrate posture baseline")
    print("   'S' -> Toggle seat side (RHD / LHD)")
    print("   'B' -> Calibrate seatbelt baseline")
    print("   'M' -> Mute / Unmute audio")
    print("   'Q' or ESC -> Exit")
    print("="*50 + "\n")

    window_name = "ADAS Driver Monitoring System (DMS)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    prev_time = time.time()
    fps = 30.0

    try:
        while True:
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

            # Process frame
            result = engine.process_frame(frame)

            # Trigger audio alarms
            if result.active_alarms:
                audio.trigger(result.active_alarms[0])

            # Render clean OEM HUD
            annotated_frame = hud.render(frame, result, fps, audio.enabled)

            # Display
            cv2.imshow(window_name, annotated_frame)

            # Keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("[DMS] Exiting...")
                break
            elif key == ord('c') or key == ord('C'):
                engine.calibrate_baseline()
                print("[DMS] Posture Baseline Calibrated.")
            elif key == ord('s') or key == ord('S'):
                new_side = engine.toggle_seat_side()
                print(f"[DMS] Switched seat side to: {new_side.upper()}-Hand Drive")
            elif key == ord('b') or key == ord('B'):
                engine.calibrate_seatbelt()
                print("[DMS] Seatbelt Baseline Calibrated.")
            elif key == ord('t') or key == ord('T'):
                sim_state = engine.toggle_seatbelt_simulation()
                print(f"[DMS] Seatbelt Desk Test Mode: {'BUCKLED' if sim_state else 'UNBUCKLED'}")
            elif key == ord('['):
                sens = engine.adjust_seatbelt_sensitivity(-0.15)
                print(f"[DMS] Seatbelt Sensitivity: {sens:.2f}x")
            elif key == ord(']'):
                sens = engine.adjust_seatbelt_sensitivity(+0.15)
                print(f"[DMS] Seatbelt Sensitivity: {sens:.2f}x")
            elif key == ord('m') or key == ord('M'):
                audio.set_enabled(not audio.enabled)
                print(f"[DMS] Audio: {'ENABLED' if audio.enabled else 'MUTED'}")



    finally:
        audio.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("[DMS] Shutdown complete.")


if __name__ == "__main__":
    main()
