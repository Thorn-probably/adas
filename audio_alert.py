"""
Threaded, non-blocking audio alarm system for ADAS alerts
"""

import time
import threading
import queue

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class AudioAlertManager:
    def __init__(self, enabled=True, min_interval=0.8):
        self.enabled = enabled
        self.min_interval = min_interval
        self._last_play_time = 0.0
        self._queue = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def trigger(self, alarm_type: str):
        if not self.enabled or not HAS_WINSOUND:
            return
        
        now = time.time()
        if now - self._last_play_time < self.min_interval:
            return
            
        # Put into queue without blocking
        try:
            self._queue.put_nowait(alarm_type)
            self._last_play_time = now
        except queue.Full:
            pass

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                alarm_type = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self.enabled or not HAS_WINSOUND:
                continue

            try:
                if alarm_type == "MICROSLEEP":
                    # Urgent rapid beeps
                    for _ in range(3):
                        winsound.Beep(2200, 100)
                        time.sleep(0.05)
                elif alarm_type == "DISTRACTION":
                    # Double alert beep
                    winsound.Beep(1200, 140)
                    time.sleep(0.08)
                    winsound.Beep(1200, 140)
                elif alarm_type == "DROWSY":
                    # Warning chime
                    winsound.Beep(1500, 200)
                elif alarm_type == "IMPAIRMENT":
                    # Low cautionary warble
                    winsound.Beep(750, 180)
                    time.sleep(0.05)
                    winsound.Beep(650, 220)
                elif alarm_type == "SEATBELT":
                    # Seatbelt reminder chime
                    winsound.Beep(880, 220)
                    time.sleep(0.1)
                    winsound.Beep(880, 220)
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
