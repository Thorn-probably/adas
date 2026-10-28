"""
Threaded, non-blocking audio alarm system for ADAS alerts
Supports:
1. Windows (winsound.Beep)
2. Linux / Raspberry Pi (terminal acoustic bell or aplay)
3. Hardware GPIO Buzzer on Raspberry Pi 4B (GPIO 18)
"""

import os
import sys
import time
import threading
import queue

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Try importing Raspberry Pi GPIO buzzer
HAS_GPIO = False
gpio_buzzer = None
try:
    from gpiozero import Buzzer
    HAS_GPIO = True
except ImportError:
    try:
        import RPi.GPIO as GPIO
        HAS_GPIO = True
    except ImportError:
        HAS_GPIO = False


class AudioAlertManager:
    def __init__(self, enabled=True, min_interval=1.0, gpio_pin=18, enable_gpio=False):
        self.enabled = enabled
        self.min_interval = min_interval
        self._last_play_time = 0.0
        self._queue = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        
        self.enable_gpio = enable_gpio and HAS_GPIO
        self.gpio_buzzer = None
        if self.enable_gpio:
            try:
                from gpiozero import Buzzer
                self.gpio_buzzer = Buzzer(gpio_pin)
                print(f"[Audio] Raspberry Pi Hardware Buzzer enabled on GPIO {gpio_pin}")
            except Exception as e:
                print(f"[Audio] Could not initialize GPIO buzzer: {e}")

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool):
        self.enabled = enabled

    def trigger(self, alarm_type: str):
        if not self.enabled:
            return
        
        now = time.time()
        if now - self._last_play_time < self.min_interval:
            return
            
        try:
            self._queue.put_nowait(alarm_type)
            self._last_play_time = now
        except queue.Full:
            pass

    def _beep(self, freq: int, duration_ms: int):
        if HAS_WINSOUND:
            try:
                winsound.Beep(freq, duration_ms)
            except Exception:
                pass
        elif self.gpio_buzzer is not None:
            try:
                self.gpio_buzzer.on()
                time.sleep(duration_ms / 1000.0)
                self.gpio_buzzer.off()
            except Exception:
                pass
        else:
            # Fallback for Linux without soundcard: terminal audible bell
            sys.stdout.write('\a')
            sys.stdout.flush()

    def _worker(self):
        while not self._stop_event.is_set():
            try:
                alarm_type = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self.enabled:
                continue

            try:
                if alarm_type == "MICROSLEEP":
                    for _ in range(3):
                        self._beep(2200, 100)
                        time.sleep(0.05)
                elif alarm_type == "DISTRACTION":
                    self._beep(1200, 140)
                    time.sleep(0.08)
                    self._beep(1200, 140)
                elif alarm_type == "DROWSY":
                    self._beep(1500, 200)
                elif alarm_type == "IMPAIRMENT":
                    self._beep(750, 180)
                    time.sleep(0.05)
                    self._beep(650, 200)
            except Exception:
                pass

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self.gpio_buzzer is not None:
            try:
                self.gpio_buzzer.close()
            except Exception:
                pass
