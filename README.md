# Automotive ADAS Driver Monitoring System (DMS)
### Specialized Eye Tracking & Drowsiness / Sleep Detection
**Engineered for Raspberry Pi 4B (ARM Cortex-A72) & Desktop**

A lightweight, real-time Computer Vision Driver Monitoring System (DMS) built to run at **35+ FPS on a Raspberry Pi 4B** using a single driver-facing camera.

---

## Core Focus: Eye State & Drowsiness

The system focuses exclusively on reliable, perspective-invariant ocular tracking and physiological drowsiness detection:

| Metric | Mechanism | Automotive Action |
| :--- | :--- | :--- |
| **Eye Openness Tracking** | Google MediaPipe Neural Blendshapes (`eyeBlinkLeft`, `eyeBlinkRight`) | Live continuous measurement from `0.0` (closed) to `1.0` (wide open). Invariant to head tilts and angles. |
| **Eye State Classification** | Thresholded eyelid closure ($>0.50$) with elapsed duration | `EYES: OPEN` vs `EYES: CLOSED (Xs)` status badge. |
| **Fatigue / PERCLOS** | Percentage of Eye Closure over rolling 4.0-second time window ($35\%$ threshold) | Framerate-agnostic detection of heavy eyelids and prolonged blink duration. |
| **Yawn Accumulator** | Neural `jawOpen` ($>0.55$) sustained $\ge 0.7\text{s}$ over rolling 90s | Distinguishes yawning from talking/singing. Warns on cumulative yawns. |
| **Microsleep Alarm** | Continuous eye closure for $\ge 1.2\text{s}$ | **CRITICAL: WAKE UP! MICROSLEEP DETECTED** audio/buzzer alarm. |
| **Driver Unresponsive / Slump** | Loss of face detection for $\ge 2.0\text{s}$ | **WARNING: DRIVER UNRESPONSIVE / FACE LOST** emergency alert. |

---

## Why MediaPipe Neural Blendshapes?

Traditional Eye Aspect Ratio (EAR) using Euclidean pixel distances fails whenever the driver tilts their head or looks at an angle because 2D perspective foreshortening artificially squashes eye height, triggering constant false alarms.

This engine utilizes MediaPipe's **3D Face Blendshapes**, which extract normalized anatomical muscle activations directly from the 3D canonical mesh. Eyelid closure is measured accurately whether looking straight, turning, or tilting.

---

## Raspberry Pi 4B Optimization Highlights

1. **Lightweight & Efficient:** Pure ocular focus cuts out heavy pose estimation and gaze matrices, running at **<20% CPU** on the Pi 4B.
2. **ARM-Optimized Resolution (320x240 / 480x360):** Guarantees 35+ FPS with TFLite + XNNPACK NEON SIMD acceleration without thermal throttling.
3. **Hardware GPIO Buzzer:** Supports direct trigger of a 5V active buzzer on Raspberry Pi **GPIO Pin 18** (`--gpio-buzzer`).
4. **Native Pi Camera Support:** Works with standard USB V4L2 webcams or Raspberry Pi Camera Module v2/v3 via `picamera2`.
5. **Headless Vehicle Mode:** Run with `--headless` for embedded automotive setups without an HDMI display.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with Live Video HUD
```bash
python main.py
```

### 3. Raspberry Pi 4B Commands
```bash
# Recommended for Raspberry Pi 4B (320x240, 35+ FPS)
python main.py --width 320 --height 240

# Using native Raspberry Pi Camera Module (CSI ribbon cable)
python main.py --camera-type picamera2 --width 320 --height 240

# Headless in-vehicle deployment with hardware GPIO 18 buzzer
python main.py --headless --gpio-buzzer --width 320 --height 240
```

---

## Interactive Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`M`** | **Mute / Unmute** | Toggles audible warning beeps |
| **`Q` / `ESC`** | **Quit** | Closes camera feed and cleanly exits |
