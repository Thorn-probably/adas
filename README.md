# Automotive ADAS Driver Monitoring System (DMS)
### Optimized for Raspberry Pi 4B (ARM Cortex-A72) & Desktop

A lightweight, real-time Computer Vision Driver Monitoring System (DMS) engineered to run at **30+ FPS on a Raspberry Pi 4B** using a single driver-facing camera.

---

## Capabilities & Three Core Pillars

| Feature | Biometric Metrics & Multi-Factor Fusion | Detection Output |
| :--- | :--- | :--- |
| **1. Drowsiness & Sleep** | Google Neural Blendshapes (`eyeBlinkLeft/Right`), PERCLOS over 120-frame rolling window, Neural `jawOpen` Yawn Accumulator (rolling 90s), Slow blink kinetics ($>330\text{ ms}$) | `EYES: ALERT` $\rightarrow$ `TIRED (1 YAWN)` $\rightarrow$ `DROWSY` $\rightarrow$ `CRITICAL: WAKE UP! MICROSLEEP DETECTED` |
| **2. Distraction & Gaze** | Neural Pupil Gaze Vector (`eyeLookDown` phone-in-lap, `eyeLookLeft/Right`) + 3D Rigid Transformation Matrix $(R \mid T)$ | Looking away or down at phone $> 1.4\text{s}$ triggers `ATTENTION: KEEP EYES ON ROAD` |
| **3. Sobriety / Impairment** | Multi-Factor AND-Gate: Gated Resting Postural Tremor (excludes intentional turns) + Co-occurring Sluggish Blink Kinetics | Sustained resting tremor + delayed ocular kinetics triggers `WARNING: SUSPECTED INTOXICATION` |

*(Note: Seatbelt detection has been completely removed to eliminate false triggers in non-standard setups and eliminate pose estimation CPU overhead).*

---

## Raspberry Pi 4B Optimization Highlights

1. **Eliminated Heavy Pose Overhead:** Dropping seatbelt detection and body pose estimation cut CPU usage by $> 40\%$.
2. **ARM-Optimized Resolution (320x240 / 480x360):** Running at 320x240 guarantees 30+ FPS on Raspberry Pi 4B's quad-core Cortex-A72 with $< 25\%$ CPU per core, preventing thermal throttling.
3. **Dual-Engine Architecture:**
   - **Engine A (Default):** MediaPipe Face Mesh running TFLite with ARM NEON / XNNPACK acceleration.
   - **Engine B (Custom Trained):** Custom-trained `TinyDMSNet` (<35k parameters, ~140 KB), achieving **2.5 ms** CPU latency.
4. **Hardware GPIO Buzzer:** Direct hardware pin trigger on Raspberry Pi GPIO Pin 18 for in-vehicle 5V active buzzers (`--gpio-buzzer`).
5. **Native Pi Camera Support:** Supports standard USB V4L2 cameras and native Raspberry Pi Camera Module v2/v3 via `picamera2`.
6. **Headless Embedded Mode:** Run with `--headless` in real car setups with no HDMI display, streaming alerts to console or buzzer.

---

## Training Your Own Custom Model (`TinyDMSNet`)

A self-contained training and export script is included to train a lightweight Depthwise Separable CNN designed for ARM processors:

```bash
# Train on synthetic/augmented driver dataset and export to models/
python train_dms_model.py --epochs 10

# Or train on a custom driver image dataset folder:
# (Expects subfolders: alert/, drowsy/, distracted/, impaired/)
python train_dms_model.py --data-dir path/to/dataset --epochs 15
```

This exports:
- `models/tiny_dms_rpi.pt` (PyTorch weights, ~148 KB)
- `models/tiny_dms_rpi.torchscript` (TorchScript traced model, ~201 KB, zero-dependency deployment on Pi)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run with GUI Display
```bash
python main.py
```

### 3. Raspberry Pi 4B Commands
```bash
# Recommended for Raspberry Pi 4B (320x240, 30+ FPS)
python main.py --width 320 --height 240

# Using native Raspberry Pi Camera Module (CSI ribbon cable)
python main.py --camera-type picamera2 --width 320 --height 240

# Headless in-vehicle embedded deployment with hardware GPIO 18 buzzer
python main.py --headless --gpio-buzzer --width 320 --height 240

# Run with custom-trained TinyDMSNet model
python main.py --model custom --width 320 --height 240
```

---

## Interactive Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`C`** | **Calibrate Posture** | Zero-centers forward road gaze and resting head baseline |
| **`M`** | **Mute / Unmute** | Toggles audible chime alerts |
| **`Q` / `ESC`** | **Quit** | Closes camera feed and cleanly exits |
