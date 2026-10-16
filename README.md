# Automotive ADAS Driver Monitoring System (DMS)

A real-time Computer Vision Driver Monitoring System (DMS) built for Advanced Driver Assistance Systems (ADAS) using a single in-cabin driver-facing camera.

---

## Capabilities & Architecture

| Feature | Primary Metrics & Algorithms | Triggers & Alerts |
| :--- | :--- | :--- |
| **a) Drunk / Impairment** | Gated Resting Postural Ataxia + Saccadic Latency & Sluggish Blink Kinetics (excludes intentional turns) | Multi-cue Impairment Risk Index triggers `WARNING: SUSPECTED INTOXICATION` |
| **b) Drowsy / Fatigue** | Google Neural Blendshapes (`eyeBlinkLeft/Right`), PERCLOS, Neural `jawOpen` Yawn Accumulator (rolling 90s) | `TIRED (1 YAWN)`, `DROWSY`, and `CRITICAL: WAKE UP! MICROSLEEP DETECTED` |
| **c) Unfocused / Distracted** | Neural Pupil Gaze Vector (`eyeLookDown`, `eyeLookLeft/Right`) + 3D Rigid Transformation Matrix $(R|T)$ | Looking down at phone or looking away $> 1.5$s triggers `ATTENTION: KEEP EYES ON ROAD` |
| **d) Seatbelt Fastened** | Directional Hough Line Filtering + Parallel Dual-Edge Webbing Pair Matching + Desk Simulation Override (`T` key) | Belt missing $> 1.0$s triggers `SAFETY ALERT: FASTEN SEATBELT` |

---

## Quick Start

### 1. Run the System
```bash
python main.py
```

### 2. Interactive Runtime Keyboard Shortcuts

| Key | Action | Description |
| :---: | :--- | :--- |
| **`T`** | **Toggle Seatbelt Sim** | Instant toggle between BUCKLED and UNBUCKLED for desk testing without a car seatbelt |
| **`B`** | **Calibrate Seatbelt** | Samples current chest / strap reading as fastened baseline |
| **`[` / `]`** | **Adjust Sensitivity** | Live adjustments to seatbelt contrast sensitivity ($0.4\times$ to $3.0\times$) |
| **`C`** | **Calibrate Posture** | Zero-centers forward road gaze and resting head baseline |
| **`S`** | **Toggle Seat Side** | Switches between Right-Hand Drive (RHD) and Left-Hand Drive (LHD) |
| **`M`** | **Mute / Unmute** | Toggles audible chime alerts |
| **`Q` / `ESC`** | **Quit** | Closes camera feed and cleans up windows |



---

## How It Works in Detail

### 1. Drunk / Impairment Detection
Single-camera impairment monitoring is based on motor-control degradation:
- **Head Sway Instability:** Inebriated individuals cannot hold a steady head position; rolling standard deviation of pitch/yaw/roll reveals continuous involuntary micro-swaying.
- **Sluggish Blink Kinetics:** Alcohol depresses oculomotor reflexes. Crisp blinks (100–250ms) are replaced by prolonged partial closures (400–900ms) without entering full sleep.
- **Micro-Nods:** Abrupt downward drops in pitch followed by compensatory jerks.
- Combined into a real-time **Impairment Risk Index (0–100%)**.

### 2. Drowsiness & Microsleep
- **EAR (Eye Aspect Ratio):**
  $$\text{EAR} = \frac{\|p_2 - p_6\| + \|p_3 - p_5\|}{2 \cdot \|p_1 - p_4\|}$$
  Drops below `0.22` when eyes close. Continuous closure for $> 1.5$ seconds triggers critical microsleep alert.
- **PERCLOS:** Calculates the percentage of eye closure over a rolling 4-second buffer (120 frames).
- **Yawn Detection (MAR):** Measures vertical mouth opening ratio. Two or more yawns within a 10-second window elevates fatigue warnings.

### 3. Distraction / Unfocused Gaze
- Computes perspective projection via `cv2.solvePnP` using 6 canonical 3D facial landmarks: nose tip, chin, eye outer corners, and mouth corners.
- Yields standard Euler angles:
  - **Yaw:** Head turned left/right away from road.
  - **Pitch:** Looking down at smartphone in lap or center screen.
  - **Roll:** Lateral head tilt.
- A 3D directional vector is projected directly from the driver's nose so gaze direction is visible in real-time.

### 4. Seatbelt Detection
- Leverages MediaPipe Pose landmarks to locate the driver's torso (outer shoulder and opposite inner hip).
- Extracts the diagonal region of interest (ROI) across the chest.
- Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and Canny edge filtering.
- Performs Hough Line Transform filtered for the characteristic $25^\circ-70^\circ$ diagonal seatbelt trajectory.
- Supports both Left-Hand Drive (outer left shoulder $\rightarrow$ inner right hip) and Right-Hand Drive (outer right shoulder $\rightarrow$ inner left hip) layouts.
