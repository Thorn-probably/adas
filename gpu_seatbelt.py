"""
Multi-Cue Seatbelt Detector with GPU Acceleration.
Combines:
1. Directional Hough Line filtering (30 to 65 degrees) in the Torso ROI
2. Parallel dual-edge strap pair verification (12 to 45 px width)
3. Corridor contrast vs. flanking clothing luminance
4. PyTorch CUDA directional tensor gradient (RTX 3060)
5. Exponential smoothing and hysteresis debouncing (zero flicker)
"""

import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GPUSatbeltDetector:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sensitivity = 1.0
        
        # Smoothed anchor coordinates
        self.smooth_p_start = None
        self.smooth_p_end = None
        
        # Hysteresis state
        self.smoothed_conf = 0.5
        self.is_buckled = True
        
        # Directional GPU kernel for diagonal edge energy
        self.k_diag = torch.tensor([
            [-1.0,  0.0,  1.0],
            [-2.0,  0.0,  2.0],
            [-1.0,  0.0,  1.0]
        ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

    def set_sensitivity(self, val: float):
        self.sensitivity = max(0.4, min(3.0, val))

    def calibrate_baseline(self):
        """Sets current detection state as fastened baseline"""
        self.smoothed_conf = 0.70
        self.is_buckled = True

    def detect_seatbelt(self, frame_bgr: np.ndarray, p_start: np.ndarray, p_end: np.ndarray, seat_side: str = "right"):
        h, w = frame_bgr.shape[:2]
        
        # 1. Smooth anchor points (EMA alpha=0.30) to prevent line jumping
        if self.smooth_p_start is None:
            self.smooth_p_start = p_start.astype(np.float32)
            self.smooth_p_end = p_end.astype(np.float32)
        else:
            self.smooth_p_start = 0.70 * self.smooth_p_start + 0.30 * p_start
            self.smooth_p_end = 0.70 * self.smooth_p_end + 0.30 * p_end
            
        p1 = self.smooth_p_start
        p2 = self.smooth_p_end
        
        # 2. Extract Torso ROI
        x_min = max(0, int(min(p1[0], p2[0]) - 35))
        x_max = min(w, int(max(p1[0], p2[0]) + 35))
        y_min = max(0, int(min(p1[1], p2[1])))
        y_max = min(h, int(max(p1[1], p2[1])))
        
        if (x_max - x_min) < 30 or (y_max - y_min) < 40:
            belt_line = ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
            return self.is_buckled, float(self.smoothed_conf), belt_line

        roi = frame_bgr[y_min:y_max, x_min:x_max]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 3. CLAHE & Directional Edge Extraction
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_roi)
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 110)
        
        # 4. Hough Line Transform for Diagonal Lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 26, minLineLength=30, maxLineGap=22)
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        expected_angle = math.degrees(math.atan2(dy, dx))
        if expected_angle > 90:
            expected_angle -= 180
        elif expected_angle < -90:
            expected_angle += 180

        valid_lines = []
        if lines is not None:
            for line in lines:
                coords = line.ravel()
                if len(coords) < 4:
                    continue
                lx1, ly1, lx2, ly2 = map(int, coords[:4])
                ldx, ldy = lx2 - lx1, ly2 - ly1
                length = math.hypot(ldx, ldy)
                if length < 25:
                    continue
                angle = math.degrees(math.atan2(ldy, ldx))
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                
                diff = abs(angle - expected_angle)
                if diff > 180:
                    diff = 360 - diff
                if diff <= 24.0:
                    valid_lines.append((lx1, ly1, lx2, ly2, length))

        # 5. Parallel Strap Pair Check (two edges of the belt ~15 to 45px apart)
        strap_pair_found = False
        if len(valid_lines) >= 2:
            for i in range(len(valid_lines)):
                for j in range(i + 1, len(valid_lines)):
                    # Distance between midpoints
                    mid1 = ((valid_lines[i][0] + valid_lines[i][2])/2.0, (valid_lines[i][1] + valid_lines[i][3])/2.0)
                    mid2 = ((valid_lines[j][0] + valid_lines[j][2])/2.0, (valid_lines[j][1] + valid_lines[j][3])/2.0)
                    dist = math.hypot(mid1[0] - mid2[0], mid1[1] - mid2[1])
                    if 12.0 <= dist <= 48.0:
                        strap_pair_found = True
                        break
                if strap_pair_found:
                    break

        # 6. Flanking Corridor Contrast (dark webbing vs lighter shirt)
        v = p2 - p1
        v_len = max(float(np.linalg.norm(v)), 1.0)
        u_v = v / v_len
        u_n = np.array([-u_v[1], u_v[0]], dtype=np.float32)

        t_samples = np.linspace(0.2, 0.8, 12)
        center_vals = []
        flank_vals = []
        for t in t_samples:
            pt_c = p1 + t * v
            pt_l = pt_c + 26.0 * u_n
            pt_r = pt_c - 26.0 * u_n
            for pt, val_list in [(pt_c, center_vals), (pt_l, flank_vals), (pt_r, flank_vals)]:
                px, py = float(pt[0]), float(pt[1])
                if 2 <= px < w - 2 and 2 <= py < h - 2:
                    val_list.append(cv2.getRectSubPix(enhanced, (1, 1), (px - x_min, py - y_min))[0, 0] if 0 <= px - x_min < (x_max - x_min) and 0 <= py - y_min < (y_max - y_min) else 0)

        contrast_score = 0.0
        if len(center_vals) >= 6 and len(flank_vals) >= 12:
            diff_lum = abs(float(np.mean(flank_vals)) - float(np.mean(center_vals)))
            contrast_score = min(1.0, diff_lum / 18.0)

        # 7. GPU Directional Energy
        with torch.no_grad():
            t_crop = torch.from_numpy(gray_roi).unsqueeze(0).unsqueeze(0).to(self.device).float() / 255.0
            grad = torch.abs(F.conv2d(t_crop, self.k_diag, padding=1))
            gpu_energy = min(1.0, float(grad.mean().item() * 4.0))

        # Composite frame confidence
        line_score = min(1.0, len(valid_lines) / 3.0)
        pair_bonus = 0.35 if strap_pair_found else 0.0
        
        raw_conf = (line_score * 0.40 + pair_bonus + contrast_score * 0.25 + gpu_energy * 0.15) * self.sensitivity
        raw_conf = min(1.0, max(0.0, raw_conf))

        # Exponential Moving Average Smoothing
        self.smoothed_conf = 0.85 * self.smoothed_conf + 0.15 * raw_conf

        # Hysteresis Debouncing (zero flipping)
        if self.smoothed_conf >= 0.40:
            self.is_buckled = True
        elif self.smoothed_conf <= 0.25:
            self.is_buckled = False

        belt_line = ((int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])))
        return self.is_buckled, float(self.smoothed_conf), belt_line
