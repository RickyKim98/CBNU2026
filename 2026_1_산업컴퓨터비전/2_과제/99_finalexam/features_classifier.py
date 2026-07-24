"""Feature extraction and classification for mosquito candidates."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config import AnalyzerConfig, CLASS_BROWN, CLASS_FOREST, CLASS_REVIEW, CLASS_UNKNOWN
from vision_algorithms import object_pixel_mask

try:
    from skimage.feature import hog as skimage_hog
except Exception:  # pragma: no cover
    skimage_hog = None


@dataclass
class CandidateResult:
    id: int
    bbox: Tuple[int, int, int, int]
    contour_area: float
    status: str = "OK"
    reason: str = ""
    color_features: Dict[str, float] = field(default_factory=dict)
    shape_features: Dict[str, float] = field(default_factory=dict)
    hog_features: Optional[np.ndarray] = None
    final_class: str = CLASS_UNKNOWN
    confidence: float = 0.0
    length_mm: Optional[float] = None
    crop_path: Optional[Path] = None


def extract_color_features(crop_bgr: np.ndarray) -> Dict[str, float]:
    """Extract HSV/LAB color ratios from object pixels only."""
    mask = object_pixel_mask(crop_bgr)
    if int(np.count_nonzero(mask)) < 10:
        mask = np.ones(crop_bgr.shape[:2], dtype=np.uint8) * 255

    obj = crop_bgr[mask > 0]
    hsv_obj = cv2.cvtColor(obj.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab_obj = cv2.cvtColor(obj.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    mean_bgr = obj.mean(axis=0)
    mean_hsv = hsv_obj.mean(axis=0)
    mean_lab = lab_obj.mean(axis=0)

    h, s, v = hsv_obj[:, 0], hsv_obj[:, 1], hsv_obj[:, 2]
    L, A, B_lab = lab_obj[:, 0], lab_obj[:, 1], lab_obj[:, 2]
    b, g, r = obj[:, 0], obj[:, 1], obj[:, 2]

    # Brown: low/red hue + yellow/brown LAB components + enough saturation.
    brown_hsv = (((h < 34) | (h > 165)) & (s > 24) & (v < 230) & (r >= 0.78 * b))
    brown_lab = ((B_lab > 132) & (A > 118) & (L < 210))
    brown = brown_hsv | brown_lab

    # Black: low lightness/value and all BGR channels relatively low.
    black = ((v < 105) & (L < 125) & (r < 145) & (g < 145) & (b < 165))
    total = max(1, obj.shape[0])

    return {
        "mean_R": float(mean_bgr[2]),
        "mean_G": float(mean_bgr[1]),
        "mean_B": float(mean_bgr[0]),
        "mean_H": float(mean_hsv[0]),
        "mean_S": float(mean_hsv[1]),
        "mean_V": float(mean_hsv[2]),
        "mean_L": float(mean_lab[0]),
        "mean_A": float(mean_lab[1]),
        "mean_B_lab": float(mean_lab[2]),
        "brown_ratio": float(np.count_nonzero(brown) / total),
        "black_ratio": float(np.count_nonzero(black) / total),
        "object_pixel_count": float(total),
    }


def extract_shape_features(mask_crop: np.ndarray) -> Dict[str, float]:
    """Extract contour shape features from a crop mask."""
    mask = (mask_crop > 0).astype(np.uint8) * 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keys = ["area", "perimeter", "width", "height", "major_axis", "minor_axis", "aspect_ratio", "circularity", "orientation", "edge_density"]
    if not contours:
        return {k: 0.0 for k in keys}

    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))
    _x, _y, w, h = cv2.boundingRect(cnt)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    major = float(max(rw, rh, w, h))
    minor_candidates = [v for v in [rw, rh, w, h] if v > 0]
    minor = float(max(1.0, min(minor_candidates or [1.0])))
    aspect = major / max(minor, 1.0)
    circularity = 0.0 if perimeter <= 0 else float(4.0 * math.pi * area / (perimeter * perimeter))
    edge_density = float(np.count_nonzero(mask) / max(1, mask.shape[0] * mask.shape[1]))
    return {
        "area": area,
        "perimeter": perimeter,
        "width": float(w),
        "height": float(h),
        "major_axis": major,
        "minor_axis": minor,
        "aspect_ratio": aspect,
        "circularity": circularity,
        "orientation": float(rect[2]),
        "edge_density": edge_density,
    }


def extract_hog_features(crop_bgr: np.ndarray, resize: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Extract HOG. If scikit-image is unavailable, return simple gradient histogram."""
    crop = cv2.resize(crop_bgr, resize, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if skimage_hog is not None:
        return skimage_hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        ).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist, _ = np.histogram(angle, bins=9, range=(0, 180), weights=mag)
    hist = hist.astype(np.float32)
    return hist / max(float(np.linalg.norm(hist)), 1e-6)


def hog_shape_score(result: CandidateResult) -> float:
    """Use HOG/shape only as a weak validation score, not as final species classifier."""
    s = result.shape_features
    aspect = s.get("aspect_ratio", 0.0)
    circ = s.get("circularity", 0.0)
    edge_density = s.get("edge_density", 0.0)
    # mosquito body/parts tend to be elongated and not circular.
    score = 0.45
    if 1.3 <= aspect <= 8.0:
        score += 0.18
    if circ < 0.70:
        score += 0.12
    if 0.03 <= edge_density <= 0.80:
        score += 0.10
    return float(min(0.90, max(0.25, score)))


def classify_candidate(result: CandidateResult, cfg: AnalyzerConfig) -> None:
    """Classify mosquito candidates while reducing unnecessary Unknown results.

    Policy:
    - Hole/glare/fragment-like objects remain Unknown.
    - A normally detected candidate is classified into the closer class: brown mosquito or Korean forest mosquito.
    - Border/overlapped/low-confidence candidates keep Review status, but still receive the closest species label.
    """
    c = result.color_features
    s = result.shape_features

    brown = c.get("brown_ratio", 0.0)
    black = c.get("black_ratio", 0.0)
    mean_h = c.get("mean_H", 0.0)
    mean_s = c.get("mean_S", 0.0)
    mean_v = c.get("mean_V", 0.0)
    mean_l = c.get("mean_L", 0.0)
    mean_a = c.get("mean_A", 0.0)
    mean_b_lab = c.get("mean_B_lab", 0.0)
    obj_px = c.get("object_pixel_count", 0.0)

    area = s.get("area", 0.0)
    aspect = s.get("aspect_ratio", 0.0)
    circularity = s.get("circularity", 0.0)
    edge_density = s.get("edge_density", 0.0)
    shape_score = hog_shape_score(result)

    # 1) Very small fragments are not reliable for species classification.
    if result.status == "Fragment" or obj_px < 18 or area < 18:
        result.final_class = CLASS_UNKNOWN
        result.confidence = 0.30
        if result.status == "OK":
            result.status = "Review"
        result.reason = result.reason or "fragment or weak object pixels"
        return

    # 2) Reject hole/glare-like circular false positives before species classification.
    # This is intentionally stricter for circularity, because real mosquitoes are elongated.
    if circularity > 0.82 and area > 100 and aspect < 1.45:
        result.final_class = CLASS_UNKNOWN
        result.status = "Review"
        result.reason = "too circular; possible hole/glare"
        result.confidence = 0.32
        return

    # 2-1) Do not classify from mean LAB/HSV alone.
    # Brown-ish hole shadows can have a brown mean color even when there are almost no
    # true brown/black object pixels. In that case, keep it as Review/Unknown instead
    # of counting it as a brown mosquito.
    min_species_ratio = getattr(cfg, "min_species_color_ratio", 0.018)
    if result.status == "Review" and result.contour_area < cfg.fragment_area_px and max(brown, black) < min_species_ratio:
        result.final_class = CLASS_UNKNOWN
        result.status = "Review"
        result.reason = result.reason or "weak species color evidence"
        result.confidence = 0.35
        return

    # 3) Species scoring.
    # Brown mosquito: brown pixel ratio + hue/LAB brownness + not-too-dark brightness.
    brown_score = 0.0
    brown_score += brown * 2.6
    if ((mean_h < 36 or mean_h > 165) and mean_s > 18):
        brown_score += 0.28
    if mean_b_lab > 129 and mean_a > 113:
        brown_score += 0.30
    if mean_v > 78 and mean_l > 88:
        brown_score += 0.12

    # Korean forest mosquito: black pixel ratio + darkness.
    black_score = 0.0
    black_score += black * 2.8
    if mean_l < 150:
        black_score += (150.0 - mean_l) / 150.0 * 0.58
    if mean_v < 135:
        black_score += (135.0 - mean_v) / 135.0 * 0.48
    if mean_s < 105 and mean_v < 145:
        black_score += 0.12

    # Shape/HOG is only a weak validation signal shared by both classes.
    if 1.15 <= aspect <= 9.0:
        brown_score += 0.10
        black_score += 0.10
    if edge_density > 0.02:
        brown_score += shape_score * 0.10
        black_score += shape_score * 0.10

    # 4) Only classify as Unknown when both color signals are extremely weak.
    # Otherwise choose the closer species candidate to reduce excessive Unknown results.
    if brown_score < 0.22 and black_score < 0.22:
        result.final_class = CLASS_UNKNOWN
        result.status = "Review"
        result.reason = "weak color and shape feature"
        result.confidence = 0.35
        return

    if black_score >= brown_score:
        result.final_class = CLASS_FOREST
        diff = black_score - brown_score
        result.confidence = float(min(0.95, 0.52 + diff * 0.75 + black * 0.35))
    else:
        result.final_class = CLASS_BROWN
        diff = brown_score - black_score
        result.confidence = float(min(0.95, 0.52 + diff * 0.75 + brown * 0.35))

    # 5) Low-confidence results are not converted to Unknown; they are marked Review.
    if result.confidence < cfg.low_confidence_threshold:
        result.status = "Review"
        result.reason = result.reason or "low confidence, closest class selected"

    # 6) Border/overlapped candidates are counted by color, but marked for review.
    if result.status in {"Border_Object", "Overlapped"}:
        result.reason = result.reason or "detected object needs review"
