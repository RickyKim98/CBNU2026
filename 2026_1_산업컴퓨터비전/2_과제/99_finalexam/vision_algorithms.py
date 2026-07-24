"""Computer vision algorithms used by the improved analyzer.

Main improvements compared with the previous simple version:
1) stronger hole masking using Hough circle + circular contour candidates,
2) K-means segmentation using color + position features,
3) connected-component/contour filtering focused on mosquito body regions.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from config import AnalyzerConfig

Circle = Tuple[int, int, int]
Box = List[float]  # x1, y1, x2, y2, score


def resize_for_work(image_bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Return a resized working image and scale factor relative to original."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return image_bgr.copy(), 1.0
    resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def _dedupe_circles(circles: List[Circle], min_center_distance: float = 18.0) -> List[Circle]:
    circles = sorted(circles, key=lambda c: c[2], reverse=True)
    kept: List[Circle] = []
    for x, y, r in circles:
        duplicate = False
        for kx, ky, kr in kept:
            if math.hypot(x - kx, y - ky) < max(min_center_distance, 0.45 * min(r, kr)):
                duplicate = True
                break
        if not duplicate:
            kept.append((x, y, r))
    return kept


def detect_holes(image_bgr: np.ndarray, cfg: AnalyzerConfig) -> List[Circle]:
    """Detect glue-tape holes.

    HoughCircles is good for regular holes, but it may miss partial/blurred holes.
    Therefore, circular contour candidates are added as a fallback.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (9, 9), 1.7)

    min_dist = max(36, int(min(h, w) * 0.045))
    circles_all: List[Circle] = []

    # 1) Hough Circle Transform
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.18,
        minDist=min_dist,
        param1=80,
        param2=13,
        minRadius=cfg.min_hole_radius_px,
        maxRadius=cfg.max_hole_radius_px,
    )
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            if 0 <= x < w and 0 <= y < h:
                circles_all.append((int(x), int(y), int(r)))

    # 2) circular contour fallback: holes are often bright/low-saturation circular components.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _hh, s, v = cv2.split(hsv)
    # bright or low saturation round regions that differ from blue background
    candidate = (((v > 145) & (s < 135)) | (v > 185)).astype(np.uint8) * 255
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < math.pi * cfg.min_hole_radius_px ** 2 * 0.35 or area > math.pi * cfg.max_hole_radius_px ** 2 * 1.8:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri <= 1:
            continue
        circularity = 4.0 * math.pi * area / (peri * peri)
        if circularity < 0.58:
            continue
        (x, y), r = cv2.minEnclosingCircle(cnt)
        if cfg.min_hole_radius_px <= r <= cfg.max_hole_radius_px:
            circles_all.append((int(round(x)), int(round(y)), int(round(r))))

    detected = _dedupe_circles(circles_all, min_center_distance=float(min_dist) * 0.40)
    return infer_missing_grid_holes((h, w), detected, cfg)


def create_hole_mask(shape_hw: Tuple[int, int], holes: Sequence[Circle], cfg: AnalyzerConfig) -> np.ndarray:
    """Create a mask where hole interiors and rims are removed from mosquito analysis."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    for x, y, r in holes:
        cv2.circle(mask, (int(x), int(y)), int(round(r * cfg.hole_mask_expand)), 255, -1)
    # remove thin hole edges robustly
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    return mask



def is_hole_only_candidate(
    candidate_mask: np.ndarray,
    hole_mask: Optional[np.ndarray],
    bbox_xywh: Tuple[int, int, int, int],
    cfg: AnalyzerConfig,
    image_bgr: Optional[np.ndarray] = None,
) -> bool:
    """Return True when a candidate is only a glue-tape hole/shadow.

    Important policy:
    - Do NOT reject every candidate overlapping a hole.
    - Reject only when the candidate pixels almost disappear after removing the hole mask.
    - If mosquito body/legs remain outside the hole area, keep the candidate.
    """
    if hole_mask is None:
        return False

    x, y, w, h = bbox_xywh
    if w <= 0 or h <= 0:
        return True

    cand_roi = candidate_mask[y:y + h, x:x + w]
    hole_roi = hole_mask[y:y + h, x:x + w]
    if cand_roi.size == 0 or hole_roi.size == 0:
        return True

    cand_area = int(cv2.countNonZero(cand_roi))
    if cand_area <= 0:
        return True

    # If a real mosquito lies on top of a hole, dark body/leg pixels are still visible.
    # Keep such candidates; otherwise the inferred grid hole mask can erase real mosquitoes.
    if image_bgr is not None:
        img_roi = image_bgr[y:y + h, x:x + w]
        if img_roi.shape[:2] == cand_roi.shape:
            hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(img_roi, cv2.COLOR_BGR2LAB)
            _hh, _ss, vv = cv2.split(hsv)
            LL, _AA, _BB = cv2.split(lab)
            bb, gg, rr = cv2.split(img_roi)
            dark = ((vv < 115) & (LL < 140) & (rr < 160) & (gg < 160) & (bb < 185)).astype(np.uint8) * 255
            dark_on_candidate = cv2.bitwise_and(cand_roi, dark)
            dark_area = int(cv2.countNonZero(dark_on_candidate))
            dark_ratio = dark_area / float(max(cand_area, 1))
            keep_dark_area = getattr(cfg, "hole_only_dark_keep_area", 70)
            keep_dark_ratio = getattr(cfg, "hole_only_dark_keep_ratio", 0.07)
            if dark_area >= keep_dark_area or (dark_area >= 25 and dark_ratio >= keep_dark_ratio):
                return False

    # Candidate pixels that remain after excluding hole region.
    non_hole_roi = cv2.bitwise_and(cand_roi, cv2.bitwise_not(hole_roi))
    remaining_area = int(cv2.countNonZero(non_hole_roi))
    remaining_ratio = remaining_area / float(max(cand_area, 1))

    hole_overlap_area = cand_area - remaining_area
    hole_overlap_ratio = hole_overlap_area / float(max(cand_area, 1))

    min_area = getattr(cfg, "hole_only_min_remaining_area", 28)
    min_ratio = getattr(cfg, "hole_only_min_remaining_ratio", 0.10)
    strong_area = getattr(cfg, "hole_only_strong_remaining_area", 65)
    overlap_ratio = getattr(cfg, "hole_only_overlap_ratio", 0.55)

    # Case 1: after removing the hole, almost nothing remains.
    if remaining_area < min_area:
        return True

    # Case 2: most of the candidate is hole/shadow and outside-hole evidence is weak.
    if hole_overlap_ratio > overlap_ratio and remaining_ratio < min_ratio and remaining_area < strong_area:
        return True

    return False



def _estimate_grid_pitch(aligned_dists: List[float]) -> Optional[float]:
    """Estimate repeated hole pitch from aligned pair distances."""
    if len(aligned_dists) < 4:
        return None
    vals = np.asarray(aligned_dists, dtype=np.float32)
    vals = vals[np.isfinite(vals)]
    if vals.size < 4:
        return None
    # 4 px bins are enough for the resized working image and are robust to perspective/noise.
    bin_w = 4.0
    start = max(1.0, float(vals.min()) - bin_w)
    stop = float(vals.max()) + bin_w * 2.0
    bins = np.arange(start, stop, bin_w, dtype=np.float32)
    if bins.size < 3:
        return float(np.median(vals))
    hist, edges = np.histogram(vals, bins=bins)
    idx = int(np.argmax(hist))
    selected = vals[(vals >= edges[idx]) & (vals < edges[idx + 1])]
    if selected.size == 0:
        return float(np.median(vals))
    return float(np.median(selected))


def _estimate_grid_offset(values: np.ndarray, pitch: float, tol: float) -> float:
    """Find grid offset that aligns the largest number of detected hole centers."""
    best_score = -1.0
    best_offset = 0.0
    # Pixel-level search is acceptable because pitch is around 80~140 px in working image.
    for offset in np.arange(0.0, pitch, 1.0, dtype=np.float32):
        dist = np.abs(((values - offset + pitch / 2.0) % pitch) - pitch / 2.0)
        score = float(np.sum(np.exp(-(dist * dist) / (2.0 * tol * tol))))
        if score > best_score:
            best_score = score
            best_offset = float(offset)
    return best_offset


def _grid_lines(offset: float, pitch: float, limit: int, min_seen: float, max_seen: float) -> List[float]:
    """Generate grid lines inside the actually observed tape-hole range."""
    lines: List[float] = []
    margin = max(10.0, pitch * 0.35)
    n0 = int(math.floor((0.0 - offset) / pitch)) - 1
    n1 = int(math.ceil((float(limit) - offset) / pitch)) + 1
    for n in range(n0, n1 + 1):
        v = offset + n * pitch
        if 0.0 <= v < float(limit) and (min_seen - margin) <= v <= (max_seen + margin):
            lines.append(float(v))
    return lines


def infer_missing_grid_holes(image_shape: Tuple[int, int], holes: Sequence[Circle], cfg: AnalyzerConfig) -> List[Circle]:
    """Infer regular glue-tape holes that were missed because their shadows look brown.

    The glue tape has a regular grid. Some holes are not detected by Hough/contour rules
    when the hole interior is slightly brown/pink. Those missed holes later become false
    brown-mosquito candidates. This function infers the missing grid intersections and
    adds them to the hole mask so that a visible hole can be excluded before species
    classification.
    """
    if len(holes) < 8:
        return list(holes)

    img_h, img_w = image_shape
    pts = np.asarray([(x, y, r) for x, y, r in holes], dtype=np.float32)
    radii = pts[:, 2]
    r_med = float(np.median(radii))
    if not np.isfinite(r_med) or r_med <= 0:
        return list(holes)

    # Use only reasonably sized circles for grid estimation. False detections around real
    # mosquitoes often have very small or very large radii.
    good = pts[(radii >= r_med * 0.55) & (radii <= r_med * 1.55)]
    if good.shape[0] < 8:
        good = pts

    horizontal_dists: List[float] = []
    vertical_dists: List[float] = []
    align_tol = max(10.0, r_med * 0.85)
    min_pitch = max(45.0, r_med * 1.8)
    max_pitch = max(min_pitch + 20.0, r_med * 5.5)

    for i in range(good.shape[0]):
        for j in range(i + 1, good.shape[0]):
            dx = abs(float(good[j, 0] - good[i, 0]))
            dy = abs(float(good[j, 1] - good[i, 1]))
            if dy < align_tol and min_pitch <= dx <= max_pitch:
                horizontal_dists.append(dx)
            if dx < align_tol and min_pitch <= dy <= max_pitch:
                vertical_dists.append(dy)

    pitch_x = _estimate_grid_pitch(horizontal_dists)
    pitch_y = _estimate_grid_pitch(vertical_dists)
    if pitch_x is None or pitch_y is None:
        return list(holes)

    # Abnormal pitch means the grid estimation is unreliable. Skip rather than over-mask.
    if not (45.0 <= pitch_x <= min(img_w, img_h) * 0.28 and 45.0 <= pitch_y <= min(img_w, img_h) * 0.28):
        return list(holes)

    tol_x = max(7.0, pitch_x * 0.09)
    tol_y = max(7.0, pitch_y * 0.09)
    offset_x = _estimate_grid_offset(good[:, 0], pitch_x, tol_x)
    offset_y = _estimate_grid_offset(good[:, 1], pitch_y, tol_y)

    xs = _grid_lines(offset_x, pitch_x, img_w, float(np.min(good[:, 0])), float(np.max(good[:, 0])))
    ys = _grid_lines(offset_y, pitch_y, img_h, float(np.min(good[:, 1])), float(np.max(good[:, 1])))
    if len(xs) < 3 or len(ys) < 3:
        return list(holes)

    merged: List[Circle] = list(holes)
    add_radius = int(round(np.clip(r_med, cfg.min_hole_radius_px, cfg.max_hole_radius_px)))

    for yy in ys:
        for xx in xs:
            # Add only if a detected hole is not already close to this grid intersection.
            duplicate = False
            for hx, hy, hr in merged:
                if math.hypot(float(hx) - xx, float(hy) - yy) < max(18.0, min(pitch_x, pitch_y) * 0.34):
                    duplicate = True
                    break
            if not duplicate:
                merged.append((int(round(xx)), int(round(yy)), add_radius))

    return _dedupe_circles(merged, min_center_distance=max(18.0, min(pitch_x, pitch_y) * 0.30))


def estimate_scale_from_holes(
    holes: Sequence[Circle],
    hole_pitch_mm: Optional[float] = None,
    hole_diameter_mm: Optional[float] = None,
) -> Tuple[Optional[float], str]:
    """Calculate pixel_per_mm using actual reference hole pitch or hole diameter."""
    if not holes:
        return None, "scale unavailable: no reference holes detected"

    if hole_pitch_mm and hole_pitch_mm > 0 and len(holes) >= 2:
        pts = np.array([(x, y) for x, y, _ in holes], dtype=np.float32)
        nearest: List[float] = []
        for p in pts:
            d = np.sqrt(((pts - p) ** 2).sum(axis=1))
            d = d[d > 10]
            if len(d):
                nearest.append(float(np.min(d)))
        if nearest:
            pitch_px = float(np.median(nearest))
            return pitch_px / hole_pitch_mm, f"actual reference hole pitch {hole_pitch_mm:.2f} mm, measured pitch {pitch_px:.1f} px"

    if hole_diameter_mm and hole_diameter_mm > 0:
        diam_px = float(np.median([2 * r for _, _, r in holes]))
        return diam_px / hole_diameter_mm, f"actual reference hole diameter {hole_diameter_mm:.2f} mm, measured median diameter {diam_px:.1f} px"

    return None, "scale not requested"


def _feature_image_for_kmeans(image_bgr: np.ndarray, xy_weight: float) -> np.ndarray:
    """Build color+position feature image for K-means segmentation."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = image_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx / max(w - 1, 1)) * 255.0 * xy_weight
    yy = (yy / max(h - 1, 1)) * 255.0 * xy_weight
    feat = np.dstack([lab, xx, yy]).astype(np.float32)
    return feat


def _kmeans_assign_full(features_full: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign every full-resolution pixel to the nearest K-means center."""
    h, w, c = features_full.shape
    flat = features_full.reshape(-1, c)
    # distance to centers, chunked for memory safety
    labels = np.empty((flat.shape[0],), dtype=np.int32)
    chunk = 200000
    for start in range(0, flat.shape[0], chunk):
        part = flat[start:start + chunk]
        d = ((part[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels[start:start + chunk] = np.argmin(d, axis=1)
    return labels.reshape(h, w)


def kmeans_mosquito_mask(image_bgr: np.ndarray, hole_mask: np.ndarray, cfg: AnalyzerConfig) -> np.ndarray:
    """K-means segmentation that separates blue tape background from mosquito-like clusters."""
    h, w = image_bgr.shape[:2]
    # train K-means on a smaller image for speed, then apply centers to the full image
    scale = min(1.0, cfg.kmeans_sample_side / max(h, w))
    small = image_bgr if scale >= 1.0 else cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small_features = _feature_image_for_kmeans(small, cfg.xy_weight).reshape(-1, 5)

    # random subsample to keep K-means stable and fast
    max_samples = 80000
    if small_features.shape[0] > max_samples:
        rng = np.random.default_rng(1234)
        idx = rng.choice(small_features.shape[0], size=max_samples, replace=False)
        train = small_features[idx]
    else:
        train = small_features

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.6)
    compactness, labels, centers = cv2.kmeans(train.astype(np.float32), cfg.kmeans_k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    full_features = _feature_image_for_kmeans(image_bgr, cfg.xy_weight)
    full_labels = _kmeans_assign_full(full_features, centers)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    bgr = image_bgr

    # Determine background cluster: the dominant cluster on image borders, normally blue tape.
    border = np.zeros((h, w), dtype=bool)
    m = max(4, int(min(h, w) * 0.035))
    border[:m, :] = True; border[-m:, :] = True; border[:, :m] = True; border[:, -m:] = True
    border_labels = full_labels[border & (hole_mask == 0)]
    if border_labels.size:
        bg_label = int(np.bincount(border_labels, minlength=cfg.kmeans_k).argmax())
    else:
        bg_label = 0

    mosquito_labels: List[int] = []
    for k in range(cfg.kmeans_k):
        mask = (full_labels == k) & (hole_mask == 0)
        count = int(np.count_nonzero(mask))
        if count < 50:
            continue
        hsv_mean = hsv[mask].mean(axis=0)
        lab_mean = lab[mask].mean(axis=0)
        bgr_mean = bgr[mask].mean(axis=0)
        H, S, V = hsv_mean
        B, G, R = bgr_mean
        L, A, Bb = lab_mean
        is_background = (k == bg_label)
        is_dark = (V < 125 and L < 145)
        # brown/orange low hue, enough saturation, red component not weaker than blue
        is_brown = (((H < 35) or (H > 165)) and S > 25 and V < 230 and R > B * 0.88)
        # reject very bright glare/holes
        is_glare = (V > 200 and S < 45)
        if (not is_background) and (not is_glare) and (is_dark or is_brown):
            mosquito_labels.append(k)

    mask = np.isin(full_labels, mosquito_labels).astype(np.uint8) * 255
    return mask


def color_rule_mosquito_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Strong color rules for brown and black mosquito body regions."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    h, s, v = cv2.split(hsv)
    L, A, B_lab = cv2.split(lab)
    b, g, r = cv2.split(image_bgr)

    # 검정 몸통: 어둡고, 파란 배경보다 명확히 낮은 L/V
    dark = (v < 118) & (L < 138) & (r < 165) & (g < 165) & (b < 185)

    # 갈색 몸통: hue가 갈색/적갈색 범위, 채도 존재, LAB b 성분이 높고 너무 밝지 않음
    brown_hsv = (((h < 34) | (h > 165)) & (s > 24) & (v < 225) & (r.astype(np.int16) >= (0.78 * b).astype(np.int16)))
    brown_lab = (B_lab > 132) & (A > 118) & (L < 205)
    brown = brown_hsv | brown_lab

    return (dark | brown).astype(np.uint8) * 255


def blackhat_mosquito_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Detect small dark objects on a relatively bright glue-tape background."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # estimate local background and subtract object darkness
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def segment_mosquito_candidates(image_bgr: np.ndarray, hole_mask: np.ndarray, cfg: AnalyzerConfig) -> Tuple[np.ndarray, dict]:
    """Final candidate segmentation mask.

    Combines K-means segmentation, HSV/LAB color rules, and black-hat dark-object extraction.
    """
    k_mask = kmeans_mosquito_mask(image_bgr, hole_mask, cfg)
    c_mask = color_rule_mosquito_mask(image_bgr)
    bh_mask = blackhat_mosquito_mask(image_bgr)

    combined = cv2.bitwise_or(k_mask, c_mask)
    # blackhat is useful but noisy; keep only where not too bright/glare and not hole
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _h, s, v = cv2.split(hsv)
    bh_valid = ((bh_mask > 0) & (v < 185) & (s > 8)).astype(np.uint8) * 255
    combined = cv2.bitwise_or(combined, bh_valid)
    # Do not remove the whole hole interior here. Many mosquitoes sit on top of holes;
    # removing the filled circle would erase valid body pixels. Hole masking is already
    # applied to K-means/blackhat candidates and the strong color mask is preserved.

    # remove small noise and connect body fragments
    combined = cv2.medianBlur(combined, 3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    combined = cv2.dilate(combined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return combined, {"kmeans_mask": k_mask, "color_mask": c_mask, "blackhat_mask": bh_mask}


def object_pixel_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """Mask mosquito pixels inside one crop using HSV/LAB rules."""
    if crop_bgr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    mask = color_rule_mosquito_mask(crop_bgr)
    # within crop, include connected dark features with blackhat too
    mask = cv2.bitwise_or(mask, blackhat_mosquito_mask(crop_bgr))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return mask


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def _merge_boxes(boxes: List[Box], merge_distance: float) -> List[Box]:
    changed = True
    while changed:
        changed = False
        used = [False] * len(boxes)
        merged: List[Box] = []
        for i, a in enumerate(boxes):
            if used[i]:
                continue
            cur = a[:]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                b = boxes[j]
                ca = ((cur[0] + cur[2]) / 2.0, (cur[1] + cur[3]) / 2.0)
                cb = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
                dist = math.hypot(ca[0] - cb[0], ca[1] - cb[1])
                # merge close fragments of one mosquito body, but avoid merging far individual mosquitoes
                if _iou(cur, b) > 0.04 or dist < merge_distance:
                    cur = [min(cur[0], b[0]), min(cur[1], b[1]), max(cur[2], b[2]), max(cur[3], b[3]), cur[4] + b[4]]
                    used[j] = True
                    changed = True
            merged.append(cur)
        boxes = merged
    return boxes


def extract_candidate_boxes(candidate_mask: np.ndarray, hole_mask: Optional[np.ndarray], cfg: AnalyzerConfig, image_bgr: Optional[np.ndarray] = None) -> List[Box]:
    """Find connected components and return expanded/merged candidate boxes.

    Hole-only false positives are removed here, before classification.
    This prevents glue-tape hole shadows from being labeled as brown mosquitoes.
    """
    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = candidate_mask.shape
    boxes: List[Box] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < cfg.min_body_area or area > cfg.max_body_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3:
            continue

        # 핵심 수정:
        # 구멍과 겹친 후보를 무조건 제거하지 않는다.
        # 후보 mask에서 hole mask를 뺀 후 남는 픽셀이 거의 없을 때만 제거한다.
        if is_hole_only_candidate(candidate_mask, hole_mask, (x, y, w, h), cfg, image_bgr=image_bgr):
            continue

        # avoid nearly circular big holes/glare that escaped mask
        peri = cv2.arcLength(cnt, True)
        circ = 0.0 if peri <= 0 else 4.0 * math.pi * area / (peri * peri)
        if circ > 0.82 and area > 140:
            continue

        pad = cfg.candidate_expand_px
        boxes.append([
            float(max(0, x - pad)),
            float(max(0, y - pad)),
            float(min(img_w, x + w + pad)),
            float(min(img_h, y + h + pad)),
            area,
        ])

    return _merge_boxes(boxes, cfg.merge_center_distance_px)


def filter_candidate_box(box: Sequence[float], image_shape: Tuple[int, int], cfg: AnalyzerConfig) -> Tuple[bool, str, str]:
    """Filter one candidate box and assign status."""
    img_h, img_w = image_shape
    x1, y1, x2, y2, score = box
    w = x2 - x1
    h = y2 - y1
    if w < cfg.min_candidate_size_px or h < cfg.min_candidate_size_px:
        return False, "Fragment", "too small"
    if w > cfg.max_candidate_size_px or h > cfg.max_candidate_size_px:
        return True, "Overlapped", "candidate box is unusually large"
    if x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2:
        return True, "Border_Object", "touches image boundary"
    if score > cfg.overlapped_area_px:
        return True, "Overlapped", "large merged object"
    if score < getattr(cfg, "min_countable_fragment_area_px", 80.0):
        return False, "Fragment", "too small to count"
    if score < cfg.fragment_area_px:
        return True, "Review", "small or partial object"
    return True, "OK", ""


def crop_candidate(image_bgr: np.ndarray, bbox_xywh: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox_xywh
    return image_bgr[y:y + h, x:x + w].copy()
