"""High-level analyzer: selected image -> masks/crops/overlay/summary."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    AnalyzerConfig,
    CLASS_BROWN,
    CLASS_DISPLAY_KR,
    CLASS_FOREST,
    CLASS_UNKNOWN,
    FONT_CANDIDATES,
    OUTPUT_DIR,
)
from features_classifier import (
    CandidateResult,
    classify_candidate,
    extract_color_features,
    extract_hog_features,
    extract_shape_features,
)
from vision_algorithms import (
    create_hole_mask,
    crop_candidate,
    detect_holes,
    estimate_scale_from_holes,
    extract_candidate_boxes,
    filter_candidate_box,
    object_pixel_mask,
    resize_for_work,
    segment_mosquito_candidates,
)


def read_image_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def write_image(path: Path, image_bgr: np.ndarray) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, image_bgr)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def prepare_output_dirs(base_dir: Path = OUTPUT_DIR) -> Dict[str, Path]:
    dirs = {
        "base": base_dir,
        "overlay": base_dir / "overlay",
        "crops": base_dir / "crops",
        "masks": base_dir / "masks",
        "reports": base_dir / "reports",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _load_korean_font(size: int = 24) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in FONT_CANDIDATES:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


def _draw_size_index_panel(
    pil_img: Image.Image,
    results: List[CandidateResult],
    panel_width: int,
) -> Image.Image:
    """Append a white right-side index panel that lists each detected mosquito measurement."""
    src_w, src_h = pil_img.size
    panel_width = max(260, int(panel_width))
    out = Image.new("RGB", (src_w + panel_width, src_h), (255, 255, 255))
    out.paste(pil_img, (0, 0))

    draw = ImageDraw.Draw(out)
    title_font = _load_korean_font(25)
    text_font = _load_korean_font(21)
    small_font = _load_korean_font(18)

    x0 = src_w
    margin = 18
    y = 24

    # panel separator
    draw.line([(x0, 0), (x0, src_h)], fill=(210, 210, 210), width=2)

    title = "개체별 크기 Index"
    draw.text((x0 + margin, y), title, fill=(0, 0, 0), font=title_font)
    y += 40

    valid_lengths = [r.length_mm for r in results if r.length_mm is not None]
    if valid_lengths:
        avg = sum(valid_lengths) / len(valid_lengths)
        draw.text((x0 + margin, y), f"평균 크기: {avg:.1f} mm", fill=(70, 70, 70), font=small_font)
    else:
        draw.text((x0 + margin, y), "평균 크기: N/A", fill=(70, 70, 70), font=small_font)
    y += 32

    # header line
    draw.text((x0 + margin, y), "ID   종류          크기", fill=(80, 80, 80), font=small_font)
    y += 25
    draw.line([(x0 + margin, y), (x0 + panel_width - margin, y)], fill=(225, 225, 225), width=1)
    y += 12

    colors = {
        CLASS_BROWN: (255, 145, 0),
        CLASS_FOREST: (0, 100, 255),
        CLASS_UNKNOWN: (160, 150, 0),
    }

    row_h = 32
    max_y = src_h - 42
    drawn = 0
    for res in results:
        if y + row_h > max_y:
            remain = len(results) - drawn
            if remain > 0:
                draw.text((x0 + margin, y), f"... 외 {remain}개", fill=(90, 90, 90), font=small_font)
            break

        color = colors.get(res.final_class, colors[CLASS_UNKNOWN])
        name = CLASS_DISPLAY_KR.get(res.final_class, "Unknown")
        size_text = "N/A" if res.length_mm is None else f"{res.length_mm:.1f} mm"
        status = "" if res.status == "OK" else f" ({res.status})"

        # Color marker and row text
        draw.rectangle([x0 + margin, y + 7, x0 + margin + 13, y + 20], outline=color, fill=color)
        row = f"ID{res.id:02d}  {name:<6}  {size_text}{status}"
        draw.text((x0 + margin + 22, y), row, fill=(0, 0, 0), font=text_font)
        y += row_h
        drawn += 1

    note_y = max(y + 8, src_h - 36)
    if note_y < src_h - 10:
        draw.text((x0 + margin, note_y), "※ 실제 홀 5.00mm 기준 환산 길이", fill=(100, 100, 100), font=small_font)

    return out


def draw_overlay(image_bgr: np.ndarray, results: List[CandidateResult], config: Optional[AnalyzerConfig] = None) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _load_korean_font(23)
    colors = {
        CLASS_BROWN: (255, 145, 0),
        CLASS_FOREST: (0, 100, 255),
        CLASS_UNKNOWN: (255, 230, 0),
    }
    for res in results:
        x, y, w, h = res.bbox
        color = colors.get(res.final_class, colors[CLASS_UNKNOWN])
        if res.status not in {"OK", "Review"}:
            color = (255, 0, 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=4)
        name = CLASS_DISPLAY_KR.get(res.final_class, "Unknown")
        text = f"ID{res.id:02d} {name} {res.confidence:.2f}"
        if res.length_mm is not None:
            text += f" {res.length_mm:.1f}mm"
        if res.status != "OK":
            text += f" {res.status}"
        tx, ty = x, max(2, y - 30)
        tb = draw.textbbox((tx, ty), text, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=(0, 0, 0))
        draw.text((tx, ty), text, fill=color, font=font)

    cfg = config or AnalyzerConfig()
    if getattr(cfg, "show_size_index_panel", True):
        pil_img = _draw_size_index_panel(
            pil_img,
            results,
            getattr(cfg, "size_index_panel_width_px", 390),
        )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def count_results(results: List[CandidateResult]) -> Dict[str, int]:
    counts = {"korean_forest": 0, "brown_mosquito": 0, "unknown_review": 0, "total": len(results)}
    for r in results:
        if r.final_class == CLASS_FOREST:
            counts["korean_forest"] += 1
        elif r.final_class == CLASS_BROWN:
            counts["brown_mosquito"] += 1
        else:
            counts["unknown_review"] += 1
    return counts


def save_summary(path: Path, image_name: str, holes_count: int, results: List[CandidateResult], scale_note: str) -> None:
    counts = count_results(results)
    lines: List[str] = [
        f"Image: {image_name}",
        f"Detected holes: {holes_count}",
        f"Scale: {scale_note}",
        f"Detected mosquito candidates: {len(results)}",
        "",
        "Final Summary:",
        f"한국숲모기: {counts['korean_forest']}",
        f"갈색모기: {counts['brown_mosquito']}",
        f"Unknown/Review: {counts['unknown_review']}",
        f"Total: {counts['total']}",
        "",
        "Detail:",
    ]
    for res in results:
        name = CLASS_DISPLAY_KR.get(res.final_class, "Unknown")
        length = "N/A" if res.length_mm is None else f"{res.length_mm:.2f} mm"
        cf = res.color_features
        sf = res.shape_features
        lines.append(
            f"ID{res.id:03d} | {name} | conf={res.confidence:.2f} | length={length} | "
            f"status={res.status} | reason={res.reason} | "
            f"brown={cf.get('brown_ratio', 0):.2f}, black={cf.get('black_ratio', 0):.2f}, "
            f"aspect={sf.get('aspect_ratio', 0):.2f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_image(image_path: str | Path, config: Optional[AnalyzerConfig] = None, output_dir: Path = OUTPUT_DIR) -> Tuple[List[CandidateResult], Path]:
    cfg = config or AnalyzerConfig()
    path = Path(image_path)
    original = read_image_bgr(path)
    if original is None:
        raise RuntimeError(f"이미지를 읽을 수 없습니다: {path}")

    dirs = prepare_output_dirs(output_dir)
    image, scale = resize_for_work(original, cfg.max_working_side)
    print(f"\nProcessing: {path.name}")
    if scale < 1.0:
        print(f"Working image resized: scale={scale:.3f}, size={image.shape[1]}x{image.shape[0]}")

    holes = detect_holes(image, cfg)
    hole_mask = create_hole_mask(image.shape[:2], holes, cfg)
    pixel_per_mm, scale_note = estimate_scale_from_holes(holes, cfg.hole_pitch_mm, cfg.hole_diameter_mm)

    candidate_mask, debug_masks = segment_mosquito_candidates(image, hole_mask, cfg)
    raw_boxes = extract_candidate_boxes(candidate_mask, hole_mask, cfg, image_bgr=image)

    if cfg.save_masks:
        write_image(dirs["masks"] / f"{path.stem}_hole_mask.png", hole_mask)
        write_image(dirs["masks"] / f"{path.stem}_candidate_mask.png", candidate_mask)
        for name, mask in debug_masks.items():
            write_image(dirs["masks"] / f"{path.stem}_{name}.png", mask)

    results: List[CandidateResult] = []
    for box in raw_boxes:
        keep, status, reason = filter_candidate_box(box, image.shape[:2], cfg)
        if not keep:
            continue
        x1, y1, x2, y2, score = box
        x, y = int(round(x1)), int(round(y1))
        w, h = int(round(x2 - x1)), int(round(y2 - y1))
        crop = crop_candidate(image, (x, y, w, h))
        if crop.size == 0:
            continue

        local_mask = object_pixel_mask(crop)
        color = extract_color_features(crop)
        shape = extract_shape_features(local_mask)
        hog = extract_hog_features(crop, cfg.crop_resize)
        res = CandidateResult(
            id=len(results) + 1,
            bbox=(x, y, w, h),
            contour_area=float(score),
            status=status,
            reason=reason,
            color_features=color,
            shape_features=shape,
            hog_features=hog,
        )
        if pixel_per_mm:
            length_px = max(shape.get("major_axis", 0.0), float(max(w, h) * 0.58))
            res.length_mm = float(length_px / pixel_per_mm)
        classify_candidate(res, cfg)

        crop_name = f"{path.stem}_ID{res.id:03d}_{res.final_class}.jpg"
        crop_path = dirs["crops"] / crop_name
        write_image(crop_path, crop)
        res.crop_path = crop_path
        results.append(res)

    overlay = draw_overlay(image, results, cfg)
    overlay_path = dirs["overlay"] / f"{path.stem}_overlay.jpg"
    write_image(overlay_path, overlay)

    summary_path = dirs["reports"] / f"{path.stem}_summary.txt"
    save_summary(summary_path, path.name, len(holes), results, scale_note)

    counts = count_results(results)
    print(f"Detected holes: {len(holes)}")
    print(f"Detected mosquito candidates: {len(results)}")
    print("\n[종류별 개체 수]")
    print(f"한국숲모기: {counts['korean_forest']} 마리")
    print(f"빨간집모기: {counts['brown_mosquito']} 마리")
    print(f"Unknown/Review: {counts['unknown_review']} 마리")
    print(f"Total: {counts['total']} 마리")
    print(f"\nSaved overlay: {overlay_path}")
    print(f"Saved summary: {summary_path}")
    return results, overlay_path
