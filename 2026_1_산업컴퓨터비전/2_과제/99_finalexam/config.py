"""Configuration for the improved mosquito glue-tape analyzer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_images"
OUTPUT_DIR = BASE_DIR / "outputs"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_BROWN = "brown_mosquito"        # 갈색모기 / 빨간집모기 후보
CLASS_FOREST = "korean_forest"        # 한국숲모기 후보
CLASS_UNKNOWN = "unknown"
CLASS_REVIEW = "review"

CLASS_DISPLAY_KR = {
    CLASS_BROWN: "빨간집모기",
    CLASS_FOREST: "한국숲모기",
    CLASS_UNKNOWN: "Unknown",
    CLASS_REVIEW: "Review",
}

FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]

@dataclass
class AnalyzerConfig:
    # 처리 속도를 위해 긴 변을 이 크기 이하로 축소해서 분석합니다.
    # 원본 사진이 너무 크면 K-means가 느려지므로 1400 정도가 적당합니다.
    max_working_side: int = 1400

    # 실제 기준 치수입니다. 현재 포획 테이프의 홀 실제 지름은 5.00mm입니다.
    # 이 값은 추정값이 아니며, 픽셀을 mm로 환산하기 위한 기준자입니다.
    # 다른 테이프를 쓰거나 기준 치수를 모르면 None으로 변경하세요.
    hole_pitch_mm: Optional[float] = None
    hole_diameter_mm: Optional[float] = 5.0

    # 결과 overlay 오른쪽에 개체별 크기 index panel 표시
    show_size_index_panel: bool = True
    size_index_panel_width_px: int = 390

    # 구멍 검출 및 제거
    hole_mask_expand: float = 1.28       # 구멍 테두리까지 제거하기 위해 반지름 확장
    min_hole_radius_px: int = 12
    max_hole_radius_px: int = 58

    # 구멍 음영 오검출 제거
    # 후보 mask에서 hole mask를 뺀 뒤 남는 픽셀이 거의 없으면 '구멍만 있는 후보'로 제거합니다.
    hole_only_min_remaining_area: int = 28
    hole_only_min_remaining_ratio: float = 0.10
    hole_only_strong_remaining_area: int = 65
    hole_only_overlap_ratio: float = 0.55
    # 구멍 위에 실제 모기가 있는 경우에는 검정/어두운 다리·몸통 픽셀이 남습니다.
    # 이런 dark evidence가 충분하면 구멍과 겹쳐도 제거하지 않습니다.
    hole_only_dark_keep_area: int = 70
    hole_only_dark_keep_ratio: float = 0.07

    # K-means segmentation
    kmeans_k: int = 4
    kmeans_sample_side: int = 420
    xy_weight: float = 0.06              # RGB/LAB + 위치정보를 같이 쓰기 위한 위치 가중치

    # 후보 객체 필터링
    min_body_area: float = 25.0
    max_body_area: float = 8500.0
    candidate_expand_px: int = 32        # 다리/날개 일부를 포함하기 위해 bbox 확장
    merge_center_distance_px: float = 42.0
    min_candidate_size_px: int = 22
    max_candidate_size_px: int = 340
    fragment_area_px: float = 220.0
    # 이 값보다 작은 contour는 모기 개체가 아니라 다리 조각/먼지/구멍 음영으로 보고 카운트하지 않습니다.
    min_countable_fragment_area_px: float = 80.0
    overlapped_area_px: float = 6500.0

    # 분류 기준
    low_confidence_threshold: float = 0.40
    black_ratio_threshold: float = 0.045
    brown_ratio_threshold: float = 0.045
    crop_resize: Tuple[int, int] = (128, 128)
    # brown/black 실제 픽셀 비율이 이 값보다 낮으면 평균색만으로 종 분류하지 않습니다.
    min_species_color_ratio: float = 0.018

    # 결과 표시
    show_result_window: bool = True
    save_masks: bool = True
