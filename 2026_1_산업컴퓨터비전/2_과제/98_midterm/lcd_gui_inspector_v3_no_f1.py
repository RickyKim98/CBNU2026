# -*- coding: utf-8 -*-
"""
LCD GUI 8개 아이콘 고정 ROI 검사 프로그램 (주석 강화 버전)

[프로그램 개요]
- 정상 이미지와 시험 이미지를 비교하여
  8개 GUI 아이콘의 위치 / 크기 / 색상 이상을 판정합니다.
- 아이콘 위치가 고정되어 있다는 가정을 사용하므로,
  ROI 자동 탐색 없이 미리 지정한 ROI 좌표를 그대로 사용합니다.
- 최종 결과는
  1) CSV 리포트
  2) 정상/시험 이미지 + 우측 리포트 패널이 포함된 결과 이미지
  로 저장됩니다.

[현재 예제 기준]
- 정상 이미지 : re_lcd_std_1_color.png
- 시험 이미지 : re_lcd_std_2_color.png

[필요 패키지]
    pip install opencv-python numpy pandas

[실행 방법]
    python lcd_gui_inspector_v3_no_f1.py
"""

import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# 1. 파일 경로 / 기본 옵션
# ============================================================
# 이미지 경로
NORMAL_IMAGE_PATH   = "re_lcd_std_1_color.png"
TEST_IMAGE_PATH     = "re_lcd_std_2_color.png"

# 결과 저장 폴더
OUTPUT_DIR = "output_lcd_inspection"

# 결과 창을 OpenCV 팝업으로 띄울지 여부
# - True  : cv2.imshow()로 결과 창 표시
# - False : 파일만 저장
SHOW_WINDOWS = True


# ============================================================
# 2. ROI 설정
# ============================================================
# ROI를 미리 고정
#
# 상단 5개 아이콘
# - top_bulb : 좌측 상단 전구 아이콘
# - mode     : M 아이콘
# - plug     : 플러그 아이콘
# - blind    : 블라인드 아이콘
# - gear     : 톱니바퀴 아이콘
#
# 중앙 세로 3개 버튼
# - btn1 / btn2 / btn3 : 원형 링 버튼
ROI_CONFIG = {
    "top_bulb": (0,   0,   60,  65),
    "mode"    : (80,  0,   150, 65),
    "plug"    : (155, 0,   235, 65),
    "blind"   : (245, 0,   315, 65),
    "gear"    : (315, 0,   377, 65),

    "btn1"    : (145, 115, 255, 230),
    "btn2"    : (145, 285, 255, 405),
    "btn3"    : (145, 455, 255, 575),
}

# ROI 종류를 따로 구분합니다.
# 이유:
# - 색상 아이콘(top_bulb)은 Hue 기반 검사가 중요하고,
# - 흰색 아이콘(mode, plug, blind, gear)은 밝기 기반 검사가 더 적합하며,
# - 원형 링 버튼(btn1~btn3)은 파란색 링의 Hue와 이 중요하기 때문입니다.
ROI_TYPE = {
    "top_bulb": "color_icon",
    "mode"    : "white_icon",
    "plug"    : "white_icon",
    "blind"   : "white_icon",
    "gear"    : "white_icon",
    "btn1"    : "ring_button",
    "btn2"    : "ring_button",
    "btn3"    : "ring_button",
}

# ROI 종류별 판정 기준값입니다.
# - pos_max         : 중심 위치 차이 허용값
# - area_ratio_min  : edge 픽셀 수 비율 허용 최소값
# # - color_diff_max  : 색상 차이 허용 최대값
THRESHOLDS = {
    "color_icon": {
        "pos_max": 12.0,
        "area_ratio_min": 0.65,
                "color_diff_max": 12.0,
    },
    "white_icon": {
        "pos_max": 10.0,
        "area_ratio_min": 0.70,
                "color_diff_max": 30.0,
    },
    "ring_button": {
        "pos_max": 5.0,
        "area_ratio_min": 0.90,
                "color_diff_max": 8.0,
    },
}


# ============================================================
# 3. 공통 유틸 함수
# ============================================================
def ensure_dir(path: str) -> None:
    """
    결과 저장 폴더가 없으면 생성합니다.
    """
    if not os.path.exists(path):
        os.makedirs(path)



def circular_hue_diff(h1: float, h2: float) -> float:
    """
    OpenCV HSV Hue는 0~179 범위를 사용합니다.
    Hue는 원형 값이므로 단순 차이(abs)만 쓰면 안 됩니다.

    예:
    - Hue 2 와 Hue 178 은 실제로는 매우 가까운 색입니다.
    - 단순 abs(2 - 178) = 176 이 아니라,
      원형 차이로는 min(176, 180-176)=4 가 되어야 합니다.
    """
    diff = abs(float(h1) - float(h2))
    return min(diff, 180.0 - diff)



def crop_roi(image: np.ndarray, roi: tuple) -> np.ndarray:
    """
    이미지에서 지정 ROI만 잘라서 반환합니다.
    roi 형식: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2].copy()



def fit_text_right(canvas, text, x_right, y, font, scale, color, thickness=1, min_x=0):
    """
    우측 정렬 텍스트 출력 함수

    [용도]
    우측 리포트 패널에서 숫자 컬럼을 보기 좋게 정렬하기 위해 사용합니다.

    [동작]
    - 문자열 폭을 계산한 뒤
    - x_right 기준으로 오른쪽 정렬하여 출력합니다.
    - min_x 보다 왼쪽으로 넘어가지 않도록 제한합니다.
    """
    (w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(min_x, x_right - w)
    cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


# ============================================================
# 4.  특징 추출 함수
# ============================================================
def extract_edge_geometry(roi_bgr: np.ndarray):
    """
    ROI에서 edge 기반  정보를 추출합니다.

    [처리 순서]
    1) BGR -> Gray 변환
    2) GaussianBlur로 약한 노이즈 완화
    3) Canny로 edge 추출
    4) 가장자리 2픽셀 제거
       - ROI 경계선 때문에 생기는 불필요한 외곽선을 억제하기 위함
    5) edge 픽셀의 중심점(centroid), bbox, edge 면적(area) 계산

    [반환값]
    edge가 있으면 dict 반환:
        {
            "centroid": (cx, cy),
            "bbox": (x, y, w, h),
            "area": edge 픽셀 개수,
            "edge": edge binary image
        }
    edge가 없으면 None 반환
    """
    # 컬러 영상을 그레이 영상으로 변환
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # 약한 노이즈를 줄여 Canny edge가 너무 지저분하게 나오지 않게 함
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Canny edge 추출
    edge = cv2.Canny(gray, 40, 120)

    # ROI 테두리 근처의 경계선 제거
    # - ROI를 자른 경계 자체가 edge로 검출되는 것을 줄이기 위한 처리
    edge[:2, :] = 0
    edge[-2:, :] = 0
    edge[:, :2] = 0
    edge[:, -2:] = 0

    # edge가 있는 좌표 추출
    ys, xs = np.where(edge > 0)

    # edge가 전혀 없으면  특징을 계산할 수 없으므로 None 반환
    if len(xs) == 0:
        return None

    # edge 픽셀 중심 좌표 계산
    cx = float(xs.mean())
    cy = float(ys.mean())

    # edge 전체를 감싸는 bounding box 계산
    x1 = int(xs.min())
    x2 = int(xs.max())
    y1 = int(ys.min())
    y2 = int(ys.max())

    return {
        "centroid": (cx, cy),
        "bbox": (x1, y1, x2 - x1 + 1, y2 - y1 + 1),
        "area": int(len(xs)),
        "edge": edge,
    }





# ============================================================
# 5. 색상 특징 추출 함수
# ============================================================
def dominant_hue_peak(roi_bgr: np.ndarray, roi_kind: str):
    """
    ROI에서 대표 Hue(주요 색상 peak)를 구합니다.

    [사용 대상]
    - color_icon  : 상단 좌측 bulb 아이콘
    - ring_button : 중앙 원형 버튼 3개

    [처리 방식]
    1) BGR -> HSV 변환
    2) 채도(S)가 충분히 높고, 밝기(V)가 너무 낮지 않은 픽셀만 사용
    3) ring_button은 파란색 범위(H 80~130)만 추가로 제한
    4) Hue histogram에서 최빈값(peak)을 대표 Hue로 사용

    [반환값]
    - peak_h : 가장 많이 나온 Hue 값
    - count  : 실제로 사용된 유효 픽셀 개수
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 색이 충분히 있고, 너무 어둡지 않은 픽셀만 선택
    mask = (s > 100) & (v > 40)

    # 원형 링 버튼은 파란색 계열만 보도록 제한
    if roi_kind == "ring_button":
        mask &= (h >= 80) & (h <= 130)

    hue_values = h[mask]

    # 유효 픽셀이 없으면 None 반환
    if hue_values.size == 0:
        return None, 0

    # Hue histogram에서 가장 빈도가 높은 값 선택
    hist = np.bincount(hue_values, minlength=180)
    peak_h = int(np.argmax(hist))
    return peak_h, int(hue_values.size)



def white_icon_brightness(roi_bgr: np.ndarray):
    """
    흰색 아이콘용 밝기 특징 추출 함수

    [사용 대상]
    - mode / plug / blind / gear

    [아이디어]
    흰색 아이콘은 Hue보다 밝기 정보가 더 안정적이므로,
    밝은 영역의 평균 Value를 대표값으로 사용합니다.

    [처리 방식]
    1) BGR -> HSV 변환
    2) 밝기 V가 충분히 크고, 채도 S가 낮은 픽셀만 사용
       -> 흰색/회색 계열만 남기기 위한 조건
    3) 해당 픽셀들의 평균 V를 계산
    """
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    # 밝은데 채도는 낮은 영역만 선택 -> 흰색 아이콘 후보
    mask = (v > 120) & (s < 90)
    values = v[mask]

    if values.size == 0:
        return None, 0

    return float(values.mean()), int(values.size)



def extract_color_feature(roi_bgr: np.ndarray, roi_kind: str):
    """
    ROI 종류에 따라 사용할 색상 특징을 선택합니다.

    [반환 형식]
    {
        "mode": "hue" 또는 "brightness",
        "value": 대표 색상값,
        "count": 유효 픽셀 수
    }
    """
    # 색상 아이콘 / 링 버튼 -> Hue peak 사용
    if roi_kind in ("color_icon", "ring_button"):
        hue_peak, count = dominant_hue_peak(roi_bgr, roi_kind)
        return {"mode": "hue", "value": hue_peak, "count": count}

    # 흰색 아이콘 -> 평균 밝기 사용
    if roi_kind == "white_icon":
        mean_v, count = white_icon_brightness(roi_bgr)
        return {"mode": "brightness", "value": mean_v, "count": count}

    # 예외 처리
    return {"mode": "unknown", "value": None, "count": 0}



def compute_color_diff(ref_color: dict, test_color: dict) -> float:
    """
    정상 ROI와 시험 ROI의 색상 차이를 계산합니다.

    [Hue 모드]
    - 원형 Hue 차이를 사용

    [Brightness 모드]
    - 평균 밝기 절대값 차이를 사용

    [예외]
    - 어느 한쪽이라도 색상 특징이 없으면 큰 값(9999)을 반환하여 FAIL 유도
    """
    if ref_color["value"] is None or test_color["value"] is None:
        return 9999.0

    if ref_color["mode"] == "hue":
        return circular_hue_diff(ref_color["value"], test_color["value"])

    if ref_color["mode"] == "brightness":
        return abs(float(ref_color["value"]) - float(test_color["value"]))

    return 9999.0


# ============================================================
# 6. ROI 1개 검사 함수
# ============================================================
def inspect_one_roi(ref_img: np.ndarray, test_img: np.ndarray, roi_name: str):
    """
    하나의 ROI에 대해 정상/시험 비교 검사를 수행합니다.

    [검사 항목]
    1)     1) 위치(position)
       - edge 중심점 차이
    2) 크기(size)
       - edge 픽셀 수 비율
    3) 색상(color)
       - ROI 종류에 따라 Hue 차이 또는 밝기 차이

    [반환값]
    리포트용 dict 반환
    """
    # 현재 ROI의 좌표 / 종류 / 기준값 로드
    roi = ROI_CONFIG[roi_name]
    roi_kind = ROI_TYPE[roi_name]
    thr = THRESHOLDS[roi_kind]

    # 정상 / 시험 이미지에서 같은 위치의 ROI 추출
    ref_roi = crop_roi(ref_img, roi)
    test_roi = crop_roi(test_img, roi)

    #  특징(edge 기반) 추출
    ref_geom = extract_edge_geometry(ref_roi)
    test_geom = extract_edge_geometry(test_roi)

    # edge를 못 찾으면  검사 자체가 불가능하므로 FAIL 처리
    if ref_geom is None or test_geom is None:
        return {
            "roi_name": roi_name,
            "roi_kind": roi_kind,
            "status": "FAIL",
            "reason": "edge_fail",
            "position_diff": None,
            "area_ratio": None,
            "color_diff": None,
            "ref_color_value": None,
            "test_color_value": None,
            "ref_color_count": None,
            "test_color_count": None,
            "ref_bbox": None,
            "test_bbox": None,
        }

    # ------------------------------------------------------------
    # 1) 위치 차이 계산
    # ------------------------------------------------------------
    # edge 중심점(centroid) 간의 유클리드 거리 계산
    ref_c = np.array(ref_geom["centroid"], dtype=np.float32)
    test_c = np.array(test_geom["centroid"], dtype=np.float32)
    position_diff = float(np.linalg.norm(ref_c - test_c))

    # ------------------------------------------------------------
    # 2) 크기 차이 계산
    # ------------------------------------------------------------
    # edge 픽셀 수를 면적처럼 사용하여 비율 계산
    ref_area = ref_geom["area"]
    test_area = test_geom["area"]
    area_ratio = float(min(ref_area, test_area) / max(ref_area, test_area))

    # ------------------------------------------------------------
    # 3) 색상 차이 계산
    # ------------------------------------------------------------
    ref_color = extract_color_feature(ref_roi, roi_kind)
    test_color = extract_color_feature(test_roi, roi_kind)
    color_diff = float(compute_color_diff(ref_color, test_color))

    # ------------------------------------------------------------
    # 항목별 PASS / FAIL 판정
    # ------------------------------------------------------------
    position_ok = position_diff <= thr["pos_max"]
    area_ok = area_ratio >= thr["area_ratio_min"]
    color_ok = color_diff <= thr["color_diff_max"]

    # 3개 항목이 모두 정상이어야 최종 PASS
    all_ok = position_ok and area_ok and color_ok

    # 어떤 항목이 실패했는지 문자열로 기록
    fail_reasons = []
    if not position_ok:
        fail_reasons.append("position")
    if not area_ok:
        fail_reasons.append("size")
    if not color_ok:
        fail_reasons.append("color")

    # 리포트용 dict 반환
    return {
        "roi_name": roi_name,
        "roi_kind": roi_kind,
        "status": "PASS" if all_ok else "FAIL",
        "reason": "normal" if all_ok else ",".join(fail_reasons),
        "position_diff": round(position_diff, 2),
        "area_ratio": round(area_ratio, 4),
        "color_diff": round(color_diff, 2),
        "ref_color_value": ref_color["value"],
        "test_color_value": test_color["value"],
        "ref_color_count": ref_color["count"],
        "test_color_count": test_color["count"],
        "ref_bbox": ref_geom["bbox"],
        "test_bbox": test_geom["bbox"],
    }


# ============================================================
# 7. 시각화 함수
# ============================================================
def draw_roi_overlay(target: np.ndarray, results: list, x_offset: int, y_offset: int):
    """
    시험 이미지 영역 위에 ROI 사각형과 일부 라벨을 표시합니다.

    [표시 규칙]
    - PASS : 초록색 박스
    - FAIL : 빨간색 박스
    - 상단 5개 아이콘은 글자가 서로 겹치기 쉬우므로
      이미지 위 라벨은 생략하고 우측 리포트 패널에서만 표시합니다.
    - 하단 버튼 3개(btn1~btn3)는 이미지 위에 라벨을 표시합니다.
    """
    for row in results:
        name = row["roi_name"]
        x1, y1, x2, y2 = ROI_CONFIG[name]

        # PASS/FAIL에 따라 박스 색상 설정
        color = (0, 255, 0) if row["status"] == "PASS" else (0, 0, 255)

        # ROI 박스 표시
        cv2.rectangle(
            target,
            (x_offset + x1, y_offset + y1),
            (x_offset + x2, y_offset + y2),
            color,
            2
        )

        # 상단 아이콘은 라벨 생략
        if y1 < 80:
            continue

        # 하단 버튼은 라벨 표시
        label = f"{name}: {row['status']}"
        label_y = max(y_offset + 22, y_offset + y1 - 10)
        cv2.putText(
            target,
            label,
            (x_offset + x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            color,
            2,
            cv2.LINE_AA,
        )



def draw_report_panel(canvas: np.ndarray, df: pd.DataFrame, overall_result: str,
                      x0: int, y0: int, panel_w: int, panel_h: int):
    """
    오른쪽 리포트 패널을 그립니다.

    [패널 내용]
    - 제목
    - 전체 PASS / FAIL
    - 각 ROI별 검사 결과 표
      ROI / TYPE / ST / REASON / POS / AREA / CLR
    - 하단 PASS / FAIL 범례
    """
    # 패널 색상 정의
    panel_color = (248, 248, 248)
    border_color = (210, 210, 210)
    title_color = (40, 40, 40)
    good_color = (0, 150, 0)
    bad_color = (0, 0, 220)
    text_color = (35, 35, 35)

    # 패널 배경과 외곽선
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1), panel_color, -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w - 1, y0 + panel_h - 1), border_color, 2)

    # ------------------------------------------------------------
    # 제목 영역
    # ------------------------------------------------------------
    cv2.putText(canvas, "INSPECTION REPORT", (x0 + 16, y0 + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, title_color, 2, cv2.LINE_AA)

    result_color = good_color if overall_result == "PASS" else bad_color
    cv2.putText(canvas, f"OVERALL : {overall_result}", (x0 + 16, y0 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, result_color, 2, cv2.LINE_AA)

    cv2.line(canvas, (x0 + 12, y0 + 92), (x0 + panel_w - 12, y0 + 92), border_color, 1)

    # ------------------------------------------------------------
    # 컬럼 헤더
    # ------------------------------------------------------------
    header_y = y0 + 120
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(canvas, "ROI",    (x0 + 16,  header_y), font, 0.46, title_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "TYPE",   (x0 + 112, header_y), font, 0.46, title_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "ST",     (x0 + 224, header_y), font, 0.46, title_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, "REASON", (x0 + 270, header_y), font, 0.46, title_color, 1, cv2.LINE_AA)

    # 숫자 컬럼은 우측 정렬하여 보기 좋게 배치
    fit_text_right(canvas, "POS",  x0 + 390, header_y, font, 0.46, title_color, 1, x0 + 350)
    fit_text_right(canvas, "AREA", x0 + 468, header_y, font, 0.46, title_color, 1, x0 + 420)
    fit_text_right(canvas, "COLOR",x0 + 548, header_y, font, 0.46, title_color, 1, x0 + 500)

    cv2.line(canvas, (x0 + 12, header_y + 10), (x0 + panel_w - 12, header_y + 10), border_color, 1)

    # ------------------------------------------------------------
    # 표 본문(각 ROI 결과 행 출력)
    # ------------------------------------------------------------
    y = header_y + 34
    row_h = 34

    for _, row in df.iterrows():
        # 패널 아래로 넘치면 출력 중단
        if y > y0 + panel_h - 20:
            break

        st_color = good_color if row["status"] == "PASS" else bad_color

        # 문자열 컬럼 출력
        cv2.putText(canvas, str(row["roi_name"]), (x0 + 16, y), font, 0.43, text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(row["roi_kind"]), (x0 + 112, y), font, 0.36, text_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(row["status"]), (x0 + 224, y), font, 0.43, st_color, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(row["reason"]), (x0 + 270, y), font, 0.39, text_color, 1, cv2.LINE_AA)

        # 숫자 컬럼 출력
        fit_text_right(canvas, f"{row['position_diff']:.2f}", x0 + 398, y, font, 0.41, text_color, 1, x0 + 350)
        fit_text_right(canvas, f"{row['area_ratio']:.4f}",   x0 + 476, y, font, 0.41, text_color, 1, x0 + 420)
        fit_text_right(canvas, f"{row['color_diff']:.2f}",   x0 + 556, y, font, 0.41, text_color, 1, x0 + 500)

        # 각 행 구분선
        cv2.line(canvas, (x0 + 12, y + 10), (x0 + panel_w - 12, y + 10), (232, 232, 232), 1)
        y += row_h

    # ------------------------------------------------------------
    # 하단 범례
    # ------------------------------------------------------------
    legend_y = y0 + panel_h - 40

    cv2.rectangle(canvas, (x0 + 18, legend_y - 12), (x0 + 42, legend_y + 8), (0, 255, 0), 2)
    cv2.putText(canvas, "PASS", (x0 + 50, legend_y + 3), font, 0.46, good_color, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (x0 + 130, legend_y - 12), (x0 + 154, legend_y + 8), (0, 0, 255), 2)
    cv2.putText(canvas, "FAIL", (x0 + 162, legend_y + 3), font, 0.46, bad_color, 1, cv2.LINE_AA)



def make_dashboard(ref_img: np.ndarray, test_img: np.ndarray, results: list,
                   df: pd.DataFrame, overall_result: str):
    """
    최종 결과 대시보드를 생성합니다.

    [화면 구성]
    - 좌측 : 정상 이미지
    - 중앙 : 시험 이미지 + ROI 검사 결과 박스
    - 우측 : 리포트 패널
    """
    h_img, w_img = ref_img.shape[:2]

    # 이미지 위쪽에 제목을 넣기 위한 헤더 높이
    header_h = 78

    # 오른쪽 리포트 패널 폭
    report_w = 620

    # 시험 이미지와 리포트 패널 사이 여백
    gap = 18

    panel_h = header_h + h_img
    canvas_h = panel_h
    canvas_w = w_img + w_img + report_w + gap * 2

    # 전체 배경 캔버스 생성 (어두운 회색 배경)
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)

    # 각 영역 시작 좌표 계산
    x_ref = 0
    x_test = w_img
    y_img = header_h
    x_report = w_img * 2 + gap

    # ------------------------------------------------------------
    # 상단 헤더 배경
    # ------------------------------------------------------------
    cv2.rectangle(canvas, (0, 0), (w_img - 1, header_h - 1), (35, 35, 35), -1)
    cv2.rectangle(canvas, (x_test, 0), (x_test + w_img - 1, header_h - 1), (35, 35, 35), -1)

    # ------------------------------------------------------------
    # 정상 / 시험 이미지 배치
    # ------------------------------------------------------------
    canvas[y_img:y_img + h_img, x_ref:x_ref + w_img] = ref_img
    canvas[y_img:y_img + h_img, x_test:x_test + w_img] = test_img

    # ------------------------------------------------------------
    # 상단 제목 텍스트
    # ------------------------------------------------------------
    cv2.putText(canvas, "NORMAL IMAGE", (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.96, (0, 255, 255), 2, cv2.LINE_AA)

    #cv2.putText(canvas, f"OVERALL : {overall_result}", (18, 64),
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.96,
    #            (0, 255, 0) if overall_result == "PASS" else (0, 0, 255),
    #            2, cv2.LINE_AA)

    cv2.putText(canvas, "TEST IMAGE + RESULT", (x_test + 18, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.92, (0, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------
    # 시험 이미지 위에 ROI 결과 오버레이 표시
    # ------------------------------------------------------------
    draw_roi_overlay(canvas, results, x_test, y_img)

    # ------------------------------------------------------------
    # 오른쪽 리포트 패널 표시
    # ------------------------------------------------------------
    draw_report_panel(canvas, df, overall_result, x_report, 0, report_w, panel_h)

    # ------------------------------------------------------------
    # 구분선 표시
    # ------------------------------------------------------------
    cv2.line(canvas, (x_test, 0), (x_test, canvas_h - 1), (70, 70, 70), 1)
    cv2.line(canvas, (x_report - gap // 2, 0), (x_report - gap // 2, canvas_h - 1), (70, 70, 70), 1)

    return canvas


# ============================================================
# 8. 메인 함수
# ============================================================
def main():
    """
    프로그램 전체 실행 흐름

    [실행 순서]
    1) 결과 폴더 생성
    2) 정상 / 시험 이미지 읽기
    3) 크기가 다르면 시험 이미지를 정상 이미지 크기에 맞춤
    4) 8개 ROI를 순서대로 검사
    5) DataFrame 생성 및 전체 PASS / FAIL 판정
    6) 대시보드 이미지 생성
    7) CSV / PNG 저장
    8) 콘솔 출력
    9) 필요 시 OpenCV 창 표시
    """
    # 결과 폴더 생성
    ensure_dir(OUTPUT_DIR)

    # 정상 / 시험 이미지 읽기
    ref_img = cv2.imread(NORMAL_IMAGE_PATH)
    test_img = cv2.imread(TEST_IMAGE_PATH)

    # 파일이 없으면 즉시 에러 발생
    if ref_img is None:
        raise FileNotFoundError(f"정상 이미지를 읽을 수 없습니다: {NORMAL_IMAGE_PATH}")
    if test_img is None:
        raise FileNotFoundError(f"시험 이미지를 읽을 수 없습니다: {TEST_IMAGE_PATH}")

    # 두 이미지 크기가 다르면 시험 이미지를 정상 이미지 크기에 맞춤
    if ref_img.shape[:2] != test_img.shape[:2]:
        test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

    # ROI 순서대로 검사 수행
    results = [inspect_one_roi(ref_img, test_img, roi_name) for roi_name in ROI_CONFIG.keys()]

    # 결과를 표 형태로 다루기 쉽게 DataFrame으로 변환
    df = pd.DataFrame(results)

    # 모든 ROI가 PASS이면 전체 PASS
    overall_ok = bool((df["status"] == "PASS").all())
    overall_result = "PASS" if overall_ok else "FAIL"

    # 최종 결과 대시보드 이미지 생성
    dashboard = make_dashboard(ref_img, test_img, results, df, overall_result)

    # 저장 파일 경로 설정
    out_csv = os.path.join(OUTPUT_DIR, "inspection_result.csv")
    out_img = os.path.join(OUTPUT_DIR, "inspection_result.png")

    # CSV 저장
    # 인코딩은 Excel에서 한글이 깨지지 않도록 utf-8-sig 사용
    csv_df = df.copy()
    csv_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 결과 이미지 저장
    cv2.imwrite(out_img, dashboard)

    # ------------------------------------------------------------
    # 콘솔 출력
    # ------------------------------------------------------------
    print("\n================ LCD GUI 검사 결과 ================\n")
    print(df[[
        "roi_name", "roi_kind", "status", "reason",
        "position_diff", "area_ratio", "color_diff"
    ]].to_string(index=False))

    print("\n---------------------------------------------------")
    print(f"전체 판정 : {overall_result}")
    print(f"결과 CSV  : {out_csv}")
    print(f"결과 이미지: {out_img}")
    print("---------------------------------------------------\n")

    # ------------------------------------------------------------
    # OpenCV 창 표시
    # ------------------------------------------------------------
    if SHOW_WINDOWS:
        cv2.imshow("LCD GUI Inspection Result", dashboard)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================
# 9. 프로그램 시작점
# ============================================================
if __name__ == "__main__":
    main()
