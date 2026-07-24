import cv2
import numpy as np

# ------------------------------------------------------------
# 세로 이미지용 ROI similarity 비교 프로그램 (v10)
#
# [기능 요약]
# 1) 정상(left), 비정상(right) 이미지를 나란히 띄운다.
# 2) AUTO 모드에서는 미리 정한 아이콘 ROI들을 빨간 박스로 표시한다.
# 3) MANUAL 모드에서는 오른쪽 이미지에서 마우스로 ROI를 직접 지정한다.
# 4) ROI 안에서 발광 영역만 분리한 뒤,
#    - 평균 S(채도)가 높으면 -> Hue 비교
#    - 평균 S(채도)가 낮으면 -> Brightness(V) 비교
#    방식으로 similarity(%)를 계산한다.
# 5) m 키: MANUAL 모드, a 키: AUTO 모드, r 키: 수동 ROI 초기화, s 키: 저장
#
# [중요 수정 사항]
# - similarity 계산에 쓰는 기준값을 상수로 분리했다.
#   예) HUE_SIMILARITY_BASE = 90.0
#       BRIGHT_SIMILARITY_BASE = 255.0
#   필요하면 이 값을 직접 바꿔서 민감도를 조정할 수 있다.
# ------------------------------------------------------------

# ------------------------------------------------------------
# 사용자 설정: 입력 이미지 / 저장 파일 / 창 이름
# ------------------------------------------------------------
NORMAL_IMAGE_PATH = "re_lcd_std_1_color.png"
ABNORMAL_IMAGE_PATH = "re_lcd_std_2_color.png"
SAVE_PATH = "vertical_icon_similarity_result.png"
WINDOW_NAME = "Vertical LCD Icon Similarity"

# 상단 제목/정보 영역 높이
TITLE_H = 130

# MANUAL 모드에서 선택한 ROI의 발광영역 추출 과정을 저장할 파일
MANUAL_STEPS_SAVE_PATH = "manual_selected_glow_steps.png"

# ------------------------------------------------------------
# similarity 계산 기준값
# ------------------------------------------------------------
# Hue는 OpenCV HSV에서 원형 거리의 최대값이 90이므로,
# 보통 90을 기준으로 0~100%로 환산한다.
# 더 엄격하게 보고 싶으면 60 또는 45 등으로 줄이면 된다.
HUE_SIMILARITY_BASE = 20.0

# Brightness(V)는 0~255 범위이므로 보통 255를 기준으로 환산한다.
BRIGHT_SIMILARITY_BASE = 255.0

# ------------------------------------------------------------
# 발광 영역 추출 / 고채도-저채도 판단 기준값
# ------------------------------------------------------------
# 평균 Saturation이 이 값보다 크면 고채도라고 판단 -> Hue 비교 사용
S_THRESHOLD = 60

# 컬러 발광부 추출용 기준
COLOR_S_THRESHOLD = 40
COLOR_V_THRESHOLD = 40

# 흰색/회색 밝은 발광부 추출용 기준
WHITE_V_THRESHOLD = 150

# ------------------------------------------------------------
# 세로 이미지(377x640) 기준 ROI
# 너무 타이트하지 않게 설정된 고정 박스
# ------------------------------------------------------------
ICON_ROIS = {
    "Bulb": (4, 6, 46, 48),
    "Mode": (76, 8, 56, 46),
    "Plug": (165, 10, 48, 46),
    "Menu": (242, 10, 48, 46),
    "Gear": (314, 12, 48, 46),
    "Big1": (115, 107, 150, 150),
    "Big2": (115, 272, 150, 150),
    "Big3": (115, 437, 150, 150),
}

# ------------------------------------------------------------
# 전역 상태 변수
# ------------------------------------------------------------
normal_img = None
abnormal_img = None

# 마우스로 ROI를 그릴 때 사용하는 상태값
# drawing: 현재 드래그 중인지 여부
# start_pt, end_pt: 마우스 시작점/끝점
# selected_roi: 최종 선택된 수동 ROI
# view_mode: AUTO / MANUAL 모드 구분
# auto_results: AUTO 모드 아이콘별 계산 결과

drawing = False
start_pt = None
end_pt = None
selected_roi = None

view_mode = "AUTO"
auto_results = {}

# 최근 MANUAL ROI 계산 결과를 화면 상단에 표시하기 위한 값들
last_similarity = None
last_mode = None
last_normal_value = None
last_abnormal_value = None
last_diff = None
last_normal_mean_s = None
last_abnormal_mean_s = None


# ------------------------------------------------------------
# 기본 수학 함수
# ------------------------------------------------------------
def circular_hue_distance(h1, h2):
    """
    Hue는 직선 값이 아니라 원형 값이다.
    따라서 |h1-h2| 만 쓰면 안 되고,
    원 둘레에서 더 짧은 쪽 거리를 써야 한다.

    예)
    h1 = 179, h2 = 1 이면
    단순 차이는 178이지만,
    실제 색상환에서는 거의 붙어 있으므로 원형 거리는 2이다.
    """
    diff = abs(float(h1) - float(h2))
    return min(diff, 180.0 - diff)



def circular_mean_hue(h_values):
    """
    Hue 평균은 일반 평균이 아니라 원형 평균으로 구한다.
    단순 평균을 쓰면 179와 1의 평균이 90이 되어 버리는데,
    실제로는 둘 다 거의 같은 색이므로 잘못된 결과가 된다.
    """
    if h_values.size == 0:
        return 0.0

    # Hue(0~179)를 원형 각도(0~2pi)로 변환
    angles = h_values.astype(np.float32) * 2.0 * np.pi / 180.0

    # sin, cos 평균으로 대표 방향 계산
    mean_sin = np.mean(np.sin(angles))
    mean_cos = np.mean(np.cos(angles))
    mean_angle = np.arctan2(mean_sin, mean_cos)

    # 각도가 음수로 나오면 0~2pi 범위로 보정
    if mean_angle < 0:
        mean_angle += 2.0 * np.pi

    # 다시 OpenCV Hue 범위(0~179 근처)로 환산
    return float(mean_angle * 180.0 / (2.0 * np.pi))


# ------------------------------------------------------------
# 전처리 / 마스크 생성 함수
# ------------------------------------------------------------
def preprocess(img_bgr):
    """
    너무 강하지 않은 Gaussian blur를 적용한다.
    목적:
    - 작은 노이즈 완화
    - 아주 미세한 반사/점 잡음 완화

    너무 강하게 blur를 주면 아이콘 경계까지 흐려지므로
    커널을 작게 유지한다.
    """
    return cv2.GaussianBlur(img_bgr, (3, 3), 0)



def get_glow_mask_from_roi(img_bgr, roi):
    """
    ROI 내부에서 '실제로 빛나고 있는 부분'만 간단히 분리한다.

    아이디어:
    1) 컬러 발광부: S와 V가 둘 다 어느 정도 큰 영역
    2) 흰색/회색 밝은 발광부: V가 큰 영역
    3) 두 마스크를 OR로 합친다.
    4) morphology로 자잘한 점 노이즈를 정리한다.

    반환값:
    - mask (0 또는 255)
    """
    x, y, w, h = roi
    roi_bgr = img_bgr[y:y + h, x:x + w]

    # ROI만 잘라서 약하게 전처리
    filtered = preprocess(roi_bgr)

    # BGR -> HSV 변환
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    # 채도도 있고 밝기도 있는 컬러 발광부
    mask_color = ((s_ch > COLOR_S_THRESHOLD) & (v_ch > COLOR_V_THRESHOLD)).astype(np.uint8) * 255

    # 채도는 낮더라도 아주 밝은 흰색/회색 발광부
    mask_white = (v_ch > WHITE_V_THRESHOLD).astype(np.uint8) * 255

    # 두 마스크 합치기
    mask = cv2.bitwise_or(mask_color, mask_white)

    # morphology로 작은 점 노이즈 제거 + 끊긴 부분 조금 정리
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask



def make_mask_overlay_image(roi_bgr, mask):
    """
    발광영역 마스크를 원본 ROI 위에 겹쳐서 보여주는 시각화 함수
    - 발광으로 판단된 부분만 초록색으로 덧씌운다.
    """
    overlay = roi_bgr.copy()
    if mask is None:
        return overlay

    green = np.zeros_like(overlay)
    green[:, :] = (0, 255, 0)
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    overlay = np.where(mask3 > 0, cv2.addWeighted(overlay, 0.65, green, 0.35, 0), overlay)
    return overlay


def resize_for_panel(img, cell_w=170, cell_h=120, is_mask=False):
    """
    패널에 넣기 쉽게 이미지를 일정 크기로 맞춘다.
    - 일반 영상은 INTER_AREA
    - 마스크는 INTER_NEAREST 로 확대
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    return cv2.resize(img, (cell_w, cell_h), interpolation=interp)


def save_manual_glow_steps_panel(roi):
    """
    MANUAL 모드에서 선택한 ROI 하나에 대해 발광영역 추출 과정을 패널로 저장한다.

    패널 구성:
    - 열: Original ROI / Gaussian Blur / Glow Mask / Mask Overlay
    - 행: Normal / Abnormal
    """
    x, y, w, h = roi

    # 정상 / 비정상 ROI 잘라내기
    n_roi = normal_img[y:y + h, x:x + w].copy()
    a_roi = abnormal_img[y:y + h, x:x + w].copy()

    # 전처리 이미지
    n_blur = preprocess(n_roi)
    a_blur = preprocess(a_roi)

    # 마스크 생성
    n_mask = get_glow_mask_from_roi(normal_img, roi)
    a_mask = get_glow_mask_from_roi(abnormal_img, roi)

    # 오버레이 이미지 생성
    n_overlay = make_mask_overlay_image(n_roi, n_mask)
    a_overlay = make_mask_overlay_image(a_roi, a_mask)

    # 패널용 크기로 조정
    cell_w, cell_h = 170, 120
    n_roi_r = resize_for_panel(n_roi, cell_w, cell_h)
    n_blur_r = resize_for_panel(n_blur, cell_w, cell_h)
    n_mask_r = resize_for_panel(n_mask, cell_w, cell_h, is_mask=True)
    n_overlay_r = resize_for_panel(n_overlay, cell_w, cell_h)

    a_roi_r = resize_for_panel(a_roi, cell_w, cell_h)
    a_blur_r = resize_for_panel(a_blur, cell_w, cell_h)
    a_mask_r = resize_for_panel(a_mask, cell_w, cell_h, is_mask=True)
    a_overlay_r = resize_for_panel(a_overlay, cell_w, cell_h)

    # 패널 캔버스 생성
    margin = 20
    left_label_w = 90
    top_h = 55
    title_h = 40
    panel_w = left_label_w + margin * 2 + cell_w * 4 + 15 * 3
    panel_h = title_h + top_h + margin * 3 + cell_h * 2 + 20
    panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)

    # 제목
    cv2.putText(panel, 'Glow Extraction Steps (Manual ROI)', (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(panel, f'ROI = (x={x}, y={y}, w={w}, h={h})', (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

    # 열 제목
    headers = ['Original ROI', 'Gaussian Blur', 'Glow Mask', 'Mask Overlay']
    x0 = left_label_w + margin
    y0 = title_h + top_h
    for i, head in enumerate(headers):
        hx = x0 + i * (cell_w + 15)
        cv2.putText(panel, head, (hx + 10, title_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    # 행 이름
    cv2.putText(panel, 'Normal', (20, y0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Abnormal', (20, y0 + cell_h + margin + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 90, 180), 2, cv2.LINE_AA)

    # 이미지 배치 함수
    def paste(img, px, py):
        panel[py:py + cell_h, px:px + cell_w] = img
        cv2.rectangle(panel, (px, py), (px + cell_w, py + cell_h), (200, 200, 200), 1)

    # 첫째 행: Normal
    imgs1 = [n_roi_r, n_blur_r, n_mask_r, n_overlay_r]
    # 둘째 행: Abnormal
    imgs2 = [a_roi_r, a_blur_r, a_mask_r, a_overlay_r]

    for i, img in enumerate(imgs1):
        px = x0 + i * (cell_w + 15)
        paste(img, px, y0)

    for i, img in enumerate(imgs2):
        px = x0 + i * (cell_w + 15)
        paste(img, px, y0 + cell_h + margin)

    cv2.imwrite(MANUAL_STEPS_SAVE_PATH, panel)
    print(f'발광영역 추출 패널 저장 완료: {MANUAL_STEPS_SAVE_PATH}')


def get_feature_from_roi(img_bgr, roi):
    """
    ROI에서 발광 영역만 분리한 뒤,
    similarity 계산에 필요한 대표 feature를 구한다.

    반환값:
    - mean_hue : 발광 영역 평균 Hue
    - mean_v   : 발광 영역 평균 Brightness(Value)
    - mean_s   : 발광 영역 평균 Saturation
    """
    x, y, w, h = roi
    roi_bgr = img_bgr[y:y + h, x:x + w]

    # 전처리 후 HSV 변환
    filtered = preprocess(roi_bgr)
    hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    h_ch = hsv[:, :, 0].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2].astype(np.float32)

    # 발광 영역 마스크 생성
    mask = get_glow_mask_from_roi(img_bgr, roi)
    valid = mask > 0
    count = int(np.count_nonzero(valid))

    # 발광 픽셀이 너무 적으면,
    # 계산이 불안정해질 수 있으므로 ROI 전체를 fallback으로 사용한다.
    if count < 10:
        valid = np.ones(h_ch.shape, dtype=bool)

    mean_s = float(np.mean(s_ch[valid]))
    mean_v = float(np.mean(v_ch[valid]))
    mean_hue = circular_mean_hue(h_ch[valid])

    return mean_hue, mean_v, mean_s


# ------------------------------------------------------------
# similarity 계산 함수
# ------------------------------------------------------------
def similarity_from_distance(diff, base_value):
    """
    거리(diff)를 0~100% similarity로 바꾼다.

    diff = 0          -> 100%
    diff = base_value -> 0%

    예)
    Hue 비교에서 base_value = 90
    Brightness 비교에서 base_value = 255
    """
    sim = max(0.0, 100.0 * (1.0 - float(diff) / float(base_value)))
    return float(sim)



def hue_similarity_percent(h1, h2):
    """
    두 Hue 값의 원형 거리로 similarity를 계산한다.
    기준값은 HUE_SIMILARITY_BASE 상수를 사용한다.
    """
    dist = circular_hue_distance(h1, h2)
    return similarity_from_distance(dist, HUE_SIMILARITY_BASE), dist



def brightness_similarity_percent(v1, v2):
    """
    두 Brightness(Value) 값의 차이로 similarity를 계산한다.
    기준값은 BRIGHT_SIMILARITY_BASE 상수를 사용한다.
    """
    diff = abs(float(v1) - float(v2))
    return similarity_from_distance(diff, BRIGHT_SIMILARITY_BASE), diff


# ------------------------------------------------------------
# 핵심 비교 함수
# ------------------------------------------------------------
def compare_one_roi(roi):
    """
    하나의 ROI에 대해 정상/비정상 similarity를 계산한다.

    처리 순서:
    1) 정상 ROI feature 계산
    2) 비정상 ROI feature 계산
    3) 평균 S로 고채도 / 저채도 판단
    4) 고채도이면 Hue 비교
    5) 저채도이면 Brightness(V) 비교
    """
    global last_similarity, last_mode, last_normal_value, last_abnormal_value
    global last_diff, last_normal_mean_s, last_abnormal_mean_s

    # 정상/비정상에서 각각 feature 계산
    n_h, n_v, n_s = get_feature_from_roi(normal_img, roi)
    a_h, a_v, a_s = get_feature_from_roi(abnormal_img, roi)

    # 두 영상의 평균 S를 같이 보고 이번 ROI가 고채도인지 판단
    mean_s = (n_s + a_s) / 2.0

    if mean_s > S_THRESHOLD:
        # 고채도 -> Hue 비교
        mode = "HUE"
        similarity, diff = hue_similarity_percent(n_h, a_h)
        normal_value = n_h
        abnormal_value = a_h
    else:
        # 저채도 -> Brightness(V) 비교
        mode = "BRIGHT"
        similarity, diff = brightness_similarity_percent(n_v, a_v)
        normal_value = n_v
        abnormal_value = a_v

    # 결과를 딕셔너리로 정리
    result = {
        "mode": mode,
        "similarity_percent": similarity,
        "normal_value": normal_value,
        "abnormal_value": abnormal_value,
        "diff": diff,
        "normal_mean_s": n_s,
        "abnormal_mean_s": a_s,
    }

    # MANUAL 모드 표시용 최근 결과 저장
    last_similarity = similarity
    last_mode = mode
    last_normal_value = normal_value
    last_abnormal_value = abnormal_value
    last_diff = diff
    last_normal_mean_s = n_s
    last_abnormal_mean_s = a_s

    return result



def compute_auto_results():
    """
    AUTO 모드에서 미리 정한 모든 아이콘 ROI를 한 번에 계산한다.
    """
    results = {}
    for name, roi in ICON_ROIS.items():
        results[name] = compare_one_roi(roi)
    return results


# ------------------------------------------------------------
# 화면 구성 함수
# ------------------------------------------------------------
def build_canvas():
    """
    상단 제목 영역 + 좌우 원본 이미지를 붙인 화면을 만든다.
    """
    h, w = normal_img.shape[:2]
    board = np.zeros((h + TITLE_H, w * 2, 3), dtype=np.uint8)
    board[:] = (25, 25, 25)

    # 좌우 이미지 배치
    board[TITLE_H:TITLE_H + h, 0:w] = normal_img
    board[TITLE_H:TITLE_H + h, w:w + w] = abnormal_img

    # 제목만 단순하게 표시
    cv2.putText(board, "NORMAL", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(board, "ABNORMAL", (w + 20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)
    return board



def draw_auto_boxes(board):
    """
    AUTO 모드에서 고정 ROI 박스와 similarity 텍스트를 그린다.

    규칙:
    - 왼쪽 NORMAL 쪽은 글자 없이 박스만 표시
    - 오른쪽 ABNORMAL 쪽에만 아이콘 이름 + similarity 표시
    - Mode/Menu는 박스 위
    - Bulb/Plug/Gear는 박스 아래
    - Big1~3은 큰 박스의 오른쪽 위에서 시작
    """
    h, w = normal_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1

    for name, roi in ICON_ROIS.items():
        x, y, rw, rh = roi

        # 왼쪽 NORMAL: 글자 없이 박스만
        cv2.rectangle(board, (x, y + TITLE_H), (x + rw, y + rh + TITLE_H), (0, 0, 255), 2)

        # 오른쪽 ABNORMAL: 박스 + 글자
        cv2.rectangle(board, (w + x, y + TITLE_H), (w + x + rw, y + rh + TITLE_H), (0, 0, 255), 2)

        sim = auto_results[name]["similarity_percent"]
        text = f"{name} {sim:.1f}%"

        # 텍스트 위치 결정
        if name in ["Mode", "Menu"]:
            text_x = w + x
            text_y = max(15, y + TITLE_H - 6)
        elif name in ["Bulb", "Plug", "Gear"]:
            text_x = w + x
            text_y = y + TITLE_H + rh + 16
        else:
            text_x = w + x + rw
            text_y = max(15, y + TITLE_H - 6)

        cv2.putText(board, text, (text_x, text_y), font, scale, (0, 0, 255), thickness, cv2.LINE_AA)

    return board



def draw_manual_result(board):
    """
    MANUAL 모드에서 사용자가 지정한 ROI와 결과 정보를 그린다.

    표시 규칙:
    - 왼쪽 NORMAL: 글자 없이 박스만
    - 오른쪽 ABNORMAL: 박스 + similarity %
    - 상단 아이콘 5개 영역처럼 y가 작은 경우 %를 박스 아래에 표시
    - 그 외에는 박스 위에 표시
    - 상단 설명 텍스트는 서로 겹치지 않도록 줄을 나눠서 표시
    """
    h, w = normal_img.shape[:2]

    # ROI 박스와 박스 근처의 similarity 표시
    if selected_roi is not None:
        x, y, rw, rh = selected_roi

        # 왼쪽 NORMAL: 박스만
        cv2.rectangle(board, (x, y + TITLE_H), (x + rw, y + rh + TITLE_H), (0, 0, 255), 2)

        # 오른쪽 ABNORMAL: 박스 + similarity 표시
        cv2.rectangle(board, (w + x, y + TITLE_H), (w + x + rw, y + rh + TITLE_H), (0, 0, 255), 2)

        if last_similarity is not None:
            text = f"{last_similarity:.2f}%"
            tx = w + x

            # 상단 작은 아이콘 영역이면 박스 아래에 % 표시
            if y < 70:
                ty = y + TITLE_H + rh + 22
            else:
                ty = max(TITLE_H + 18, y + TITLE_H - 8)

            cv2.putText(board, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2, cv2.LINE_AA)

    # 상단 정보 표시
    if last_similarity is not None:
        line1 = f"Mode: {last_mode}   Similarity: {last_similarity:.2f}%"
        line2 = f"Normal S: {last_normal_mean_s:.2f} | Abnormal S: {last_abnormal_mean_s:.2f}"
        line3 = f"Normal value: {last_normal_value:.2f} | Abnormal value: {last_abnormal_value:.2f} | Diff: {last_diff:.2f}"

        cv2.putText(board, line1, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(board, line2, (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(board, line3, (20, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)

    return board


# ------------------------------------------------------------
# ROI 보정 / 마우스 처리 함수
# ------------------------------------------------------------
def clamp_roi(x1, y1, x2, y2, width, height):
    """
    마우스로 드래그한 좌표를 이미지 범위 안의 올바른 ROI로 바꾼다.

    반환값:
    - (x, y, w, h)
    - 너무 작은 ROI이면 None
    """
    x_min = max(0, min(x1, x2))
    y_min = max(0, min(y1, y2))
    x_max = min(width - 1, max(x1, x2))
    y_max = min(height - 1, max(y1, y2))

    rw = x_max - x_min
    rh = y_max - y_min

    if rw < 2 or rh < 2:
        return None

    return x_min, y_min, rw, rh



def mouse_callback(event, x, y, flags, param):
    """
    MANUAL 모드에서만 동작하는 마우스 콜백 함수.
    오른쪽 ABNORMAL 이미지에서 ROI를 드래그하면,
    같은 좌표의 왼쪽 NORMAL ROI와 similarity를 계산한다.
    """
    global drawing, start_pt, end_pt, selected_roi

    # AUTO 모드에서는 마우스 입력 무시
    if view_mode != "MANUAL":
        return

    h, w = normal_img.shape[:2]

    # 오른쪽 이미지 내부인지 체크
    in_right_img = (w <= x < 2 * w) and (TITLE_H <= y < TITLE_H + h)

    if event == cv2.EVENT_LBUTTONDOWN and in_right_img:
        drawing = True
        start_pt = (x - w, y - TITLE_H)
        end_pt = start_pt

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_pt = (min(max(x - w, 0), w - 1), min(max(y - TITLE_H, 0), h - 1))

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end_pt = (min(max(x - w, 0), w - 1), min(max(y - TITLE_H, 0), h - 1))

        roi = clamp_roi(start_pt[0], start_pt[1], end_pt[0], end_pt[1], w, h)
        if roi is not None:
            selected_roi = roi
            compare_one_roi(selected_roi)
            save_manual_glow_steps_panel(selected_roi)


# ------------------------------------------------------------
# 메인 루프
# ------------------------------------------------------------
def main():
    """
    프로그램 시작 함수.

    키 조작:
    - m : MANUAL 모드
    - a : AUTO 모드
    - r : MANUAL ROI 초기화
    - s : 현재 화면 저장
    - ESC : 종료
    """
    global normal_img, abnormal_img, auto_results, view_mode, selected_roi
    global last_similarity, last_mode, last_normal_value, last_abnormal_value
    global last_diff, last_normal_mean_s, last_abnormal_mean_s

    # 이미지 읽기
    normal_img = cv2.imread(NORMAL_IMAGE_PATH)
    abnormal_img = cv2.imread(ABNORMAL_IMAGE_PATH)

    if normal_img is None:
        raise FileNotFoundError(f"정상 이미지를 읽을 수 없습니다: {NORMAL_IMAGE_PATH}")
    if abnormal_img is None:
        raise FileNotFoundError(f"비정상 이미지를 읽을 수 없습니다: {ABNORMAL_IMAGE_PATH}")

    # AUTO 모드 기본 결과 미리 계산
    auto_results = compute_auto_results()

    # 창 생성 및 마우스 콜백 연결
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    while True:
        board = build_canvas()

        # 현재 모드에 따라 화면 구성
        if view_mode == "AUTO":
            board = draw_auto_boxes(board)
        else:
            board = draw_manual_result(board)

            # 드래그 중이면 실시간으로 빨간 박스 미리보기 표시
            if drawing and start_pt is not None and end_pt is not None:
                h, w = normal_img.shape[:2]
                roi_preview = clamp_roi(start_pt[0], start_pt[1], end_pt[0], end_pt[1], w, h)
                if roi_preview is not None:
                    px, py, pw, ph = roi_preview
                    cv2.rectangle(board, (px, py + TITLE_H), (px + pw, py + ph + TITLE_H), (0, 0, 255), 2)
                    cv2.rectangle(board, (w + px, py + TITLE_H), (w + px + pw, py + ph + TITLE_H), (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, board)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('m'):
            # MANUAL 모드 진입
            view_mode = "MANUAL"
        elif key == ord('a'):
            # AUTO 모드 복귀
            view_mode = "AUTO"
            selected_roi = None
        elif key == ord('r'):
            # MANUAL ROI 및 최근 결과 초기화
            selected_roi = None
            last_similarity = None
            last_mode = None
            last_normal_value = None
            last_abnormal_value = None
            last_diff = None
            last_normal_mean_s = None
            last_abnormal_mean_s = None
        elif key == ord('s'):
            # 현재 보이는 화면 저장
            cv2.imwrite(SAVE_PATH, board)
            print(f"결과 이미지를 저장했습니다: {SAVE_PATH}")

    cv2.destroyAllWindows()


# ------------------------------------------------------------
# 프로그램 시작점
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
