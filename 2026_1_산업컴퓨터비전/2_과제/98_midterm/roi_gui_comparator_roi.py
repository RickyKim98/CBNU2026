#==============================================================================================
# - Information                                                                                
#==============================================================================================
#                                                         Copyright(c) 2026 by MK, KIM		
# Originator       : Myungki, Kim                                                       
# Project Name     : ROI-Based LCD GUI Comparator                                   
# Last Update      : 2026. 04. 21                                                                                                                                        
# Created on       : 2026. 04. 21                                                              
#==============================================================================================


import cv2
import numpy as np

# ------------------------------------------------------------
# 파일 패스
# ------------------------------------------------------------
NORMAL_IMAGE_PATH   = "re_lcd_std_1_color.png"
ABNORMAL_IMAGE_PATH = "re_lcd_std_2_color.png"
SAVE_PATH           = "vertical_icon_similarity_result.png"
WINDOW_NAME         = "ROI based icon similarity test program"

TITLE_H = 130       # 상단 제목/정보 영역 높이

# ------------------------------------------------------------
# similarity 계산 기준값
# ------------------------------------------------------------
# 90:느슨한 비교, 값이 작아 질 수록 엄격해짐
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
# 각 아이콘 ROI 좌표
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
normal_img      = None
abnormal_img    = None

drawing         = False
start_pt        = None
end_pt          = None
selected_roi    = None

view_mode       = "MANUAL"
auto_results    = {}

# 최근 MANUAL ROI 계산 결과를 화면 상단에 표시하기 위한 값들
last_similarity     = None
last_mode           = None
last_normal_value   = None
last_abnormal_value = None
last_diff           = None
last_normal_mean_s  = None
last_abnormal_mean_s = None


# ------------------------------------------------------------
# 기본 수학 함수
# Hue의 직선을 원형으로 값을 변환하는 함수(opencv hue는 0~179라고 함)
# ------------------------------------------------------------
def circular_hue_distance(h1, h2):
    diff = abs(float(h1) - float(h2))
    return min(diff, 180.0 - diff)

# ------------------------------------------------------------
# 기본 수학 함수
# Hue의 원형 평균 함수
# ------------------------------------------------------------
def circular_mean_hue(h_values):
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
# Gaussian Blur
def preprocess(img_bgr):
    return cv2.GaussianBlur(img_bgr, (3, 3), 0)

# 발광 영역 마스크 생성
def get_glow_mask_from_roi(img_bgr, roi):
    x, y, w, h = roi
    roi_bgr = img_bgr[y:y + h, x:x + w]

    # ROI만 잘라서 약하게 가우시안 블러
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


# 비교할 값 (평균 H, S, V 값) 구해라
def get_feature_from_roi(img_bgr, roi):
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
    sim = max(0.0, 100.0 * (1.0 - float(diff) / float(base_value)))
    return float(sim)


def hue_similarity_percent(h1, h2):
    dist = circular_hue_distance(h1, h2)
    return similarity_from_distance(dist, HUE_SIMILARITY_BASE), dist


def brightness_similarity_percent(v1, v2):
    diff = abs(float(v1) - float(v2))
    return similarity_from_distance(diff, BRIGHT_SIMILARITY_BASE), diff


# ------------------------------------------------------------
# 비교 함수
# ------------------------------------------------------------
def compare_one_roi(roi):
    global last_similarity, last_mode, last_normal_value, last_abnormal_value
    global last_diff, last_normal_mean_s, last_abnormal_mean_s

    # 1) 정상/비정상에서 각각 feature 계산
    n_h, n_v, n_s = get_feature_from_roi(normal_img, roi)
    a_h, a_v, a_s = get_feature_from_roi(abnormal_img, roi)

    # 2) 두 이미지의 평균 S를 같이 보고 이번 ROI가 고채도인지 판단
    mean_s = (n_s + a_s) / 2.0

    # 3) 고채도는 Hue 비교
    if mean_s > S_THRESHOLD:

        mode = "HUE"
        similarity, diff = hue_similarity_percent(n_h, a_h)
        normal_value = n_h
        abnormal_value = a_h
    # 4) 저채도는 Brightness(V) 비교
    else:

        mode = "BRIGHT"
        similarity, diff = brightness_similarity_percent(n_v, a_v)
        normal_value = n_v
        abnormal_value = a_v

    # 5) 결과 정리
    result = {
        "mode": mode,
        "similarity_percent": similarity,
        "normal_value": normal_value,
        "abnormal_value": abnormal_value,
        "diff": diff,
        "normal_mean_s": n_s,
        "abnormal_mean_s": a_s,
    }

    # 6) MANUAL 모드 표시용 최근 결과 저장
    last_similarity = similarity
    last_mode = mode
    last_normal_value = normal_value
    last_abnormal_value = abnormal_value
    last_diff = diff
    last_normal_mean_s = n_s
    last_abnormal_mean_s = a_s

    return result



def compute_auto_results():
    results = {}
    for name, roi in ICON_ROIS.items():
        results[name] = compare_one_roi(roi)
    return results


# ------------------------------------------------------------
# 화면 구성 함수
# ------------------------------------------------------------
def build_canvas():
    h, w = normal_img.shape[:2]
    board = np.zeros((h + TITLE_H, w * 2, 3), dtype=np.uint8)
    board[:] = (25, 25, 25)

    # 좌우 이미지 로딩해라
    board[TITLE_H:TITLE_H + h, 0:w] = normal_img
    board[TITLE_H:TITLE_H + h, w:w + w] = abnormal_img

    # 제목 표시
    cv2.putText(board, "NORMAL", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(board, "ABNORMAL", (w + 20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)
    return board



def draw_auto_boxes(board):
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
    x_min = max(0, min(x1, x2))
    y_min = max(0, min(y1, y2))
    x_max = min(width - 1, max(x1, x2))
    y_max = min(height - 1, max(y1, y2))

    rw = x_max - x_min
    rh = y_max - y_min

    if rw < 2 or rh < 2:
        return None

    return x_min, y_min, rw, rh


# ------------------------------------------------------------
# 마우스 콜백 함수
# ------------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
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


# ------------------------------------------------------------
# 메인
# 키보드 m:매뉴얼, a:자동, r:리셋, s:화면저장, ESC:종료
# ------------------------------------------------------------
def main():
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
