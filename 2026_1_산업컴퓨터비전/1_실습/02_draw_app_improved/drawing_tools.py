import cv2


RECTANGLE_MODE = ord('r')
LINE_MODE = ord('l')


def draw_preview(image, mode, start_pt, end_pt, color=(0, 255, 0), thickness=1):
    if mode == RECTANGLE_MODE:
        cv2.rectangle(image, start_pt, end_pt, color, thickness)
    elif mode == LINE_MODE:
        cv2.line(image, start_pt, end_pt, color, thickness)
