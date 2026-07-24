import numpy as np
import cv2

from drawing_tools import draw_preview, LINE_MODE


def mouse_callback(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_pressed = True
        state.s_x, state.s_y = x, y
        state.e_x, state.e_y = x, y
        state.image_to_show = np.copy(state.img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.mouse_pressed:
            state.e_x, state.e_y = x, y
            state.image_to_show = np.copy(state.img)
            draw_preview(
                state.image_to_show,
                state.keyval,
                (state.s_x, state.s_y),
                (x, y),
            )

    elif event == cv2.EVENT_LBUTTONUP:
        state.mouse_pressed = False
        state.e_x, state.e_y = x, y

        if state.keyval == LINE_MODE:
            # Commit final line to source image
            draw_preview(
                state.img,
                state.keyval,
                (state.s_x, state.s_y),
                (state.e_x, state.e_y),
            )
            state.image_to_show = np.copy(state.img)
