import os
import cv2
import numpy as np

from state import AppState
from mouse_handler import mouse_callback
from drawing_tools import RECTANGLE_MODE, LINE_MODE


class DrawingApp:
    def __init__(self, img, image_path='', output_path='output.png'):
        self.state = AppState(img, image_path=image_path)
        self.window_name = 'image'
        self.output_path = output_path

    def setup(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, mouse_callback, self.state)

    def normalize_selection(self):
        s_x, e_x = self.state.s_x, self.state.e_x
        s_y, e_y = self.state.s_y, self.state.e_y

        if s_x > e_x:
            s_x, e_x = e_x, s_x
        if s_y > e_y:
            s_y, e_y = e_y, s_y

        return s_x, s_y, e_x, e_y

    def crop_image(self):
        # Crop is only allowed in rectangle mode
        if self.state.keyval != RECTANGLE_MODE:
            print('[INFO] Crop is available only in rectangle mode (press r).')
            return

        s_x, s_y, e_x, e_y = self.normalize_selection()

        if e_y - s_y > 1 and e_x - s_x > 1:
            self.state.img = self.state.img[s_y:e_y, s_x:e_x]
            self.state.image_to_show = np.copy(self.state.img)
            self.state.s_x = self.state.s_y = self.state.e_x = self.state.e_y = -1
            print(f'[INFO] Cropped: x={s_x}:{e_x}, y={s_y}:{e_y}')
        else:
            print('[INFO] Invalid crop area.')

    def save_image(self):
        filename = self.output_path
        success = cv2.imwrite(filename, self.state.img)
        if success:
            print(f'[INFO] Saved image: {os.path.abspath(filename)}')
        else:
            print('[ERROR] Failed to save image.')

    def print_help(self):
        print(
            '[KEY GUIDE]\n'
            'r : rectangle mode\n'
            'l : line mode\n'
            'c : crop selected rectangle (rectangle mode only)\n'
            's : save current image\n'
            'h : show help\n'
            'ESC : exit'
        )

    def handle_key(self, k):
        if k == ord('c'):
            self.crop_image()
        elif k == LINE_MODE:
            self.state.keyval = k
            print('[INFO] Switched to line mode.')
        elif k == RECTANGLE_MODE:
            self.state.keyval = k
            print('[INFO] Switched to rectangle mode.')
        elif k == ord('s'):
            self.save_image()
        elif k == ord('h'):
            self.print_help()

    def run(self):
        self.setup()
        self.print_help()

        while True:
            cv2.imshow(self.window_name, self.state.image_to_show)
            k = cv2.waitKey(1) & 0xFF

            if k == 27:
                break

            if k != 255:
                self.handle_key(k)

        cv2.destroyAllWindows()
