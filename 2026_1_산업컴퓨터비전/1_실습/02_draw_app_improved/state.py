import numpy as np


class AppState:
    def __init__(self, img, image_path=''):
        self.img = img
        self.image_to_show = np.copy(img)
        self.image_path = image_path

        self.mouse_pressed = False
        self.s_x = -1
        self.s_y = -1
        self.e_x = -1
        self.e_y = -1

        # Default mode: rectangle
        self.keyval = ord('r')
