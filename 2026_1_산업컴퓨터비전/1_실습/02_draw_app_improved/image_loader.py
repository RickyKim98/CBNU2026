import os
import cv2


def load_image(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f'Image file not found: {path}')

    img = cv2.imread(path)
    if img is None:
        raise ValueError(f'Failed to read image: {path}')

    return img
