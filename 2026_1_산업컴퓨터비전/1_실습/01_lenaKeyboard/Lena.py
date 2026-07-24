import argparse
import cv2, numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--path', default='Lenna.png', help='Image path')
params = parser.parse_args()
img = cv2.imread(params.path)
image_to_show = np.copy(img)

mouse_pressed = False
s_x = s_y = e_x = e_y = -1

def mouse_callback(event, x, y, flags, param):
    global image_to_show, s_x, s_y, e_x, e_y, mouse_pressed, keyval

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x,y
        image_to_show = np.copy(img)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_to_show = np.copy(img)

            if keyval == ord('r'):
                cv2.rectangle(image_to_show, (s_x, s_y), (x,y), (0,255,0), 1)

            elif keyval == ord('l'):
                cv2.line(image_to_show, (s_x, s_y), (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x,y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)



while True:
    cv2.imshow('image', image_to_show)
    k = cv2.waitKey(1)
    if k == 27:
        break

    if k == ord('c'):
        if s_y > e_y:
            s_y, e_y = e_y, s_y
        if s_x > e_x:
            s_x, e_x = e_x, s_x

            if e_y - s_y > 1 and e_x - s_x > 0:
                image = image[s_y:e_y, s_x:e_x]
                image_to_show = np.copy(img)

    elif k == ord('l'):
        keyval = k

    elif k == ord('r'):
        keyval = k


cv2.destroyAllWindows()

