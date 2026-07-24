import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

img = np.full((700, 700), 255, np.uint8)

img[100:300, 100:300] = aruco.drawMarker(aruco_dict, 2, 200)
img[100:300, 400:600] = aruco.drawMarker(aruco_dict, 76, 200)
img[400:600, 100:300] = aruco.drawMarker(aruco_dict, 42, 200)
img[400:600, 400:600] = aruco.drawMarker(aruco_dict, 123, 200)

img = cv2.GaussianBlur(img, (11, 11), 0)

cv2.imshow('Create AruCo makers', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

coners, ids, _ = aruco.detectMarkers(img, aruco_dict)

img_color = cv2.cvtColor(coners, cv2.COLOR_GRAY2BGR)
aruco.drawDetectedMarkers(img_color, coners, ids)

cv2.imshow('Detect AruCo makers', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

