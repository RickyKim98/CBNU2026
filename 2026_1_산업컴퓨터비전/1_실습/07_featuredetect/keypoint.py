import cv2
import numpy as np
import random

img = cv2.imread('scenetext01.jpg', cv2.IMREAD_COLOR)


fast = cv2.FastFeatureDetector_create(160, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
keypoints = fast.detect(img)

for kp in keypoints:
    kp.size = 100*random.random()
    kp.angle = 360*random.random()

matches = []

for i in range(len(keypoints)):
    matches.append(cv2.DMatch(i, i, 1))

show_img = cv2.drawKeypoints(img, keypoints, None, (255, 0, 255))
cv2.imshow('keypoints', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

show_img = cv2.drawKeypoints(img, keypoints, None, (0, 255, 0),
                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('keypoints', show_img)
cv2.waitKey()
cv2.destroyAllWindows()

show_img = cv2.drawMatches(img, keypoints, img, keypoints, matches, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Maches', show_img)
cv2.waitKey()
cv2.destroyAllWindows()