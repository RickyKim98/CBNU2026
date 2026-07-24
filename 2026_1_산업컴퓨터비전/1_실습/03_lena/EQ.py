import argparse
import cv2, numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='Lenna.png', help='Image path')
params = parser.parse_args()


# Original Image
img = cv2.imread(params.path)
cv2.imshow('origianl image', img)


# Gray scale - before EQ
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray scale image', gray)
hist, bins = np.histogram(gray, 256, [0, 256])
plt.fill(hist)
plt.xlabel('before eq pixel value')
plt.show()

# Gray scale - after EQ
gray_eq = cv2.equalizeHist(gray)
hist, bins = np.histogram(gray_eq, 256, [0, 256])
plt.fill_between(range(256), hist, 0)
plt.xlabel('after eq pixel value')
plt.show()
cv2.imshow('equalized image', gray_eq)

# Color EQ
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[..., 2] = cv2.equalizeHist(hsv[...,2])
color_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow('equalized color', color_eq)

cv2.waitKey()
cv2.destroyAllWindows()