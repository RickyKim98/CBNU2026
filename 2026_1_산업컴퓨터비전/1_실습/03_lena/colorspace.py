import argparse
import cv2, numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='Lenna.png', help='Image path')
params = parser.parse_args()
img = cv2.imread(params.path)

print('Shape', img.shape)
print('Data type', img.dtype)
cv2.imshow('original image', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print('Converted to grayscale')
print('Shape', gray.shape)
print('Data type', gray.dtype)
cv2.imshow('gray-scale image', gray)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print('Converted to HSV')
print('Shape', hsv.shape)
print('Data type', hsv.dtype)
cv2.imshow('hsv-scale image', hsv)

# Value up x2
hsv[:, :, 2] *=2
from_hsv = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shape', from_hsv.shape)
print('Data type', from_hsv.dtype)
cv2.imshow('from_hsv', from_hsv)

cv2.waitKey()
cv2.destroyAllWindows()