import argparse
import cv2, numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='Lenna.png', help='Image path')
params = parser.parse_args()


# Original Image
img = cv2.imread(params.path).astype(np.float32) / 255
cv2.imshow('origianl image', img)

# Noise
noised = (img + 0.2 * np.random.rand(*img.shape).astype(np.float32))
noised = noised.clip(0, 1)
plt.imshow(noised[:, :, [2, 1, 0]])
plt.xlabel('noised image')
plt.show()

# GaussianBlur filter
gauss_blur = cv2.GaussianBlur(noised, (7, 7), 0)
plt.imshow(gauss_blur[:, :, [2, 1, 0]])
plt.xlabel('GaussianBlur image')
plt.show()

# Median filter
median_blur = cv2.medianBlur((noised * 255).astype(np.uint8), 7)
plt.imshow(median_blur[:, :, [2, 1, 0]])
plt.xlabel('Median image')
plt.show()

# Bilateral filter
bilat = cv2.bilateralFilter(noised, -1, 0.3, 10)
plt.imshow(bilat[:, :, [2, 1, 0]])
plt.xlabel('Bilateral image')
plt.show()


# Change hsv
# 1. get hsv image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#print('Converted to HSV')
#print('Shape', hsv.shape)
#print('Data type', hsv.dtype)
#cv2.imshow('hsv-scale image', hsv)

# 2. extract h value and apply median filter
h = hsv[:, :, 0]
h_blur = cv2.medianBlur((h * 255).astype(np.uint8), 7)
hsv[:, :, 0] = h_blur
plt.imshow(median_blur[:, :, [2, 1, 0]])
plt.xlabel('h channel ')
plt.show()


cv2.waitKey()
cv2.destroyAllWindows()