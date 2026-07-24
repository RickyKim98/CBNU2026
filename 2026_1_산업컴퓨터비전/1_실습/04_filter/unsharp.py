#import argparse
import cv2, numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#parser = argparse.ArgumentParser()
#parser.add_argument('--path', default='Lenna.png', help='Image path')
#params = parser.parse_args()

# Original Image
img = cv2.imread('Lenna.png')

KSIZE = 11
ALPHA = 2

kernel = cv2.getGaussianKernel(KSIZE, 0)
kernel = -ALPHA * kernel @ kernel.T
kernel[KSIZE//2, KSIZE//2] += 1 + ALPHA
print(kernel.shape, kernel.dtype, kernel.sum())

filtered = cv2.filter2D(img, -1, kernel)

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('img')
plt.imshow(img[:, :, [2, 1, 0]])

plt.subplot(122)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered[:, :, [2, 1, 0]])
plt.tight_layout()
plt.show()

#cv2.imshow('brefore', img)
#cv2.imshow('Unsharp', filtered)

cv2.waitKey()
cv2.destroyAllWindows()