#import argparse
import cv2, numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math

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

'''
plt.figure(figsize=(8,4))
plt.subplot(121)
plt.axis('off')
plt.title('filtered')
#plt.imshow(filtered[:, :, [2, 1, 0]])
plt.tight_layout()
plt.show()
'''


#cv2.imshow('brefore', img)
#cv2.imshow('Unsharp', filtered)


# Sobel Filter
dx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(filtered, cv2.CV_32F, 0, 1)


plt.figure(figsize=(8,3))

plt.subplot(151)
plt.axis('off')
plt.title('original')
plt.imshow(img[:, :, [2, 1, 0]])

plt.subplot(152)
plt.axis('off')
plt.title('Unsharp')
plt.imshow(filtered[:, :, [2, 1, 0]])

plt.subplot(153)
plt.axis('off')
plt.title('image')
plt.imshow(filtered[:, :, [2, 1, 0]], cmap='gray')

plt.subplot(154)
plt.axis('off')
plt.imshow(dx, cmap='gray')
plt.title(r'$\frac{dI}{dx}$')

plt.subplot(155)
plt.axis('off')
plt.title(r'$\frac{dI}{dy}$')
plt.imshow(dy, cmap='gray')
plt.tight_layout()
plt.show()


# Gabor Filter
filtered_gray32 = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY).astype(np.float32)

kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel * kernel).sum())

filtered_gabor = cv2.filter2D(filtered_gray32, -1, kernel)

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(filtered_gray32, cmap='gray')

plt.subplot(132)
plt.title('kernel')
plt.imshow(kernel, cmap='gray')

plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered_gabor, cmap='gray')
plt.tight_layout()
plt.show()


cv2.waitKey()
cv2.destroyAllWindows()