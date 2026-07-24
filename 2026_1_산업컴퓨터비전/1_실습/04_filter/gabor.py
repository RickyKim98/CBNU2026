#import argparse
import cv2, numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math
#parser = argparse.ArgumentParser()
#parser.add_argument('--path', default='Lenna.png', help='Image path')
#params = parser.parse_args()


image = cv2.imread('Lenna.png', 0).astype(np.float32) / 255

kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel /= math.sqrt((kernel * kernel).sum())

filtered = cv2.filter2D(image, -1, kernel)

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')

plt.subplot(132)
plt.title('kernel')
plt.imshow(kernel, cmap='gray')

plt.subplot(133)
plt.axis('off')
plt.title('filtered')
plt.imshow(filtered, cmap='gray')
plt.tight_layout()
plt.show()