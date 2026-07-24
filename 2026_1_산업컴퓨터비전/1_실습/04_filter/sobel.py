#import argparse
import cv2, numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#parser = argparse.ArgumentParser()
#parser.add_argument('--path', default='Lenna.png', help='Image path')
#params = parser.parse_args()


image = cv2.imread('Lenna.png', 0)

dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

plt.figure(figsize=(8,3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')

plt.subplot(132)
plt.axis('off')
plt.imshow(dx, cmap='gray')
plt.title(r'$\frac{dI}{dx}$')

plt.subplot(133)
plt.axis('off')
plt.title(r'$\frac{dI}{dy}$')
plt.imshow(dy, cmap='gray')
plt.tight_layout()
plt.show()