import cv2
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

corners = cv2.goodFeaturesToTrack(img, 100, 0.05, 10)

for c in corners:
    x, y = c[0]
    x, y = int(x), int(y)
    cv2.circle(img, (x, y), 5, 255, -1)

plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
