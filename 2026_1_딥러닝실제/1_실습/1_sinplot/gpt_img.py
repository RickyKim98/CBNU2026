# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:19:28 2026

@author: MK
"""

import matplotlib.pyplot as plt

# 사용자 바탕화면 경로
image_path = r"C:\Users\MK\Desktop\KakaoTalk_20250101_174242028.jpg"

# 이미지 읽기
img = plt.imread(image_path)

# 이미지 출력
plt.imshow(img)
plt.axis("off")   # 축 숨기기
plt.show()