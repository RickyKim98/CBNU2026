# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:07:56 2026

@author: MK
"""

import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('Lenna.png')

plt.imshow(img)
plt.show()