# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 21:24:13 2026

@author: MK
"""


import numpy as np
import matplotlib.pyplot as plt

#데이터 준비
x = np.arange(1, 10, .1)
y = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y,  label = "sin")
plt.plot(x,y2, linestyle="--", label="cos")
plt.legend()
plt.grid(1)

plt.show()

