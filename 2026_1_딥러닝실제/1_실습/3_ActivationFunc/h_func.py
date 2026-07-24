# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 19:45:08 2026

@author: MK
"""

import numpy as np
import matplotlib.pyplot as plt

# x 범위
x = np.linspace(-5, 5, 1000)

# 함수 정의
sigmoid = 1 / (1 + np.exp(-x))
step = np.where(x >= 0, 1, 0)
relu = np.maximum(0, x)

# subplot 생성
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Sigmoid
axes[0].plot(x, sigmoid, linewidth=2)
axes[0].set_title("Sigmoid Function")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].grid(True)
axes[0].axhline(0, linewidth=0.8)
axes[0].axvline(0, linewidth=0.8)
axes[0].set_ylim(-0.2, 1.2)

# 2. Step Function
axes[1].plot(x, step, linewidth=2)
axes[1].set_title("Step Function")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].grid(True)
axes[1].axhline(0, linewidth=0.8)
axes[1].axvline(0, linewidth=0.8)
axes[1].set_ylim(-0.2, 1.2)

# 3. ReLU
axes[2].plot(x, relu, linewidth=2)
axes[2].set_title("ReLU Function")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
axes[2].grid(True)
axes[2].axhline(0, linewidth=0.8)
axes[2].axvline(0, linewidth=0.8)
axes[2].set_ylim(-0.5, 5.5)

plt.tight_layout()
plt.show()