# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:12:40 2026

@author: MK
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, t_train), (x_test, t_test) = mnist.load_data()

print(x_train.shape)   # (60000, 28, 28)
print(t_train.shape)   # (60000,)

plt.imshow(x_train[0], cmap='gray')
plt.title(f"label = {t_train[0]}")
plt.axis('off')
plt.show()