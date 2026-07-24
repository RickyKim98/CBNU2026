# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 20:07:54 2026

@author: MK
"""
import numpy as np

"""
# 1st AND (Threshold)
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
    
# 1st OR (Threshold) 
def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.2
    tmp = x1*w1 + x2*w2
    
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
 """   
# 2nd AND (BIAS) 
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    
    if tmp <= 0:
        return 0
    else:
        return 1
    
# 2nd NAND (BIAS) 
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    
    if tmp <= 0:
        return 0
    else:
        return 1
    
# 2nd OR (BIAS)     
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    
    if tmp <= 0:
        return 0
    else:
        return 1
    
# XOR
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))   # 0
print(XOR(0, 1))   # 1
print(XOR(1, 0))   # 1
print(XOR(1, 1))   # 0