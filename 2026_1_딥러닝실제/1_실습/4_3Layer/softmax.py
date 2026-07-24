# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 19:44:45 2026

@author: MK
"""

import numpy as np

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a 
    
    return y


a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y=exp_a / sum_exp_a
print("softmax process:", y)


softmax(a)
print("softmax func: ", y)