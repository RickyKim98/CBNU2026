import cv2
import numpy as np


np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)


data = np.load('data/stereo/case1/stereo.npy').item()

Kl, Kr, Dl, Dr, left_pts, right_pts, E_from_stereo, F_from_stereo = \
    data['Kl'], data['Kr'], data['Dl'], data['Dr'], data['left_pts'], data['right_pts'], data['E'], data['F']


left_pts = np.vstack(left_pts)
right_pts = np.vstack(right_pts)


left_pts = cv2.undistortPoints(left_pts, Kl, Dl, P=Kl)
right_pts = cv2.undistortPoints(right_pts, Kr, Dr, P=Kr)


F, mask = cv2.findFundamentalMat(left_pts, right_pts, cv2.FM_LMEDS)


E = Kr.T @ F @ Kl


print('Fundamental matrix:')
print(F)

print('Essential matrix:')
print(E)