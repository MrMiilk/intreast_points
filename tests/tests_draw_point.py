import numpy as np
import cv2
import matplotlib.pyplot as plt

a = np.load("4.npy")
# b = np.array(a+0.5, dtype=np.int)
# print(a, b)
# img = np.zeros((240, 320, 1))
# cv2.circle(img, (10, 10), 0, (1, ))
# cv2.imshow('', img)
# cv2.waitKey()
# print(img[8:12, 8:12])

img = cv2.imread('D:\\projects\\superpoint\\read_notes\\datas\\draw_cube\\images\\validation\\37.png')
