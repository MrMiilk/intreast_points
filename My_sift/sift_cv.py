import cv2
import numpy as np


img = cv2.imread('imgs/img[1].jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# key_points = sift.detect(gray_img, None)
key_points, descriptors = sift.detectAndCompute(gray_img, None)

# print(len(key_points), type(key_points))
# print(key_points[:2])
print(type(descriptors))
print(len(descriptors[0]))

cv2.drawKeypoints(img, key_points, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('', img)
cv2.waitKey()