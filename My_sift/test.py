import cv2
import matplotlib.pyplot as plt

img = cv2.imread('./imgs/img[1].jpg')
grag = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SURF_create()
kp = sift.detect(grag, None)
print(kp)
img = cv2.drawKeypoints(grag, kp, img)

cv2.imshow('img:', img)
cv2.waitKey()