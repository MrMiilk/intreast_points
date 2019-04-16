import cv2
import numpy as np
import matplotlib.pyplot as plt
from conf import *
from bins import *


def main(img):
    Gauss_pyramid = {}
    DoG_pyramid = {}
    img2 = resize2(img)  # 双线性插值得到扩大的图像
    img2 = to_gray(img2)
    octave = Get_octave(img.shape[:2])
    print('Image Octave:', octave)
    #获取Gauss金字塔
    Gauss_pyramid = get_Gauss_pyramid(img2, octave)

    #测试：输出Gauss金字塔
    # for i in Gauss_pyramid.keys():
    #     for s in range(S + 3):
    #         cv2.imshow('img'+str(s), Gauss_pyramid[i][s])
    #     cv2.waitKey()
    DoG_pyramid = get_DoG_pyramid(Gauss_pyramid, octave)
    #测试：输出DoG金字塔
    # for i in DoG_pyramid.keys():
    #     for s in range(S + 2):
    #         cv2.imshow('img'+str(s), DoG_pyramid[i][s])
    #     cv2.waitKey()

    #从DoG获取大致兴趣点位置
    points_dir = find_points_from_DoG(DoG_pyramid, octave)
    # 大致兴趣点绘图
    pointed_img = img.copy()
    for o in points_dir.keys():
        octave = int(o)
        points = points_dir[o]
        temp = 2**(octave - 1)
        points = [(int(y*temp), int(x*temp)) for (x, y, _) in points]
        for point in points:
            cv2.circle(pointed_img, point, 1, (255, 0, 0), 1)
    cv2.imshow('Img11:', pointed_img)
    cv2.waitKey()

    #修正，获取兴趣点
    fixed_points = get_real_points(DoG_pyramid, points_dir, octave)
    #绘图
    pointed_img = img.copy()
    for o in fixed_points.keys():
        octave = int(o)
        points = fixed_points[o]
        temp = 2**(octave - 1)
        points = [(int(y*temp), int(x*temp)) for (x, y, _) in points]
        for point in points:
            cv2.circle(pointed_img, point, 1, (255, 0, 0), 1)
    cv2.imshow('ImgFinal:', pointed_img)

    #获取兴趣点主方向，经过这步，每个关键点有三个信息：位置、尺度、方向
    poins_with_position_direction = get_main_direct(Gauss_pyramid, DoG_pyramid, fixed_points, octave)
    print(poins_with_position_direction)
    cv2.waitKey()
    #描述符计算


if __name__ == '__main__':
    img = cv2.imread('./imgs/img[1].jpg')
    img = resize(img)#对不符合2^n的图像进行边缘拓展
    main(img)