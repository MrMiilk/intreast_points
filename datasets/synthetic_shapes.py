import os
from pathlib import Path
import numpy as np
import cv2

from config import *

def get_batch(batch_size, iter=100):
    """生成器提供数据输入
    角点位置有很多是小数，这里使用最近邻的整数作为角点位置
    """
    ##TODO: 使用CPU生成图形，GPU运行网络##
    type_lists = list(os.listdir(SYMTHETIC_FILE_PATH))
    class_per_time = 2
    num_types = len(type_lists)
    num_batches = 10000//batch_size
    for _ in range(iter):
        idxs = np.arange(num_batches*batch_size)
        np.random.shuffle(idxs)
        idxs = idxs.reshape((num_batches, -1))
        lab = np.zeros((batch_size * class_per_time, H, W, C))
        X = np.zeros((batch_size * class_per_time, H, W, 3))
        for i in range(num_types // class_per_time):
            type_list = type_lists[i * class_per_time:i * class_per_time+2]
            for j in range(num_batches):
                batch = idxs[j]
                for i, idx in enumerate(batch):
                    for idx_t, t in enumerate(type_list):
                        point_path = Path(Path(SYMTHETIC_FILE_PATH, t, POINT_PATH), str(idx) + ".npy")
                        img_path = SYMTHETIC_FILE_PATH + '/' + t + '/' + IMAGE_PATH + str(idx) + ".png"
                        with open(point_path, 'rb') as f:
                            points = np.array(np.load(f) + 0.5, dtype=np.int)
                            img = np.zeros((240, 320, 1))
                            for point in points:
                                cv2.circle(img, tuple(point), 0, (1,))
                        grayImage = cv2.imread(img_path)
                        lab[i * idx_t] = img
                        X[i * idx_t] = grayImage
                yield X, lab


if __name__ == '__main__':
    for X, lab in get_batch(3, 3):
        # print(X, lab)
        pass