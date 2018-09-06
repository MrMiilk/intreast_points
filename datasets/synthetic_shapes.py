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
    file_list = os.walk(SYMTHETIC_FILE_PATH)
    files = []
    for file in file_list:
        files.extend(file[-1])
    num_batches = len(files)//batch_size
    for _ in range(iter):
        idxs = np.arange(num_batches*batch_size)
        np.random.shuffle(idxs)
        idxs = idxs.reshape((num_batches, -1))
        # print(H, W, C)
        lab = np.zeros((batch_size, H, W, C))
        X = np.zeros((batch_size, H, W, 3))
        for j in range(num_batches):
            batch = idxs[j]
            for i, idx in enumerate(batch):
                tmp = Path(SYMTHETIC_FILE_PATH, str(idx)+".npy")
                tmp2 = SYMTHETIC_FILE_PATH2 + str(idx)+".png"
                with open(tmp, 'rb') as f:
                    points = np.array(np.load(f) + 0.5, dtype=np.int)
                    img = np.zeros((240, 320, 1))
                    for point in points:
                        cv2.circle(img, tuple(point), 0, (1,))
                grayImage = cv2.imread(tmp2)
                lab[i] = img
                X[i] = grayImage
            yield X, lab


if __name__ == '__main__':
    for X, lab in get_batch(3, 3):
        # print(X, lab)
        pass