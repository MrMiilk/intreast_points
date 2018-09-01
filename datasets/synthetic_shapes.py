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
        idxs = np.random.shuffle(np.arange(num_batches*batch_size)).reshape((num_batches, -1))
        lab = np.zeros((batch_size, H/8, W/8, C))
        X = np.zeros((batch_size, H, W, 3))
        for j in range(num_batches):
            batch = idxs[j]
            for i, idx in enumerate(batch):
                tmp = Path(SYMTHETIC_FILE_PATH, str(idx)+".npy")
                tmp2 = SYMTHETIC_FILE_PATH2 + str(idx)+".png"
                with open(tmp, 'rb') as f:
                    arr = np.transpose(np.load(f), [1, 2, 0])
                grayImage = cv2.imread(tmp2)
                lab[i] = arr
                X[i] = grayImage
            yield X, lab