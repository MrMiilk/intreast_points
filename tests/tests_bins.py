import numpy as np
import tensorflow as tf
import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import base64


FILE_PATH = "train//"
FILE_PATH2 = "training//"
H, W, C = 30, 40, 65

def get_batch(batch_size, iter=100):
    '''测试输入数据'''
    file_list = os.walk(FILE_PATH)
    # lile_list2 = os.walk(FILE_PATH2)
    files = []
    # files2 = []
    for file in file_list:
        files.extend(file[-1])
    # print(files)
    low = 0
    high = len(files)
    for _ in range(iter):
        idxs = np.random.randint(low, high, batch_size)
        lab = np.zeros((batch_size, H, W, C))
        X = np.zeros((batch_size, H*8, W*8, 3))
        for i, idx in enumerate(idxs):
            tmp = Path(FILE_PATH, str(idx)+".npy")
            tmp2 = FILE_PATH2 + str(idx)+".png"
            with open(tmp, 'rb') as f:
                arr = np.transpose(np.load(f), [1, 2, 0])
            grayImage = cv2.imread(tmp2)
            lab[i] = arr
            X[i] = grayImage
        # print(X.shape, lab.shape)
        yield X, lab


if __name__ == '__main__':
    count = 0
    for _ in range(3):
        for res in get_batch(10, 10):
            count += 1
            print(count)