""""tests_ckpt为训练数据生成器，修改，将数据点画到图像上，检查输入数据集没问题"""
import sys
import os

path = "/home/int_point/"
sys.path.append(path)
from magic_point import *


def get_batch():
    type_lists = list(os.listdir(SYMTHETIC_FILE_PATH))
    X = np.zeros((1, H, W, 1))
    Label = np.zeros((1, H, W, C))
    for type_ in type_lists:
        basic_dir = SYMTHETIC_FILE_PATH + '/' + type_
        for idx in range(200):
            point_path = Path(basic_dir, TEST_POINT_PATH, str(idx) + ".npy")
            img_path = basic_dir + '/' + TEST_IMAGE_PATH + str(idx) + ".png"
            grayImage = cv2.imread(img_path)
            with open(point_path, 'rb') as f:
                points = np.array(np.load(f) + 0.5, dtype=np.int)
                img = np.zeros((240, 320, 1))
                for point in points:
                    cv2.circle(img, tuple(point), 0, (1,))
            # X[0, ..., 0] = grayImage[..., 0]
            # Label[0] = img
            yield grayImage[..., 0], img


if __name__ == '__main__':
    i = 0
    if not os.path.exists('./imgs/' + 'tmp/'):
        os.makedirs('./imgs/' + 'tmp/')
    for img, img_2 in get_batch():
        print(img_2.shape)
        print(np.sum(img_2))
        # cv2.imwrite('./imgs/' + 'tmp/' + str(i) + '.png', img)
        i += 1
