import sys
import os

path = "/home/int_point/"
sys.path.append(path)
from magic_point import *


def get_batch():
    type_lists = list(os.listdir(SYMTHETIC_FILE_PATH))
    X = np.zeros((1, H, W, 3))
    Label = np.zeros((1, H, W, C))
    for type_ in type_lists:
        basic_dir = SYMTHETIC_FILE_PATH + '/' + type_
        for idx in range(200):
            point_path = Path(basic_dir, TEST_POINT_PATH, str(idx) + ".npy")
            img_path = basic_dir + '/' + TEST_IMAGE_PATH + str(idx) + ".png"
            with open(point_path, 'rb') as f:
                points = np.array(np.load(f) + 0.5, dtype=np.int)
                img = np.zeros((240, 320, 1))
                for point in points:
                    cv2.circle(img, tuple(point), 0, (1,))
            grayImage = cv2.imread(img_path)
            X[0] = grayImage
            Label[0] = img
            yield X, Label, idx, type_, grayImage


if __name__ == '__main__':
    tf.reset_default_graph()
    input_shape = [None, 240, 320, 3]
    label_shape = [None, 240, 320, 1]
    batch_size = 1
    ckpt_name = 'checkpoints/-15000'
    Model = Magic_point()
    saver = tf.train.Saver()
    Model.model(input_shape, label_shape, opt='adam', lr=0.001, training=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_name)
        loss = 0.
        for X, label, idx, type_, gray_img in get_batch():
            feed_dict = {
                Model.inputs: X,
                Model.label_input: label,
                Model.training: 0,
            }
            point_position, loss_now = sess.run((Model.point_position, Model.loss), feed_dict=feed_dict)
            ##save point position##
            # np.save('./test_outputs/' + type_ + str(idx) + '.npy', point_position)
            point_position = point_position.reshape(240, 320)  # Atention, only for one picture
            point_position = np.vstack(np.nonzero(point_position)).T
            for point in point_position:
                cv2.circle(gray_img, tuple(point), 0, (1,))
            cv2.imwrite(str(Path('./test_outputs/' + type_, '{}.png'.format(i))), gray_img)
            loss += loss_now # 好像没什么意义
