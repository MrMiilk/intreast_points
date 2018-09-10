import sys
import os

path = "/home/int_point/"
sys.path.append(path)
from magic_point import *


if __name__ == '__main__':
    tf.reset_default_graph()
    input_shape = [None, 240, 320, 3]
    label_shape = [None, 240, 320, 1]
    batch_size = 1
    ckpt_name = 'checkpoints/-15000'
    Model = Magic_point()
    saver = tf.train.Saver()
    Model.model(input_shape, label_shape, opt='sgd', lr=0.001, training=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt_name)
        loss = 0.
        i = 0
        for X, label in get_batch(batch_size, iter=1, img_p=TEST_IMAGE_PATH, pot_p=POINT_PATH):
            feed_dict = {
                Model.inputs: X,
                Model.label_input: label,
                Model.training: 0,
            }
            point_position, loss_now = sess.run((Model.point_position, Model.loss), feed_dict=feed_dict)
            ##save point position##
            np.save('./test_outputs/' + str(i) + '.npy', point_position)
            i += 1
            loss += loss_now # 好像没什么意义
