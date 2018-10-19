import tensorflow as tf
import numpy as np
#import cv2
"""测试网络输入输出的reshape，没有问题"""

def sample_label(input, config=None, needs=None):
    '''从H*W-->(H/8)*(W/8)*65'''
    # H, W, C = parse_config('sample_conv', config)
    with tf.name_scope("label_reshape"):
        def func_(x):
            b = 1 - np.sum(x, axis=-1) > 0
            return np.expand_dims(np.array(b, dtype=np.float32), axis=-1)
        x = tf.space_to_depth(input, block_size=8)
        # paddings = tf.constant([
        #     [0, 0],
        #     [0, 0],
        #     [0, 0],
        #     [0, 1]
        # ])
        # output = tf.pad(x, paddings=paddings)
        b = tf.py_func(func_, [x], tf.float32)
        output = tf.concat([x, b], axis=-1)
    return output


def reshape_output(x):
    '''最后从(H/8)*(W/8)*65-->H*W*1'''
    def func_(x):
        sign = np.max(x, axis=-1, keepdims=True)
        res = x == sign
        return np.array(res, dtype=np.float32)
    # x = self.decoder_output
    x = tf.nn.softmax(x, axis=-1)
    with tf.name_scope('Reshape_output'):
        # x = tf.nn.softmax(x, axis=-1)
        x = tf.py_func(func_, [x], tf.float32)
        x = x[..., :-1]
        point_position = tf.depth_to_space(x, block_size=8)
        # print('check point_position: ', self.point_position)
        return point_position


if __name__ == '__main__':
    x = tf.placeholder(shape=[None, 240, 320, 1], dtype=tf.float32)
    x_test = np.zeros([1, 240, 320, 1], dtype=np.float32)
    x_test[0, 100, 150, 0] = 1
    output = sample_label(x)
    print(output)
    x_pred = reshape_output(output)
    print(x_pred)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = {
            x: x_test
        }
        output__, x_pred__ = sess.run([output, x_pred], feed_dict=feed_dict)
    print(output__.shape)
    print(x_pred__.shape)
    print(x_pred__[0, 100, 150, 0])
    print(np.sum(output__))
    print(np.sum(x_pred__))
    point_position = np.vstack(np.nonzero(x_pred__)).T
    gray_img = np.zeros([1, 240, 320, 1], dtype=np.float32)
    # for point in point_position:
    #     cv2.circle(gray_img, center=tuple(point), radius=0, color=(1, 1, 1))
    # print(np.vstack(np.nonzero(gray_img)).T)