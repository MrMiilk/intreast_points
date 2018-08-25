import tensorflow as tf
import numpy as np

from bins import *


def conv_block(input, config, needs):
    layer_index, initializer, training = needs
    layer_name = 'conv' + str(layer_index)
    variable_params, layer_params = parse_config('conv_block', config)
    filter_shape, bias_shape = variable_params
    nolinear_func, stride, padding, use_bn = layer_params

    with tf.name_scope(layer_name):
        with tf.variable_scope('Variables', reuse=tf.AUTO_REUSE):
            filter = tf.get_variable(name='filter'+str(layer_name), shape=filter_shape, initializer=initializer)
            bias = tf.get_variable(name='bias'+str(layer_name), shape=bias_shape, initializer=tf.zeros_initializer)
        with tf.name_scope('conv'):
            x = tf.nn.conv2d(input, filter, stride, padding) + bias
            output = getattr(tf.nn, nolinear_func)(x)
    if use_bn:
        with tf.variable_scope('batch_normalization', reuse=tf.AUTO_REUSE):
            output = tf.layers.batch_normalization(output, training=training, name='bn' + str(layer_index))
    return output


def pooling_block(input, config, needs):
    layer_index = needs[0]
    layer_name = 'pool' + str(layer_index)
    window_shape, pooling_type, padding, stride = parse_config('pool', config)

    with tf.name_scope(layer_name):
        output = tf.nn.pool(input, window_shape, pooling_type, padding, strides=stride)
    return output


def fc_block():
    pass


# def sample_label(input, config=None, needs=None):
#     '''从H*W-->(H/8)*(W/8)*65'''
#     # H, W, C = parse_config('sample_conv', config)
#     with tf.name_scope('kernels'):
#         kernels = []
#         for i in range(8):
#             for j in range(8):
#                 tmp = np.zeros((8, 8))
#                 tmp[i, j] = 1.
#                 kernels.append(tf.Variable(initial_value=tmp.reshape(8, 8, 1, 1), trainable=False, name='kernel'+str(i+1)+str(j+1)))
#     with tf.name_scope('conv_samp'):
#         def func_(a, b):
#             tmp = np.sum(a, axis=-1, keepdims=True)
#             b[tmp > 0] = 1.
#             return b
#         channels = []
#         for i in range(8):
#             for j in range(8):
#                 tmp = tf.nn.conv2d(input, filter=kernels[i*8+j], padding='VALID', strides=(1, 8, 8, 1))
#                 channels.append(tmp)
#         tmp = tf.zeros_like(channels[0])
#         output = tf.concat(channels, -1)
#         tmp = tf.py_func(func_, [output, tmp], tf.float32)
#         # output = tf.Variable(expected_shape=(H, W, C), initial_value=tf.zeros_initializer(), trainable=False)
#         output = tf.concat([output, tmp], axis=-1)
#     return output


def sample_label(input, config=None, needs=None):
    '''从H*W-->(H/8)*(W/8)*65'''
    # H, W, C = parse_config('sample_conv', config)
    x = tf.space_to_depth(input, block_size=8)
    paddings = tf.constant([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 1]
    ])
    output = tf.pad(x, paddings=paddings)
    return output