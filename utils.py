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
    window_shape, pooling_type, padding = parse_config('pool', config)

    with tf.name_scope(layer_name):
        output = tf.nn.pool(input, window_shape, pooling_type, padding)
    return output

def fc_block():
    pass