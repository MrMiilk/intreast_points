import numpy as np
import tensorflow as tf

from utils import *


class Basic_model():
    '''
    为模型提供基类
    '''
    default_encoder_layers = ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv',]
    default_encoder_config = [
        [#block
            #for variable_param
            [(3, 3, 1, 64), (64,)],
            #for layer_param
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
        [
            [(3, 3, 64, 64), (64,)],
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
        [#max_pooling
            (2, 2), 'MAX', 'VALID'
        ],
        [
            [(3, 3, 64, 64), (64,)],
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
        [
            [(3, 3, 64, 64), (64,)],
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
        [  # max_pooling
            (2, 2), 'MAX', 'VALID'
        ],
        [
            [(3, 3, 64, 128), (128,)],
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
        [
            [(3, 3, 128, 128), (128,)],
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
        [  # max_pooling
            (2, 2), 'MAX', 'VALID'
        ],
        [
            [(3, 3, 128, 128), (128,)],
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
        [
            [(3, 3, 128, 128), (128,)],
            ['relu', (1, 2, 2, 1), 'SAME', True]
        ],
    ]

    def __init__(self, encoder_config=default_encoder_config, encoder_layers=default_encoder_layers):
        self.encoder_config = encoder_config
        self.encoder_layers = encoder_layers
        self.training = None
        self.inputs = None
        self.output = None
        self.label = None
        self.encoder_output = None
        self.decoder_output = None
        self.target = None
        self.loss = None
        self.optimzer = None
        self.initializer = tf.random_normal_initializer()

    def set_inputs(self, input_shape, label_shape):
        '''
        create placeholder for X_train and y_train
        :return:
        '''
        self.H_W = input_shape[1], input_shape[2]
        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.label = tf.placeholder(dtype=tf.float32, shape=label_shape, name='label')
        return

    def create_encoder(self):
        '''
        VGG_like encoder
        :return:
        '''
        x = self.inputs
        layers = self.encoder_layers
        config = self.encoder_config
        with tf.name_scope('Encoder'):
            for l in range(len(layers)):
                if layers[l] == 'conv':
                    needs = [l, self.initializer, self.training]
                    x = conv_block(x, config[l], needs)
                if layers[l] == 'pool':
                    needs = [l, ]
                    x = pooling_block(x, config[l], needs)
        self.encoder_output = x

    def define_loss(self):
        pass

    def Lp(self, logits, labels):
        labels = tf.to_float(labels)
        # W, H = self.H_W
        print(labels)
        print(logits)
        print(self.encoder_output)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss

    def model(self, input_shape, label_shape, opt):
        '''define how to build model'''
        pass

if __name__ == '__main__':
    input_shape = [None, 240, 320, 1]
    label_shape = []
    model = Basic_model()
    with tf.Session() as sess:
        initializer = tf.global_variables_initializer()
        sess.run(initializer)
        model.set_inputs(input_shape, label_shape)
        model.create_encoder()
        writer = tf.summary.FileWriter('logs/', sess.graph)