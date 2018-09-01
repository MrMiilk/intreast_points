import numpy as np
import tensorflow as tf

from utils import *


class Basic_model():
    '''
    为模型提供基类
    default_encoder_layers和default_encoder_config定义了网络结构
    '''
    default_encoder_layers = ['conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv', 'pool', 'conv', 'conv',]
    default_encoder_config = [
        [#block
            #for variable_param
            [(3, 3, 3, 64), (64,)],
            #for layer_param
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [
            [(3, 3, 64, 64), (64,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [#max_pooling
            (2, 2), 'MAX', 'SAME', [2, 2]
        ],
        [
            [(3, 3, 64, 64), (64,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [
            [(3, 3, 64, 64), (64,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [  # max_pooling
            (2, 2), 'MAX', 'SAME', [2, 2]
        ],
        [
            [(3, 3, 64, 128), (128,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [
            [(3, 3, 128, 128), (128,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [  # max_pooling
            (2, 2), 'MAX', 'SAME', [2, 2]
        ],
        [
            [(3, 3, 128, 128), (128,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [
            [(3, 3, 128, 128), (128,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
    ]

    def __init__(self, encoder_config=default_encoder_config, encoder_layers=default_encoder_layers):
        '''传入描述符头的结构，同时为网络提供一些占位符'''
        self.encoder_config = encoder_config
        self.encoder_layers = encoder_layers
        self.training = None                   # 描述本次是否是训练，在测试样本时时为False
        self.inputs = None                     # 输入训练集
        self.output = None
        self.label = None                      # 训练的标签
        self.encoder_output = None             # 这个类的输出，之后再传入decoder
        self.decoder_output = None             # decoder 的输出，目前为特征点位置
        self.target = None
        self.loss = None                       # 损失函数
        self.optimzer = None                   # 优化器
        self.initializer = tf.random_normal_initializer() # 网络权重初始化方法，正太分布

    def set_inputs(self, input_shape, label_shape):
        '''
        create placeholder for X_train and y_train，数据占位符，为输入数据提供占位
        '''
        self.H_W = input_shape[1], input_shape[2]
        with tf.name_scope('Input'):
            self.inputs = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input')
            self.training = tf.placeholder(dtype=tf.bool, name='training')
            self.label = tf.placeholder(dtype=tf.float32, shape=label_shape, name='label')
            self.label = sample_label(self.label)
        return

    def create_encoder(self):
        '''
        VGG_like encoder
        :return:
        '''
        x = self.inputs
        layers = self.encoder_layers
        config = self.encoder_config
        # 根据参数堆叠模块
        with tf.name_scope('Encoder'):
            for l in range(len(layers)):
                if layers[l] == 'conv':
                    needs = [l, self.initializer, self.training]
                    x = conv_block(x, config[l], needs)
                if layers[l] == 'pool':
                    needs = [l, ]
                    x = pooling_block(x, config[l], needs)
                # print(x)
        self.encoder_output = x

    def define_loss(self):
        '''损失函数：不同的网络里面会具体定义'''
        pass

    def Lp(self, logits, labels):
        '''论文使用的符号，特征点位置检测的损失函数'''
        ##TODO:需要修改，目前是交叉熵损失##
        labels = tf.to_float(labels)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        return loss

    def Ld(self):
        '''论文使用的符号，特征点描述符计算的损失函数'''
        ##TODO：##
        pass

    def model(self, input_shape, label_shape, opt):
        '''define how to build model，在不同的网络会具体完成'''
        pass