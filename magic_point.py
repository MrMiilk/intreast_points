from basic_model import Basic_model
from utils import *
from tensorflow.train import AdamOptimizer, GradientDescentOptimizer
from tests_bins import *


class Magic_point(Basic_model):
    default_decoder_layer = ['conv', 'conv']
    default_decoder_config = [
        [
            [(3, 3, 128, 265), (265,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [
            [(1, 1, 265, 65), (65,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
    ]

    def create_decode_head(self, layer=default_decoder_layer, config=default_decoder_config):
        '''
        特征点检测头
        :return:
        '''
        with tf.name_scope('Point_decoder'):
            x = self.encoder_output
            for l in range(len(layer)):
                if layer[l] == 'conv':
                    needs = [20+l, self.initializer, self.training]
                    x = conv_block(x, config[l], needs)
        self.decoder_output = x

    def define_loss(self):
        H, W = self.H_W
        self.loss = (1. / W * H) * tf.reduce_sum(self.Lp(self.decoder_output, self.label))

    def model(self, input_shape, label_shape, opt, lr=1e-3):
        '''define how to build model'''
        self.set_inputs(input_shape, label_shape)
        self.create_encoder()
        self.create_decode_head()
        self.define_loss()

        if opt == 'adam':
            self.optimzer = AdamOptimizer(learning_rate=lr)
        elif opt == 'sgd':
            self.optimzer = GradientDescentOptimizer(learning_rate=lr)
        self.train_op = self.optimzer.minimize(self.loss)



if __name__ == '__main__':
    input_shape = [None, 240, 320, 3]
    label_shape = [None, 30, 40, 65]
    epoch = 1000
    batch_size = 16
    opt = 'sgd'

    Model = Magic_point()
    with tf.Session() as sess:
        # initer = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        Model.model(input_shape, label_shape, opt)
        writer = tf.summary.FileWriter('logs/', sess.graph)
        sess.run(tf.initialize_all_variables())
        for i in range(epoch):
            for X, labels in get_batch(batch_size, 1000):
                # print(type(X), type(labels))
                feed_dict = {
                    Model.inputs: X,
                    Model.label: labels,
                    Model.training: 1,
                }
                sess.run(Model.train_op, feed_dict=feed_dict)



