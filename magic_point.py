from basic_model import Basic_model
from utils import *
from tensorflow.train import AdamOptimizer, GradientDescentOptimizer
from datasets.synthetic_shapes import *


class Magic_point(Basic_model):
    '''Magic Point检测器，下面两个依旧是默认的网络建立方式'''
    default_decoder_layer = ['conv', 'conv']
    default_decoder_config = [
        [
            [(3, 3, 128, 256), (256,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
        [
            [(1, 1, 256, 65), (65,)],
            ['relu', (1, 1, 1, 1), 'SAME', True]
        ],
    ]
    default_reshape_config = []

    def create_decode_head(self, layer=default_decoder_layer, config=default_decoder_config):
        '''特征点检测头，根据参数搭建'''
        with tf.name_scope('Point_decoder'):
            x = self.encoder_output
            for l in range(len(layer)):
                if layer[l] == 'conv':
                    needs = [20+l, self.initializer, self.training]
                    x = conv_block(x, config[l], needs)
        self.decoder_output = x

    def reshape_output(self, config=default_reshape_config):
        '''最后从(H/8)*(W/8)*65-->H*W*1'''
        def func_(x):
            sign = np.max(x, axis=-1, keepdims=True)
            res = x == sign
            return np.array(res, dtype=np.float32)
        x = self.decoder_output
        x = tf.nn.softmax(x, axis=-1)
        with tf.name_scope('Reshape_output'):
            # x = tf.nn.softmax(x, axis=-1)
            x = tf.py_func(func_, [x], tf.float32)
            x = x[..., :-1]
            self.point_position = tf.depth_to_space(x, block_size=8)
            # print('check point_position: ', self.point_position)
        return

    def define_loss(self):
        '''定义损失函数计算，Lp需要修改'''
        with tf.name_scope('Loss'):
            H, W = self.H_W
            self.loss = (64. / W * H) * tf.reduce_sum(self.Lp(self.decoder_output, self.label))
            tf.summary.scalar('Loss', self.loss)
        return

    def model(self, input_shape, label_shape, opt, lr=1e-4, training=True):
        '''define how to build model'''
        ##TODO:定义网络，使用"Ctrl + 点击函数名"查看函数##
        self.set_inputs(input_shape, label_shape)
        self.create_encoder()
        self.create_decode_head()
        if not training:
            self.reshape_output()
        self.define_loss()

        ##TODO：选择优化器， 优化器参数完善##
        if opt == 'adam':
            self.optimzer = AdamOptimizer(learning_rate=lr)
        elif opt == 'sgd':
            self.optimzer = GradientDescentOptimizer(learning_rate=lr)
        ##TODO：一次训练迭代操作##
        self.train_op = self.optimzer.minimize(self.loss, global_step=self.global_step)


if __name__ == '__main__':
    '''Magic Point运行检测'''
    input_shape = [None, 240, 320, 1]
    label_shape = [None, 240, 320, 1]
    epoch = 2000                      # 迭代的epoch
    batch_size = 14
    lr = 0.0001
    opt = 'adam'

    Model = Magic_point()
    with tf.Session() as sess:
        # initer = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        Model.model(input_shape, label_shape, opt, lr=lr)               # 定义模型
        sess.run(tf.initialize_all_variables())                  # 初始化网络
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/', sess.graph)  # 写入logs文件
        saver = tf.train.Saver(max_to_keep=4)
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints/'))
        stepts = 0
        for i in range(epoch):                                   # 迭代
            for X, labels in get_batch(batch_size, 1000):
                # print(type(X), type(labels))
                stepts += 1
                feed_dict = {
                    Model.inputs: X,
                    Model.label_input: labels,
                    Model.training: 1,
                }                               # 输入数据填充占位
                sess.run(Model.train_op, feed_dict=feed_dict)   # 向前运行一次网络

                if stepts % 50 == 0:
                    rs, loss = sess.run((merged, Model.loss), feed_dict=feed_dict)
                    writer.add_summary(rs, i)
                    print('In step {}, loss:{}'.format(stepts, loss))

                if stepts % 500 == 0:
                    saver.save(sess, SAVE_PATH, global_step=stepts)