from magic_point import *


if __name__ == '__main__':
    input_shape = [None, 240, 320, 3]
    label_shape = [None, 240, 320, 1]
    batch_size = 4
    ckpt_name = ''
    Model = Magic_point()
    saver = tf.train.Saver()
    Model.model(input_shape, label_shape, opt='sgd', lr=0.001, training=False)
    with tf.Session() as sess:
        saver.restore(sess, ckpt_name)
        loss = 0.
        for X, label in get_batch(batch_size, iter=1, img_p=TEST_IMAGE_PATH, pot_p=POINT_PATH):
            feed_dict = {
                Model.inputs: X,
                Model.label_input: label,
                Model.training: 0,
            }
            point_position, loss_now = sess.run((Model.point_position, Model.loss), feed_dict=feed_dict)
            ##save point position##
            np.save('', point_position)
            loss += loss_now # 好像没什么意义
