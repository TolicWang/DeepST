# @Time    : 2018/12/10 10:59
# @Email  : wangchengo@126.com
# @File   : STResNet.py
# package version:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2
#               tensorflow 1.5.0
import tensorflow as tf
import numpy as np
import logging
import os


def _Convolution2D(input, kernel_size, padding='SAME', is_relu=False):
    if is_relu:
        input = tf.nn.relu(input)
    with tf.variable_scope('conv_on_%s' % input.name[:-2]):
        weights = tf.get_variable(name='weight', shape=kernel_size, dtype=tf.float32,
                                  initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
    bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[kernel_size[3]]))
    conv = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding=padding)
    convs = tf.nn.bias_add(conv, bias)
    return convs


def _shortcut(input, residual):
    return input + residual


def _bn_relu_conv(input, nb_filter, bn=False):
    conv = _Convolution2D(input, kernel_size=[3, 3, input.shape[3], nb_filter], is_relu=True)
    return conv


def _residual_unit(input, nb_filter):
    residual = _bn_relu_conv(input, nb_filter)
    residual = _bn_relu_conv(residual, nb_filter)
    return _shortcut(input, residual)


def _Fusion(inputs):
    outputs = []
    for input in inputs:
        with tf.name_scope('fusion_%s' % input.name[:-2]):
            w = tf.Variable(tf.random_uniform(shape=[32, 32, 2], dtype=tf.float32), name='w')
            output = input * w
        outputs.append(output)
    return outputs[0] + outputs[1] + outputs[2]


def _ResUnits(input, nb_filter, repetations=3):
    for i in range(repetations):
        input = _residual_unit(input, nb_filter)
    return input


def _FullCon(input, output_dim, is_relu=False):
    with tf.variable_scope('fc_%s' % input.name[:-2]):
        w = tf.get_variable(name='w', shape=[input.shape[1], output_dim],
                            initializer=tf.glorot_uniform_initializer(dtype=tf.float32))
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[output_dim]), name='b')
    out = tf.nn.xw_plus_b(input, w, b)
    if is_relu:
        out = tf.nn.relu(out)
    return out


def gen_batch(x, y, index, batch_size=32):
    xc, xp, xt, ext = x[0], x[1], x[2], x[3]
    begin = index * batch_size
    end = begin + batch_size
    if (index + 1) * batch_size + batch_size > len(y):  # the last batch
        end = len(y)
    x_batch = [xc[begin:end], xp[begin:end], xt[begin:end], ext[begin:end], ]
    y_batch = y[begin:end]
    return x_batch, y_batch


class STResNet():
    def __init__(self,
                 learning_rate=0.0001,
                 epoches=50,
                 batch_size=32,
                 model_path='MODEL',
                 len_closeness=3,
                 len_period=1,
                 len_trend=1,
                 external_dim=28,
                 map_heigh=32,
                 map_width=32,
                 nb_flow=2,
                 nb_residual_unit=2):
        self.epoches = epoches
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model_path = model_path
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend
        self.external_dim = external_dim
        self.map_heigh = map_heigh
        self.map_width = map_width
        self.nb_flow = nb_flow
        self.nb_residual_unit = nb_residual_unit
        self.logger = logging.getLogger(__name__)
        self._build_placeholder()
        self._build_stresnet()

    def _build_placeholder(self):
        with tf.name_scope('model_inputs'):
            self.input_xc = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    self.nb_flow * self.len_closeness], name='input_xc')
            self.input_xp = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    self.nb_flow * self.len_period], name='input_xp')
            self.input_xt = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    self.nb_flow * self.len_trend], name='input_xt')
            self.input_ext = tf.placeholder(dtype=tf.float32, shape=[None, self.external_dim], name='input_external')
            self.output_y = tf.placeholder(dtype=tf.float32, shape=[None, self.map_heigh, self.map_width,
                                                                    self.nb_flow], name='output_y')

    def _build_stresnet(self, ):
        self.logger.info('### Building Network...')
        with tf.name_scope('build_CPT'):  # Closeness, Period, Trend
            inputs = [self.input_xc, self.input_xp, self.input_xt]
            outputs = []
            for input in inputs:
                conv1 = _Convolution2D(input, kernel_size=[3, 3, input.shape[3], 64])
                residual_output = _ResUnits(input=conv1, nb_filter=64, repetations=self.nb_residual_unit)
                conv2 = _Convolution2D(residual_output, kernel_size=[3, 3, residual_output.shape[3], self.nb_flow],
                                       is_relu=True)
                outputs.append(conv2)
            if len(outputs) == 1:
                main_output = outputs[0]
            else:
                main_output = _Fusion(outputs)
            # self.logger.debug('### Shape after fusion operation:', main_output.shape)

        with tf.name_scope('build_E'):  # external
            if self.external_dim > 0:
                embedding = _FullCon(self.input_ext, output_dim=10, is_relu=True)
                h1 = _FullCon(embedding, output_dim=self.nb_flow * self.map_heigh * self.map_width, is_relu=True)
                external_output = tf.reshape(h1, shape=[-1, self.map_heigh, self.map_width, self.nb_flow])
                main_output += external_output
                # self.logger.debug('### Shape after add external data:', main_output.shape)
        self.logits = tf.nn.tanh(main_output)

        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(self.logits - self.output_y))

    def evaluate(self, mmn, x, y):
        with tf.name_scope('mse'):
            _min = mmn._min
            _max = mmn._max
            predict = 0.5 * (self.logits + 1) * (_max - _min) + _min
            output_y = 0.5 * (self.output_y + 1) * (_max - _min) + _min
            square = tf.reduce_sum(tf.square(predict - output_y))
        saver = tf.train.Saver(max_to_keep=3)
        with tf.Session() as sess:
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
            else:
                return
            n_chunk = len(y) // self.batch_size
            total_square = 0
            for batch in range(n_chunk):
                x_batch, y_batch = gen_batch(x, y, batch, batch_size=self.batch_size)
                feed = {self.input_xc: x_batch[0],
                        self.input_xp: x_batch[1],
                        self.input_xt: x_batch[2],
                        self.input_ext: x_batch[3],
                        self.output_y: y_batch}
                total_square += sess.run(square, feed_dict=feed)
            rmse = np.sqrt(total_square / (len(y) * self.map_heigh * self.map_width * self.nb_flow))
            print('### RMSE: ', rmse)

    def train(self, x, y):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        with tf.device('/gpu:0'):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        self.logger.info('### Training...')
        saver = tf.train.Saver(max_to_keep=3)
        model_name = 'c{}.p{}.t{}.resunit{}.lr{}'.format(
            self.len_closeness, self.len_period, self.len_trend, self.nb_residual_unit, self.learning_rate)
        with tf.Session() as sess:
            self.logger.debug('### Initializing...')
            sess.run(tf.global_variables_initializer())
            start_epoch = 0
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
                start_epoch += int(check_point.split('-')[-1])
                self.logger.info("### Loading exist model <{}> successfully...".format(check_point))
            total_loss = 0
            try:
                for epoch in range(start_epoch, self.epoches):
                    n_chunk = len(y) // self.batch_size
                    ave_loss = total_loss / n_chunk
                    total_loss = 0
                    for batch in range(n_chunk):
                        x_batch, y_batch = gen_batch(x, y, batch, batch_size=self.batch_size)
                        feed = {self.input_xc: x_batch[0],
                                self.input_xp: x_batch[1],
                                self.input_xt: x_batch[2],
                                self.input_ext: x_batch[3],
                                self.output_y: y_batch}
                        loss, _ = sess.run([self.loss, train_op], feed_dict=feed)
                        total_loss += loss
                        if batch % 50 == 0:
                            self.logger.info(
                                '### Epoch:%d, last epoch loss ave:%.5f batch:%d, current epoch loss:%.5f' % (
                                    epoch, ave_loss, batch, loss))
                    if epoch % 3 == 0:
                        self.logger.info('### Saving model...')
                        saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch)
            except KeyboardInterrupt:
                self.logger.warning("KeyboardInterrupt saving...")
                saver.save(sess, os.path.join(self.model_path, model_name), global_step=epoch - 1)

