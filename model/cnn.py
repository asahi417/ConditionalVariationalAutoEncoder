import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
import tensorflow.contrib.slim as slim


def convolution(x, weight_shape, stride, initializer, padding="SAME"):
    """ 2d convolution layer
    - weight_shape: width, height, input channel, output channel
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.nn.conv2d(x, weight, strides=[1, stride[0], stride[1], 1], padding=padding), bias)


def full_connected(x, weight_shape, initializer):
    """ fully connected layer
    - weight_shape: input size, output size
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)


class CNN(object):
    """ CNN """

    def __init__(self, activation=tf.nn.relu, learning_rate=0.001, batch_size=100, save_path=None, load_model=None,
                 keep_prob=0.8, architecture=None):
        """
        :param architecture:
        :param activation: activation function (tensor flow function)
        :param float learning_rate:
        :param int batch_size:
        """

        self.activation = activation
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        if architecture:
            self.architecture = architecture
        else:
            self.architecture = {
                "output": 10, "input": [28, 28, 1],
                "filter1": [4, 4, 1, 8], "stride1": [1, 1],
                "filter2": [4, 4, 8, 16], "stride2": [1, 1],
                "filter3": [4, 4, 16, 32], "stride3": [1, 1],
                "filter4": [4, 4, 32, 64], "stride4": [1, 1]
            }
        # Initializer
        if "relu" in self.activation.__name__:
            # # print("relu")
            # tf.truncated_normal(shape, stddev=0.1)
            self.ini_c, self.ini = variance_scaling_initializer(), variance_scaling_initializer()
        else:
            # # print("not relu")
            self.ini_c, self.ini = xavier_initializer_conv2d(), xavier_initializer()

        # Create network
        self._create_network()

        # Summary
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("accuracy", self.accuracy)
        # Launch the session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        # Summary writer for tensor board
        self.summary = tf.summary.merge_all()
        if save_path:
            self.writer = tf.summary.FileWriter(save_path, self.sess.graph)
        # Load model
        if load_model:
            tf.reset_default_graph()
            self.saver.restore(self.sess, load_model)

    def _create_network(self):
        """ Create Network, Define Loss Function and Optimizer """
        # tf Graph input
        _in_size = [None]
        _in_size += self.architecture["input"]
        self.x = tf.placeholder(tf.float32, _in_size, name="input")
        self.y = tf.placeholder(tf.float32, [None, self.architecture["output"]], name="output")
        self.is_training = tf.placeholder(tf.bool)
        _keep_prob = self.keep_prob if self.is_training is True else 1

        # print(self.x.shape, self.y.shape)
        _layer = convolution(self.x, self.architecture["filter1"], self.architecture["stride1"], self.ini_c)
        _layer = tf.nn.max_pool(_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _layer = self.activation(_layer)
        _layer = tf.nn.dropout(_layer, _keep_prob)
        # print(_layer.shape)
        _layer = convolution(_layer, self.architecture["filter2"], self.architecture["stride2"], self.ini_c)
        _layer = tf.nn.max_pool(_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _layer = self.activation(_layer)
        _layer = tf.nn.dropout(_layer, _keep_prob)
        # print(_layer.shape)
        _layer = convolution(_layer, self.architecture["filter3"], self.architecture["stride3"], self.ini_c)
        _layer = tf.nn.max_pool(_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _layer = self.activation(_layer)
        _layer = tf.nn.dropout(_layer, _keep_prob)
        # print(_layer.shape)
        _layer = convolution(_layer, self.architecture["filter4"], self.architecture["stride4"], self.ini_c)
        _layer = tf.nn.max_pool(_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        _layer = self.activation(_layer)
        _layer = tf.nn.dropout(_layer, _keep_prob)
        # print(_layer.shape)
        _layer = slim.flatten(_layer)
        _shape = _layer.shape.as_list()
        _logit = full_connected(_layer, [_shape[-1], self.architecture["output"]], self.ini)

        self.prediction = tf.nn.softmax(_logit)
        # Define loss function (cross entropy)
        self.loss = -tf.reduce_sum(self.y * tf.log(self.prediction + 1e-10))
        # Accuracy
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.prediction, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Define optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = slim.learning.create_train_op(self.loss, optimizer)
        # saver
        self.saver = tf.train.Saver()


if __name__ == '__main__':
    import os

    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    CNN()
