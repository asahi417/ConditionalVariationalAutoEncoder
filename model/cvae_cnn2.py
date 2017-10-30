import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
import tensorflow.contrib.slim as slim
import numpy as np


def image_size(_shape, stride):
    return int(np.ceil(_shape[0] / stride[0])), int(np.ceil(_shape[1] / stride[1]))


def convolution(x, weight_shape, stride, initializer, padding="SAME"):
    """ 2d convolution layer
    - weight_shape: width, height, input channel, output channel
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.nn.conv2d(x, weight, strides=[1, stride[0], stride[1], 1], padding=padding), bias)


def deconvolution(x, weight_shape, output_shape, stride, initializer, padding="SAME"):
    """ 2d deconvolution layer
    - weight_shape: width, height, input channel, output channel
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[2]]), dtype=tf.float32)
    _layer = tf.nn.conv2d_transpose(x, weight, output_shape=output_shape, strides=[1, stride[0], stride[1], 1],
                                    padding=padding, data_format="NHWC")
    return tf.add(_layer, bias)


def full_connected(x, weight_shape, initializer):
    """ fully connected layer
    - weight_shape: input size, output size
    """
    weight = tf.Variable(initializer(shape=weight_shape))
    bias = tf.Variable(tf.zeros([weight_shape[-1]]), dtype=tf.float32)
    return tf.add(tf.matmul(x, weight), bias)


def reconstruction_loss(original, reconstruction, eps=1e-10):
    """
    The reconstruction loss (the negative log probability of the input under the reconstructed Bernoulli distribution
    induced by the decoder in the data space). This can be interpreted as the number of "nats" required for
    reconstructing the input when the activation in latent is given.
    Adding 1e-10 to avoid evaluation of log(0.0)
    """
    _tmp = original * tf.log(eps + reconstruction) + (1 - original) * tf.log(eps + 1 - reconstruction)
    return -tf.reduce_sum(_tmp, 1)


def latent_loss(latent_mean, latent_log_sigma_sq):
    """
    The latent loss, which is defined as the Kullback Leibler divergence between the distribution in latent space
    induced by the encoder on the data and some prior. This acts as a kind of regularizer. This can be interpreted as
    the number of "nats" required for transmitting the the latent space distribution given the prior.
    """
    # clipping: remedy for explosion
    latent_log_sigma_sq = tf.clip_by_value(latent_log_sigma_sq, clip_value_min=-1e-5, clip_value_max=1e+5)
    return -0.5 * tf.reduce_sum(1 + latent_log_sigma_sq - tf.square(latent_mean) - tf.exp(latent_log_sigma_sq), 1)


class ConditionalVAE(object):
    """ Conditional VAE
    Inputs data must be normalized to be in range of 0 to 1
    (since VAE uses Bernoulli distribution for reconstruction loss)
    """

    def __init__(self, label_size, network_architecture=None, activation=tf.nn.relu,
                 learning_rate=0.001, batch_size=100, save_path=None, load_model=None):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input
            n_z: dimensionality of latent space

        :param activation: activation function (tensor flow function)
        :param float learning_rate:
        :param int batch_size:
        """
        if network_architecture:
            self.network_architecture = network_architecture
        else:
            self.network_architecture = dict(n_input=[28, 28, 1], n_z=20)

        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.label_size = label_size

        # Initializer
        if "relu" in self.activation.__name__:
            self.ini_c, self.ini = variance_scaling_initializer(), variance_scaling_initializer()
        else:
            self.ini_c, self.ini = xavier_initializer_conv2d(), xavier_initializer()

        # Create network
        self._create_network()

        # Summary
        tf.summary.scalar("loss", self.loss)
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
        _in_size += self.network_architecture["n_input"]
        self.x = tf.placeholder(tf.float32, _in_size, name="input")
        self.y = tf.placeholder(tf.float32, [None, self.label_size], name="output")

        # Build conditional input
        _label = tf.reshape(self.y, [-1, 1, 1, self.label_size])
        _one = tf.ones([self.batch_size] + self.network_architecture["n_input"][0:-1] + [self.label_size])
        _label = _one * _label
        _layer = tf.concat([self.x, _label], axis=3)
        _ch = self.network_architecture["n_input"][2] + self.label_size

        # Encoder network to determine mean and (log) variance of Gaussian distribution in latent space
        with tf.variable_scope("encoder"):
            # convolution 1
            _layer = convolution(_layer, [5, 5, _ch, 16], [2, 2], self.ini_c, padding="VALID")
            _layer = self.activation(_layer)
            # convolution 2
            _layer = convolution(_layer, [5, 5, 16, 32], [2, 2], self.ini_c, padding="VALID")
            _layer = self.activation(_layer)

            _layer = slim.flatten(_layer)
            # print(_layer.shape)
            _shape = _layer.shape.as_list()
            self.z_mean = full_connected(_layer, [_shape[-1], self.network_architecture["n_z"]], self.ini)
            self.z_log_sigma_sq = full_connected(_layer, [_shape[-1], self.network_architecture["n_z"]], self.ini)

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        _layer = tf.concat([self.z, self.y], axis=1)

        # Decoder to determine mean of Bernoulli distribution of reconstructed input
        with tf.variable_scope("decoder"):
            stride_0, stride_1, stride_2 = [2, 2], [2, 2], [2, 2]
            _w0, _h0 = self.network_architecture["n_input"][0:-1]
            _w1, _h1 = image_size([_w0, _h0], stride_0)
            _w2, _h2 = image_size([_w1, _h1], stride_1)
            # print(_w0, _w1, _w2)
            # full connect
            _in_size = self.network_architecture["n_z"] + self.label_size
            _out = 8
            # print(_layer.shape)
            _layer = full_connected(_layer, [_in_size, int(_w2 * _h2 * _out)], self.ini)
            _layer = self.activation(_layer)
            # reshape to the image
            # print(_layer.shape)
            _layer = tf.reshape(_layer, [-1, _w2, _h2, _out])
            # deconvolution 1
            # print(_layer.shape)
            _out, _in = 8, _out
            _layer = deconvolution(_layer, [5, 5, _out, _in], [self.batch_size, _w1, _h1, _out], stride_2, self.ini_c)
            _layer = self.activation(_layer)
            # deconvolution 2
            # print(_layer.shape)
            _out, _in = 1, _out
            _layer = deconvolution(_layer, [5, 5, _out, _in], [self.batch_size, _w0, _h0, _out], stride_1, self.ini_c)
            _logit = self.activation(_layer)
            self.x_decoder_mean = tf.nn.sigmoid(_logit)

        # Define loss function
        with tf.name_scope('loss'):
            loss_1 = reconstruction_loss(original=self.x, reconstruction=self.x_decoder_mean)
            loss_2 = latent_loss(self.z_mean, self.z_log_sigma_sq)
            self.loss = tf.reduce_mean(loss_1 + loss_2)  # average over batch

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train = slim.learning.create_train_op(self.loss, optimizer)
        # saver
        self.saver = tf.train.Saver()

    def reconstruct(self, inputs, label):
        """Reconstruct given data. """
        assert len(inputs) == self.batch_size
        assert len(label) == self.batch_size
        return self.sess.run(self.x_decoder_mean, feed_dict={self.x: inputs, self.y: label})

    def encode(self, inputs, label):
        """ Embed given data to latent vector. """
        return self.sess.run(self.z_mean, feed_dict={self.x: inputs, self.y: label})

    def decode(self, label, z=None):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is generated.
        Otherwise, z_mu is drawn from prior in latent space.
        """
        z = np.random.normal(size=self.network_architecture["n_z"]) if z is None else z
        return self.sess.run(self.x_decoder_mean, feed_dict={self.z: z, self.y: label})


if __name__ == '__main__':
    import os

    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    ConditionalVAE(10)
