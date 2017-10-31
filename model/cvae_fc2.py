import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
import tensorflow.contrib.slim as slim
import numpy as np


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
    latent_mean = tf.clip_by_value(latent_mean, clip_value_min=-1e-10, clip_value_max=1e+10)
    latent_log_sigma_sq = tf.clip_by_value(latent_log_sigma_sq, clip_value_min=-1e-10, clip_value_max=1e+10)
    return -0.5 * tf.reduce_sum(1 + latent_log_sigma_sq - tf.square(latent_mean) - tf.exp(latent_log_sigma_sq), 1)


class ConditionalVAE(object):
    """ Conditional VAE
    Inputs data must be normalized to be in range of 0 to 1
    (since VAE uses Bernoulli distribution for reconstruction loss)
    """

    def __init__(self, label_size, network_architecture=None, activation=tf.nn.softplus, learning_rate=0.001,
                 batch_size=100, save_path=None, load_model=None):
        """
        :param dict network_architecture: dictionary with following elements
            n_input: shape of input
            n_z: dimensionality of latent space
        :param float learning_rate: learning rate
        :param activation: activation function (tensor flow function)
        :param float learning_rate:
        :param int batch_size:
        """
        if network_architecture:
            self.network_architecture = network_architecture
        else:
            self.network_architecture = dict(n_hidden_encoder_1=500, n_hidden_encoder_2=500, n_hidden_decoder_1=500,
                                             n_hidden_decoder_2=500, n_input=784, n_z=20)

        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.label_size = label_size

        # Initializer
        if "relu" in self.activation.__name__:
            self.ini = variance_scaling_initializer()
        else:
            self.ini = xavier_initializer()

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
        self.x = tf.placeholder(tf.float32, [None, self.network_architecture["n_input"]], name="input")
        self.y = tf.placeholder(tf.float32, [None, self.label_size], name="output")

        # Build conditional input
        _layer = tf.concat([self.x, self.y], axis=1)

        # Encoder network to determine mean and (log) variance of Gaussian distribution in latent space
        with tf.variable_scope("encoder"):
            # full connected 1
            _layer = full_connected(_layer, [self.network_architecture["n_input"] + self.label_size,
                                             self.network_architecture["n_hidden_encoder_1"]], self.ini)
            _layer = self.activation(_layer)
            # full connected 2
            _layer = full_connected(_layer, [self.network_architecture["n_hidden_encoder_1"],
                                             self.network_architecture["n_hidden_encoder_2"]], self.ini)
            _layer = self.activation(_layer)
            # full connect to get "mean" and "sigma"
            self.z_mean = full_connected(_layer, [self.network_architecture["n_hidden_encoder_2"],
                                                  self.network_architecture["n_z"]], self.ini)
            self.z_log_sigma_sq = full_connected(_layer, [self.network_architecture["n_hidden_encoder_2"],
                                                          self.network_architecture["n_z"]], self.ini)

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
        # print(self.z.shape)
        _layer = tf.concat([self.z, self.y], axis=1)

        # Decoder to determine mean of Bernoulli distribution of reconstructed input
        with tf.variable_scope("decoder"):
            # full connected 1
            _layer = full_connected(_layer, [self.network_architecture["n_z"] + self.label_size,
                                             self.network_architecture["n_hidden_decoder_1"]], self.ini)
            _layer = self.activation(_layer)
            # full connected 2
            _layer = full_connected(_layer, [self.network_architecture["n_hidden_decoder_1"],
                                             self.network_architecture["n_hidden_decoder_2"]], self.ini)
            _layer = self.activation(_layer)
            # full connected 3 to output
            _logit = full_connected(_layer, [self.network_architecture["n_hidden_decoder_2"],
                                             self.network_architecture["n_input"]], self.ini)
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

    def decode(self, label, z=None, std=0.01, mu=0):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is generated.
        Otherwise, z_mu is drawn from prior in latent space.
        """
        z = mu + np.random.randn(self.batch_size, self.network_architecture["n_z"]) * std if z is None else z
        return self.sess.run(self.x_decoder_mean, feed_dict={self.z: z, self.y: label})


if __name__ == '__main__':
    import os

    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    ConditionalVAE(10)
