import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d, variance_scaling_initializer, xavier_initializer
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
    latent_log_sigma_sq = tf.clip_by_value(latent_log_sigma_sq, clip_value_min=-1e-10, clip_value_max=1e+2)
    return -0.5 * tf.reduce_sum(1 + latent_log_sigma_sq - tf.square(latent_mean) - tf.exp(latent_log_sigma_sq), 1)


class VariationalAutoencoder(object):
    """ Variational Autoencoder
    - Encoder: input (1d vector) -> FC x 3 -> latent
    - Decoder: latent -> FC x 3 -> output (1d vector)
    """

    def __init__(self, network_architecture, activation=tf.nn.softplus, max_grad_norm=None,
                 learning_rate=0.001, batch_size=100, save_path=None, load_model=None):
        """
        :param dict network_architecture: dictionary with following elements
            n_hidden_encoder_1: 1st layer encoder neurons
            n_hidden_encoder_2: 2nd layer encoder neurons
            n_hidden_decoder_1: 1st layer decoder neurons
            n_hidden_decoder_2: 2nd layer decoder neurons
            n_input: shape of input
            n_z: dimensionality of latent space

        :param activation: activation function (tensor flow function)
        :param float learning_rate:
        :param int batch_size:
        """
        self.network_architecture = network_architecture
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

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

        # Encoder network to determine mean and (log) variance of Gaussian distribution in latent space
        with tf.variable_scope("encoder"):
            # full connected 1
            _layer = full_connected(self.x, [self.network_architecture["n_input"],
                                             self.network_architecture["n_hidden_encoder_1"]], self.ini)
            _layer = self.activation(_layer)
            # full connected 2
            _layer = full_connected(_layer, [self.network_architecture["n_hidden_encoder_1"],
                                             self.network_architecture["n_hidden_encoder_2"]], self.ini)
            _layer = self.activation(_layer)
            # full connect to get "mean" and "sigma"
            self.z_mean = full_connected(_layer, [self.network_architecture["n_hidden_encoder_2"],
                                                  self.network_architecture["n_z"]], self.ini)
            # self.z_mean = tf.where(tf.is_inf(z_mean), 0.0, z_mean)
            self.z_log_sigma_sq = full_connected(_layer, [self.network_architecture["n_hidden_encoder_2"],
                                                          self.network_architecture["n_z"]], self.ini)

        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Decoder to determine mean of Bernoulli distribution of reconstructed input
        with tf.variable_scope("decoder"):
            # full connected 1
            _layer = full_connected(self.z, [self.network_architecture["n_z"],
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
            self.re_loss = tf.reduce_mean(reconstruction_loss(original=self.x, reconstruction=self.x_decoder_mean))
            self.latent_loss = tf.reduce_mean(latent_loss(self.z_mean, self.z_log_sigma_sq))
            self.loss = tf.where(tf.is_nan(self.re_loss), 0.0, self.re_loss) + \
                tf.where(tf.is_nan(self.latent_loss), 0.0, self.latent_loss)

        # Define optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        if self.max_grad_norm:
            _var = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, _var), self.max_grad_norm)
            self.train = optimizer.apply_gradients(zip(grads, _var))
        else:
            self.train = optimizer.minimize(self.loss)

        # saver
        self.saver = tf.train.Saver()

    def reconstruct(self, inputs):
        """Reconstruct given data. """
        assert len(inputs) == self.batch_size
        return self.sess.run(self.x_decoder_mean, feed_dict={self.x: inputs})

    def encode(self, inputs):
        """ Embed given data to latent vector. """
        return self.sess.run(self.z_mean, feed_dict={self.x: inputs})

    def decode(self, z=None, std=0.01, mu=0):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is generated.
        Otherwise, z_mu is drawn from prior in latent space.
        """
        z = mu + np.random.randn(self.batch_size, self.network_architecture["n_z"]) * std if z is None else z
        return self.sess.run(self.x_decoder_mean, feed_dict={self.z: z})


if __name__ == '__main__':
    import os
    print(VariationalAutoencoder.__doc__)

    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    VariationalAutoencoder({
  "n_hidden_encoder_1": 500,
  "n_hidden_encoder_2": 500,
  "n_hidden_decoder_1": 500,
  "n_hidden_decoder_2": 500,
  "n_input":784,
  "n_z": 20
})
