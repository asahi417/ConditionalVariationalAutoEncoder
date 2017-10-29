import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


def _initialize_weights(initializer, n_hidden_encoder_1, n_hidden_encoder_2,
                        n_hidden_decoder_1, n_hidden_decoder_2, n_input, n_z):
    all_weights = dict()
    all_weights['weights_encoder'] = {
        'h1': tf.Variable(initializer(shape=(n_input, n_hidden_encoder_1))),
        'h2': tf.Variable(initializer(shape=(n_hidden_encoder_1, n_hidden_encoder_2))),
        'out_mean': tf.Variable(initializer(shape=(n_hidden_encoder_2, n_z))),
        'out_log_sigma': tf.Variable(initializer(shape=(n_hidden_encoder_2, n_z)))}
    all_weights['biases_encoder'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_encoder_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_encoder_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
    all_weights['weights_decoder'] = {
        'h1': tf.Variable(initializer(shape=(n_z, n_hidden_decoder_1))),
        'h2': tf.Variable(initializer(shape=(n_hidden_decoder_1, n_hidden_decoder_2))),
        'out_mean': tf.Variable(initializer(shape=(n_hidden_decoder_2, n_input))),
        'out_log_sigma': tf.Variable(initializer(shape=(n_hidden_decoder_2, n_input)))}
    all_weights['biases_decoder'] = {
        'b1': tf.Variable(tf.zeros([n_hidden_decoder_1], dtype=tf.float32)),
        'b2': tf.Variable(tf.zeros([n_hidden_decoder_2], dtype=tf.float32)),
        'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
        'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
    return all_weights


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
    latent_log_sigma_sq = tf.clip_by_value(latent_log_sigma_sq, clip_value_min=-50, clip_value_max=50)
    return -0.5 * tf.reduce_sum(1 + latent_log_sigma_sq - tf.square(latent_mean) - tf.exp(latent_log_sigma_sq), 1)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE)
    Inputs data must be normalized to be in range of 0 to 1
    (since VAE uses Bernoulli distribution for reconstruction loss)
    """

    def __init__(self, network_architecture=None, activation=tf.nn.softplus,
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
        if network_architecture is None:
            self.network_architecture = dict(n_hidden_encoder_1=500, n_hidden_encoder_2=500, n_hidden_decoder_1=500,
                                             n_hidden_decoder_2=500, n_input=784, n_z=20)
        else:
            self.network_architecture = network_architecture
        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size

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

        if load_model:
            tf.reset_default_graph()
            self.saver.restore(self.sess, load_model)

    def _create_network(self):
        """ Create Network, Define Loss Function and Optimizer """
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.network_architecture["n_input"]], name="input")

        # Initialize auto-encoder weights and biases
        if "relu" in self.activation.__name__:
            initializer = tf.contrib.layers.variance_scaling_initializer()
        else:
            initializer = tf.contrib.layers.xavier_initializer(uniform=True)
        _weights = _initialize_weights(initializer, **self.network_architecture)

        # Encoder network to determine mean and (log) variance of Gaussian distribution in latent space
        with tf.variable_scope("encoder"):
            _layer = tf.add(tf.matmul(self.x, _weights["weights_encoder"]['h1']), _weights["biases_encoder"]['b1'])
            _layer = self.activation(_layer)
            _layer = tf.add(tf.matmul(_layer, _weights["weights_encoder"]['h2']), _weights["biases_encoder"]['b2'])
            _layer = self.activation(_layer)
            self.z_mean = tf.add(tf.matmul(_layer, _weights["weights_encoder"]['out_mean']),
                                 _weights["biases_encoder"]['out_mean'])
            self.z_log_sigma_sq = tf.add(tf.matmul(_layer, _weights["weights_encoder"]['out_log_sigma']),
                                         _weights["biases_encoder"]['out_log_sigma'])
        # Draw one sample z from Gaussian distribution
        eps = tf.random_normal((self.batch_size, self.network_architecture["n_z"]), mean=0, stddev=1, dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Decoder to determine mean of Bernoulli distribution of reconstructed input
        with tf.variable_scope("decoder"):
            _layer = tf.add(tf.matmul(self.z, _weights["weights_decoder"]['h1']), _weights["biases_decoder"]['b1'])
            _layer = self.activation(_layer)
            _layer = tf.add(tf.matmul(_layer, _weights["weights_decoder"]['h2']), _weights["biases_decoder"]['b2'])
            _layer = self.activation(_layer)
            _layer = tf.add(tf.matmul(_layer, _weights["weights_decoder"]['out_mean']),
                            _weights["biases_decoder"]['out_mean'])
            self.x_decoder_mean = tf.nn.sigmoid(_layer)

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

    def reconstruct(self, inputs):
        """Reconstruct given data. """
        assert len(inputs) == self.batch_size
        return self.sess.run(self.x_decoder_mean, feed_dict={self.x: inputs})

    def encode(self, inputs):
        """ Embed given data to latent vector. """
        return self.sess.run(self.z_mean, feed_dict={self.x: inputs})

    def decode(self, z=None):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is generated.
        Otherwise, z_mu is drawn from prior in latent space.
        """
        z = np.random.normal(size=self.network_architecture["n_z"]) if z is None else z
        return self.sess.run(self.x_decoder_mean, feed_dict={self.z: z})

