import os
import tensorflow as tf
from util import mnist_train
from model import ConditionalVAE

if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    save_path = "./log/cvae/"
    network_architecture = dict(n_input=[28, 28, 1], n_z=20)
    model = ConditionalVAE(10, network_architecture=network_architecture, batch_size=100, save_path=save_path,
                           learning_rate=0.0001, activation=tf.nn.relu)
    mnist_train(model=model, epoch=75, save_path=save_path, mode="conditional", input_image=True)

    # label_size, network_architecture = None, activation = tf.nn.relu,
    # learning_rate = 0.001, batch_size = 100, save_path = None, load_model = None
