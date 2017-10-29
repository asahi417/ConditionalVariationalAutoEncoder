import os
import tensorflow as tf
from util import mnist_train
from model import CNN

if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    save_path = "./log/cnn/"
    model = CNN(batch_size=100, save_path=save_path, learning_rate=0.0005, keep_prob=0.8, activation=tf.nn.relu)
    mnist_train(model=model, epoch=75, save_path=save_path, mode="supervised", input_image=True)
