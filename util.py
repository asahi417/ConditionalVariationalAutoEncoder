import logging
import os
import numpy as np
import tensorflow as tf


def create_log(name):
    """Logging."""
    if os.path.exists(name):
        os.remove(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # handler for logger file
    handler1 = logging.FileHandler(name)
    handler1.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    # handler for standard output
    handler2 = logging.StreamHandler()
    handler2.setFormatter(logging.Formatter("H1, %(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def mnist_loader():
    from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    mnist = read_data_sets('MNIST_data', one_hot=True)
    n_sample = mnist.train.num_examples
    return mnist, n_sample


def mnist_train(model, epoch, save_path="./"):
    """Train model based on mini-batch of input data."""
    # load mnist
    data, n = mnist_loader()
    # train
    n_iter = int(n / model.batch_size)
    # logger
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    logger = create_log(save_path+"log")
    logger.info("train: data size(%i), batch num(%i), batch size(%i)" % (n, n_iter, model.batch_size))
    loss_mean = []
    # Initializing the tensor flow variables
    model.sess.run(tf.global_variables_initializer())
    for _e in range(epoch):
        _loss_mean = []
        for _b in range(n_iter):
            _x, _ = data.train.next_batch(model.batch_size)
            feed_dict = {model.x: _x}
            summary, loss, _ = model.sess.run([model.summary, model.loss, model.train], feed_dict=feed_dict)
            model.writer.add_summary(summary, int(_b + _e * model.batch_size))
            _loss_mean.append(loss)
        loss_mean.append(np.array(_loss_mean).mean(0))
        logger.info("epoch %i: loss (%0.3f)" % (_e, loss_mean[-1]))
        if _e % 50 == 0:
            model.saver.save(model.sess, "%s/progress-%i-model.ckpt" % (save_path, _e))
            np.save("%s/progress-%i-acc.npy" % (save_path, _e), np.array(loss_mean))
    model.saver.save(model.sess, "%s/model.ckpt" % save_path)
    np.savez("%s/acc.npz" % save_path, loss=np.array(loss_mean), learning_rate=model.learning_rate, epoch=epoch,
             batch_size=model.batch_size)
