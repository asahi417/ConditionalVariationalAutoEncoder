import os
from util import mnist_train_2
from model_cvae.model import ConditionalVAE

if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    save_path = "./log/cvae/"
    network_architecture = dict(n_input=[28, 28, 1], n_z=20)
    model = ConditionalVAE(10, network_architecture=network_architecture, batch_size=100, save_path=save_path)
    mnist_train_2(model=model, epoch=75, save_path=save_path)
