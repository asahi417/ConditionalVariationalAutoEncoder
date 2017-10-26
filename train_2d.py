import os
from util import mnist_train
from model_vae.model import VariationalAutoencoder

if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    save_path = "./log/vae_2d/"
    network_architecture = dict(n_hidden_encoder_1=500, n_hidden_encoder_2=500, n_hidden_decoder_1=500,
                                n_hidden_decoder_2=500, n_input=784, n_z=2)
    model = VariationalAutoencoder(network_architecture=network_architecture, batch_size=100, save_path=save_path)
    mnist_train(model=model, epoch=75, save_path=save_path)
