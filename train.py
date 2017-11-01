import argparse
import os
import sys
import json
from util import mnist_train

parser = argparse.ArgumentParser(description='This script is ...')

parser.add_argument('model', action='store', nargs=None, const=None, default=None, type=str, choices=None,
                    help='name of model', metavar=None)

parser.add_argument('-n', '--latent_dim', action='store', nargs='?', const=None, default=20, type=int,
                    choices=None, help='latent dimension', metavar=None)

parser.add_argument('-b', '--batch_size', action='store', nargs='?', const=None, default=100, type=int,
                    choices=None, help='batch size', metavar=None)

parser.add_argument('-e', '--epoch', action='store', nargs='?', const=None, default=150, type=int,
                    choices=None, help='epoch', metavar=None)

parser.add_argument('-l', '--lr', action='store', nargs='?', const=None, default=0.005, type=float,
                    choices=None, help='learning rate', metavar=None)

parser.add_argument('-c', '--clip', action='store', nargs='?', const=None, default=None, type=float,
                    choices=None, help='gradient clipping', metavar=None)


def get_parameter(path, latent_dim):
    with open(path) as f:
        p = json.load(f)
    if latent_dim:
        p["n_z"] = latent_dim
    return p


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    args = parser.parse_args()
    print("\n Start train %s \n" % args.model)
    save_path = "./log/%s_%i/" % (args.model, args.latent_dim)
    param = get_parameter("./parameter/%s.json" % args.model, args.latent_dim)
    opt = dict(network_architecture=param, batch_size=args.batch_size, learning_rate=args.lr, save_path=save_path,
               max_grad_norm=args.clip)

    if args.model == "cvae_cnn3_0":
        from model import CvaeCnn3_0 as Model
        _mode, _inp_img = "conditional", True
    elif args.model == "cvae_cnn3_1":
        from model import CvaeCnn3_1 as Model
        _mode, _inp_img = "conditional", True
    elif args.model == "cvae_fc2":
        from model import CvaeFc2 as Model
        _mode, _inp_img = "conditional", False
    elif args.model == "vae":
        from model import VAE as Model
        _mode, _inp_img = "unsupervised", False
    else:
        sys.exit("unknown model !")

    if _mode == "conditional":
        opt["label_size"] = 10
    print(Model.__doc__)
    model = Model(**opt)
    mnist_train(model=model, epoch=args.epoch, save_path=save_path, mode=_mode, input_image=_inp_img)


