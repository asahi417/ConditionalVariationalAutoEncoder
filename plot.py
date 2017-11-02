import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from util import mnist_loader, shape_2d


def get_parameter(path, latent_dim):
    with open(path) as f:
        p = json.load(f)
    if latent_dim:
        p["n_z"] = latent_dim
    return p


def generate_image_random(model, feeder, save_path=None, mean_num=500, target_digit=9, std=0.1,
                          input_image=False):
    # generate latent vector
    _code = []
    for i in range(mean_num):
        _x, _y = feeder.test.next_batch(model.batch_size)
        _x = shape_2d(_x, model.batch_size) if input_image else _x
        __code = model.encode(_x, _y).tolist()
        for __x, __y, _c in zip(_x, _y, __code):
            if np.argmax(__y) == target_digit:
                _code.append(_c)

    # label
    o_h = np.zeros(model.label_size)
    o_h[target_digit] = 1
    true_label = np.tile(o_h, [model.batch_size, 1])

    z = np.tile(np.mean(_code, 0), [model.batch_size, 1])
    z += np.random.randn(model.batch_size, model.network_architecture["n_z"]) * std

    generated = model.decode(true_label, z)
    plt.figure(figsize=(6, 10))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.imshow(generated[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Generated %i" % np.argmax(true_label[i]))
        plt.colorbar()
        plt.tight_layout()
    if save_path:
        plt.savefig("%sgenerated_image_rand_%i_%0.3f.eps" % (save_path, target_digit, std), bbox_inches="tight")
        plt.savefig("%sgenerated_image_rand_%i_%0.3f.png" % (save_path, target_digit, std), bbox_inches="tight")


def generate_image_mean(model, feeder, save_path=None, mean_num=500, input_image=False):

    _code = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    for i in range(mean_num):
        _x, _y = feeder.test.next_batch(model.batch_size)
        _x = shape_2d(_x, model.batch_size) if input_image else _x
        __code = model.encode(_x, _y).tolist()
        for __x, __y, _c in zip(_x, _y, __code):
            _code[int(np.argmax(__y))].append(_c)

    # convert label to one hot vector
    o_h = np.eye(model.label_size)[[i for i in range(model.label_size)]]
    true_label = np.tile(o_h, [int(model.batch_size / model.label_size), 1])

    tmp = np.vstack([np.mean(_code[_a], 0) for _a in range(model.label_size)])
    z = np.tile(tmp, [int(model.batch_size / model.label_size), 1])

    generated = model.decode(true_label, z)
    plt.figure(figsize=(6, 10))
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.imshow(generated[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Generated %i" % np.argmax(true_label[i]))
        plt.colorbar()
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "generated_image_mean.eps", bbox_inches="tight")
        plt.savefig(save_path + "generated_image_mean.png", bbox_inches="tight")


def plot_reconstruct(model, _mode, feeder, _n=5, save_path=None, input_image=False):
    # feed test data and reconstruct
    _x, _y = feeder.test.next_batch(model.batch_size)
    _x = shape_2d(_x, model.batch_size) if input_image else _x
    if _mode == "conditional":
        reconstruction = model.reconstruct(_x, _y)
    elif _mode == "unsupervised":
        reconstruction = model.reconstruct(_x)
    # plot
    plt.figure(figsize=(8, 12))
    for i in range(_n):
        plt.subplot(_n, 2, 2 * i + 1)
        plt.imshow(_x[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input %i" % np.argmax(_y[i]))
        plt.colorbar()
        plt.subplot(_n, 2, 2 * i + 2)
        plt.imshow(reconstruction[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path + "reconstruction.eps", bbox_inches="tight")
        plt.savefig(save_path + "reconstruction.png", bbox_inches="tight")


if __name__ == '__main__':
    # Ignore warning message by tensor flow
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # Parser
    parser = argparse.ArgumentParser(description='This script is ...')
    parser.add_argument('model', action='store', nargs=None, const=None, default=None, type=str, choices=None,
                        help='name of model', metavar=None)
    parser.add_argument('-n', '--latent_dim', action='store', nargs='?', const=None, default=20, type=int,
                        choices=None, help='latent dimension', metavar=None)
    parser.add_argument('-p', '--progress', action='store', nargs='?', const=None, default=None, type=str,
                        choices=None, help='progress model', metavar=None)
    parser.add_argument('-s', '--std', action='store', nargs='?', const=None, default=0.1, type=float,
                        choices=None, help='std of random generated data', metavar=None)
    parser.add_argument('-t', '--target', action='store', nargs='?', const=None, default=0, type=int,
                        choices=None, help='target of random generated data', metavar=None)
    args = parser.parse_args()

    print("\n Plot the result of %s \n" % args.model)
    pr = "progress-%s-" % args.progress if args.progress else ""
    acc = np.load("./log/%s_%i/%sacc.npz" % (args.model, args.latent_dim, pr))
    param = get_parameter("./parameter/%s.json" % args.model, args.latent_dim)
    opt = dict(network_architecture=param, batch_size=acc["batch_size"])

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

    model_instance = Model(load_model="./log/%s_%i/%smodel.ckpt" % (args.model, args.latent_dim, pr), **opt)
    mnist, size = mnist_loader()

    fig_path = "./figure/%s_%i/" % (args.model, args.latent_dim)
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)

    plot_reconstruct(model_instance, _mode, mnist, 5, input_image=_inp_img, save_path=fig_path)
    if _mode == "conditional":
        generate_image_mean(model_instance, mnist, input_image=_inp_img, save_path=fig_path)
        generate_image_random(model_instance, mnist, input_image=_inp_img, save_path=fig_path, std=args.std,
                              target_digit=args.target)


