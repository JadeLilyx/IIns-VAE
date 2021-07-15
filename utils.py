import torch
import torchvision
from torchvision.utils import save_image, make_grid

import argparse
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import umap

# from baseline import svm_regressor  # for CDF comparison
from data_tools import label_dictionary
from dataset import err_mitigation_dataset


def get_args(parser):
    parser = argparse.ArgumentParser()
    # learning setting
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--test_epoch", type=int, default=500, help="epoch to test model performance")
    
    # optimization parameters
    parser.add_argument("--batch_size", type=int, default=500, help="size of the batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order moment of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order moment of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to load data")

    # network choice
    parser.add_argument("--net_ablation", type=str, default="loop", help="choices: loop, loops")  # detach

    # network structure
    parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks")
    parser.add_argument("--n_downsample", type=int, default=4, help="number of downsampling layers")
    parser.add_argument("--filters", type=int, default=16, help="number of filters in first encoder layer")
    parser.add_argument("--env_dim", type=int, default=16, help="dimension of environment code")
    parser.add_argument("--use_soft", type=bool, default=False, help="estimate soft range information")
    parser.add_argument("--identifier_type", type=int, default=1, help="structure for identifier for label, 1 for linear, 2 for conv1d, and 3 for conv2d")
    parser.add_argument("--regressor_type", type=int, default=1, help="structure for regressor for ranging error, 1 for linear, 2 for conv1d, 3 for conv2d")
    
    # data choices
    parser.add_argument("--dataset_name", type=str, default="zenodo", help="name of the dataset for usage")
    parser.add_argument("--dataset_env", type=str, default="nlos", help=" different environment options for zenodo dataset")
    parser.add_argument("--mode", type=str, default="full", help="mode to assign train and test data, have additional 'paper' option for zenodo set")
    parser.add_argument("--split_factor", type=float, default=0.8, help="factor to split train and test data")

    # check intervals
    parser.add_argument("--sample_interval", type=int, default=20, help="epoch interval between saving generated samples")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="epoch interval between saving model checkpoint")

    return parser


# --------------- Visualize Figures ---------------------


def reduce_latents(env_latent, labels):

    latents = env_latent.view(env_latent.size(0), -1)
    latents = latents.cpu().numpy()
    labels = labels.cpu().numpy()
    
    if latents.shape[1] > 2:
        reduced_latents = umap.UMAP().fit_transform(latents)
    else:
        reduced_latents = latents
    
    return reduced_latents, labels


def visualize_latents(features_arr, labels_arr, save_path, epoch, dataset_env='nlos', title=None):

    data_module = dict()
    labels_list = labels_arr.tolist()
    reduced_latents_list = features_arr.tolist()
    
    classes = set()
    for i in range(len(labels_list)):
        label_item = labels_list[i][0]
        if label_item not in data_module:
            data_module[label_item] = list()
            classes.add(label_item)
        data_module[label_item].append(reduced_latents_list[i])

    for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
        class_features = np.asarray(data_module[cls])
        label_str_dict = label_dictionary(dataset_env)
        plt.scatter(class_features[:, 0], class_features[:, 1], c=[color], label=label_str_dict[cls], s=[2], alpha=0.5)

    if title is not None:
        plt.set_title(title)
    plt.legend([label_str_dict[item] for item in label_str_dict])
    plt.savefig(os.path.join(save_path, "latent_env_epoch%d.png" % epoch))
    plt.close()


def CDF_plot(err_arr, num=200, color='brown'):
    
    data = np.abs(err_arr)
    blocks_num = num
    pred_error_max = np.max(data)
    pred_error_cnt = np.zeros((blocks_num + 1,))
    step = pred_error_max / blocks_num

    # normalize to (0, 1) by dividing max
    for i in range(data.shape[0]):
        index = int(data[i] / step)
        pred_error_cnt[index] = pred_error_cnt[index] + 1
    pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)

    # accumulate error at each point to CDF
    CDF = np.zeros((blocks_num + 1,))
    for i in range(blocks_num + 1):
        if i == 0:
            CDF[i] = pred_error_cnt[i]
        else:
            CDF[i] = CDF[i - 1] + pred_error_cnt[i]

    plt.plot(np.linspace(0, pred_error_max, num=blocks_num + 1), CDF, color=color)
    plt.xlim((0, 0.6))


def assign_train_test(opt, root):
    data_train, data_test, _, _ = err_mitigation_dataset(
        root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env, split_factor=opt.split_factor,
        scaling=True, mode=opt.mode, feature_flag=False
    )
    return data_train, data_test