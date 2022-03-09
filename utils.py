import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull
import scipy.cluster.hierarchy as shc
import pandas as pd

import torch
import torchvision
from torchvision.utils import save_image, make_grid

from dataset import *
from visualization import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=501, help="number of epochs of training")
    parser.add_argument("--test_epoch", type=int, default=500, help="epoch to test model performance")

    parser.add_argument("--dataset_name", type=str, default="zenodo", help="name of the dataset, ewine or zenodo")
    parser.add_argument("--dataset_env", type=str, default='obstacle_full',
                        help="dataset (zenodo) of different environments,"
                             "choice for ewine is always los/nlos,"
                             "choices for zenodo include 'obstacle_full', 'obstacle_part1', 'obstacle_part2'")
    # parser.add_argument("--mode", type=str, default="full",
    #                     help="simulated mode train/test for data usage, paper or full")
    parser.add_argument("--split_factor", type=float, default=0.8, help="ratio to split data for training and testing")

    parser.add_argument("--batch_size", type=int, default=500, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

    parser.add_argument("--n_downsample", type=int, default=4, help="number downsampling layers in encoder")
    parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder/decoder")
    parser.add_argument("--dim", type=int, default=4, help="number of filters in first encoder layer")
    parser.add_argument("--env_dim", type=int, default=10, help="dimensionality of the env code")
    parser.add_argument("--range_dim", type=int, default=10, help="dimensionality of the range code")
    parser.add_argument("--conv_type", type=int, default=2, help="use 1 dim or 2 dim for convolution")
    # parser.add_argument("--if_expand", type=bool, default=False, help="Expand the cir signal to square or not")
    parser.add_argument("--regressor_type", type=str, default='Linear',
                        help="structure for regressor net: linear, conv1d, or conv2d")
    parser.add_argument("--classifier_type", type=str, default='Conv2d',
                        help="structure for classifier to estimate label")
    parser.add_argument("--use_soft", type=bool, default=False, help="estimate soft range estimation or hard one")

    parser.add_argument("--sample_interval", type=int, default=20, help="epoch interval saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="epoch interval between saving model checkpoint")

    return parser


def assign_train_test(opt, data_root):
    data_train, data_test = err_mitigation_dataset(
        root=data_root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env,
        split_factor=opt.split_factor, scaling=True
    )
    return data_train, data_test


def CDF_plot(err_arr, num=20, color='brown', marker='*'):

    data = np.abs(err_arr)
    pred_error_max = np.max(data)
    pred_error_cnt = np.zeros((num + 1,))
    step = pred_error_max / num

    # normalize to (0, 1) by dividing max
    for i in range(data.shape[0]):
        index = int(data[i] / step)
        pred_error_cnt[index] = pred_error_cnt[index] + 1
    pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)

    # accumulate error at each point to CDF
    CDF = np.zeros((num + 1,))
    for i in range(num + 1):
        if i == 0:
            CDF[i] = pred_error_cnt[i]
        else:
            CDF[i] = CDF[i-1] + pred_error_cnt[i]

    plt.plot(np.linspace(0, pred_error_max, num=num + 1), CDF, color=color, marker=marker)
    # plt.xlim((0, 0.6))
    plt.xlabel("Residual Range Error (m)")
    plt.ylabel("CDF")
    plt.show()


def CDF_plot2(err_arr, color='brown', marker="*"):

    # use absolute values
    err_arr = np.abs(err_arr)

    # sort the data
    err_sorted = np.sort(err_arr)

    # calculate the proportional values of samples
    p = 1. * np.arange(len(err_arr)) / (len(err_arr) - 1)

    # plot the sorted data
    fig, ax = plt.subplots()
    # ticks = [0.1, 0.5, 0.9]
    # ticklabels = [f'{t*100} %' for t in ticks]
    ax.plot(err_sorted, p, color=color, marker=marker)
    ax.set_xlabel("Residual Range Error (m)")
    ax.set_ylabel("CDF")
    # ax.yaxis.set_ticks(ticks)
    # ax.yaxis.set_ticklabels(ticklabels)
    # ax.yaxis.grid(True)
    # ax.set_ylim(0, 1)
    # plt.plot(err_sorted, p, color=color, marker=marker)
    # plt.xlabel("Residual Range Error (m)")
    # plt.ylabel("CDF")
    plt.show()


def DoubleY_plot(X, Y1, Y2, labelx="Learning Iterations", label1="Learning Curve 1", label2="Learning Curve 2",
                 title=None, color1="tab:blue", color2="tab:red", marker="o"):

    # Plot Line1 (Left Y Axis)
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
    ax1.plot(X, Y1, color=color1)

    # Plot Line2 (Right Y Axis)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(X, Y2, color=color2)

    # Decorations
    # ax1 (left Y axis)
    ax1.set_xlabel(labelx, fontsize=20)
    ax1.tick_params(axis='x', rotation=0, labelsize=12)
    ax1.set_ylabel(label1, color=color1, fontsize=20)
    ax1.tick_params(axis='y', rotation=0, labelcolor=color1)

    # ax2 (right Y axis)
    ax2.set_ylabel(label2, color=color2, fontsize=20)
    ax2.tick_params(axis='y', labelcolor=color2)

    # grid for x
    ax2.set_xticks(np.arange(0, len(X), 60))
    ax2.set_xticklabels(X[::60], rotation=90, fontdict={'fontsize': 10})
    if title is not None:
        ax2.set_title(title, fontsize=22)
    fig.tight_layout()
    plt.show()


def Dendrogram_plot(data, labels, title=None):

    # Plot figure
    plt.figure(figsize=(16, 10), dpi=80)
    if title is not None:
        plt.title(title, fontsize=22)
    data = shc.linkage(data, 'ward')
    # print(data.shape)
    dend = shc.dendrogram(data, labels=labels, color_threshold=100)
    # data = [env_code], labels = [olabel/rlabel]
    plt.xticks(fontsize=12)
    plt.show()


def AggCluster_plot(data, n_cluster=5, title=None):
    """Temporally not used since the illustration on the first 2 dims is not that reasonable,
    each dim in our case (i.e. env_code) does not have specific meaning.
    Maybe can serve as a sequential step after umap."""
    # Convert array to dataframe
    # e.g., data = [[1, 2, 3, 4], ...], label = ['amy', ...]
    df_label = ['Dim1', 'Dim2']
    df = pd.DataFrame(data, columns=df_label)

    # Agglomerative Clustering
    cluster = AgglomerativeClustering(n_clusters=n_cluster, affinity='euclidean', linkage='ward')
    cluster.fit_predict(df)

    # Plot with the first two dims
    plt.figure(figsize=(14, 10), dpi=80)
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=cluster.labels_, cmap='tab10')

    # Encircle
    def encircle(x, y, ax=None, **kw):
        if not ax: ax=plt.gca()
        p = np.c_[x, y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices, :], **kw)
        ax.add_patch(poly)

    # Draw polygon surrounding vertices in the first 2 dims
    for i in range(n_cluster):
        c_list = ["gold", "tab:blue", "tab:red", "tab:green", "tab:orange", "tab:yellow", "brown"]
        encircle(df.loc[cluster.labels_ == i, df_label[0]], df.loc[cluster.labels_ == i, df_label[1]],
                 ec="k", fc=c_list[i], alpha=0.2, linewidth=0)

    # Decorations
    plt.xlabel('Dim One'); plt.xticks(fontsize=12)
    plt.ylabel('Dim Two'); plt.yticks(fontsize=12)
    # plt.legend([label_str_dict[item] for item in label_str_dict])
    if title is not None:
        plt.title(title, fontsize=22)
    plt.show()


if __name__ == '__main__':

    # Load ewine data
    root = ['./data/data_ewine/dataset1/',
            './data/data_ewine/dataset2'
    ]
    data_train, data_test = err_mitigation_dataset(
        root=root, dataset_name="ewine", split_factor=0.8, scaling=True
    )

    # # 1 Test CDF plots
    # data = np.random.randn(1000)
    # CDF_plot2(data)
    #
    # CDF_plot(data, num=10)
    #
    # # 2 Test cir data
    # test_cir, test_err, test_label = data_test
    # x = np.arange(0, 300)
    # y1 = test_err[0:300]
    # y2 = test_label[0:300]
    #
    # DoubleY_plot(x, y1, y2)
    #
    # # 3 Test dendrogram
    # data = [
    #     [12, 10], [15, 12], [17, 16], [82, 23], [89, 1]
    # ]
    # labels = ['andy', 'tom', 'amy', 'leo', 'mary']
    # Dendrogram_plot(data, labels)
    #
    # data2 = test_cir[:10][:100]  # latents
    # labels2 = test_label[:10]
    # for label in labels2:  # [0] -> 0, sum(label)
    #     print(label[0])
    # labels2_str = [label_int2str("ewine", label[0]) for label in labels2]
    # Dendrogram_plot(data2, labels2_str)

    # 4 Test AggCluster
    data = [
        [12, 1], [15, 12], [17, 16], [82, 23], [89, 1],
        [12, 134], [145, 1], [27, 15], [81, 3], [8, 21],
        [1, 14], [45, 100], [2, 16], [1, 32], [84, 2]
    ]
    # labels = ['dim1', 'dim2']
    AggCluster_plot(data, n_cluster=2, title=None)

    cir_arr, err_arr, label_arr = data_test
    features_arr = reduce_latents(cir_arr, method="t-sne", n_components=2)
    # labels2 = label_arr
    # The defficiency is no actual env label noted, but can tell if distinguishable
    # labels2 = ['dim1', 'dim2']
    data2 = features_arr  # 2D
    AggCluster_plot(data2, n_cluster=2, title=None)

