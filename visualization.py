import numpy as np
import os
import umap
from sklearn import manifold
from scipy.spatial import ConvexHull
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch

from dataset import *


def reduce_latents(latents, method="umap", n_components=2):
    # latents = env_latent.view(env_latent.size(0), -1)
    # latents = latents.cpu().numpy()
    # labels = labels.cpu().numpy()

    if latents.shape[1] > n_components:
        if method == "umap":
            e_umap = umap.UMAP(n_components=n_components)
            reduced_latents = e_umap.fit_transform(latents)
        elif method == "t-sne":
            e_tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
            reduced_latents = e_tsne.fit_transform(latents)
        else:
            raise ValueError("Unknown dimension reduction method %s..." % method)
    else:
        reduced_latents = latents

    return reduced_latents


def visualize_latents(features_arr, labels_arr, save_path, epoch,
                      dataset_name='zenodo', dataset_env='obstacle_full', level='room',
                      method="umap", n_components=2, title=None):
    # match labels and features to lists
    data_module = dict()
    # print("label array: ", labels_arr)  # notice obs and room labels
    labels_list = labels_arr.tolist()
    reduced_latents_list = features_arr.tolist()

    classes = set()
    for i in range(len(labels_list)):
        label_item = labels_list[i][0]
        if label_item not in data_module:
            data_module[label_item] = list()
            classes.add(label_item)
        data_module[label_item].append(reduced_latents_list[i])

    if n_components == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        for cls, color in zip(classes, plt.cm.get_cmap('tab20').colors):
            # print("int label: %d" % cls)
            class_features = np.asarray(data_module[cls])
            label_str_dict = label_dictionary(dataset_name, dataset_env, level)
            # fig, ax = plt.subplots()
            # print("label: %d-%s" % (cls, label_str_dict[cls]))
            ax.scatter(class_features[:, 0], class_features[:, 1], c=[color],
                       label=label_str_dict[cls], s=[2], alpha=0.5)

    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
            class_features = np.asarray(data_module[cls])
            label_str_dict = label_dictionary(dataset_name, dataset_env, level)
            ax.scatter(class_features[:, 0], class_features[:, 1], class_features[:, 2], c=[color],
                       label=label_str_dict[cls], s=[2], alpha=0.5)
            # ax.view_init(4, -72)

    if title is not None:
        plt.title(title)
    plt.legend([label_str_dict[item] for item in label_str_dict],
               loc="upper right", title="Environment Labels")
    plt.savefig(os.path.join(save_path, "latent_data%s-%s_method%s-%dD_epoch%d.png"
                             % (dataset_name, level, method, n_components, epoch)))
    plt.savefig(os.path.join(save_path, "latent_data%s-%s_method%s-%dD_epoch%d.pdf"
                             % (dataset_name, level, method, n_components, epoch)))
    plt.show()
    plt.close()


def visualize_latents_encircle(features_arr, labels_arr, cir_labels_arr, save_path, epoch,
                               dataset_name='zenodo', dataset_env='obstacle_full', level='room',
                               method="umap", n_components=2, encircle=True, title=None):
    # match labels and features to lists
    data_module = dict()
    # print("label array: ", labels_arr)  # notice obs and room labels
    labels_list = labels_arr.tolist()
    reduced_latents_list = features_arr.tolist()

    classes = set()
    for i in range(len(labels_list)):
        label_item = labels_list[i][0]
        if label_item not in data_module:
            data_module[label_item] = list()
            classes.add(label_item)
        data_module[label_item].append(reduced_latents_list[i])

    # encircle with cir_label_arr
    def encircle(x, y, ax=None, **kw):
        if not ax: ax=plt.gca()
        p = np.c_[x, y]
        hull = ConvexHull(p)
        poly = plt.Polygon(p[hull.vertices, :], **kw)
        ax.add_patch(poly)

    if n_components == 2:
        fig = plt.figure()
        ax = fig.add_subplot()
        for cls, color in zip(classes, plt.cm.get_cmap('tab20').colors):
            # print("int label: %d" % cls)
            class_features = np.asarray(data_module[cls])
            label_str_dict = label_dictionary(dataset_name, dataset_env, level)
            # fig, ax = plt.subplots()
            # print("label: %d-%s" % (cls, label_str_dict[cls]))
            ax.scatter(class_features[:, 0], class_features[:, 1], c=[color],
                       label=label_str_dict[cls], s=[2], alpha=0.5)
        # encircle data points according to cir_labels_arr
        if encircle:
            encircle(class_features[:, 0], class_features[:, 1], ec="k", fc="gold", alpha=0.1)
            encircle(class_features[:, 0], class_features[:, 1], ec="firebrick", fc="none", linewidth=1.5)

    elif n_components == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
            class_features = np.asarray(data_module[cls])
            label_str_dict = label_dictionary(dataset_name, dataset_env, level)
            ax.scatter(class_features[:, 0], class_features[:, 1], class_features[:, 2], c=[color],
                       label=label_str_dict[cls], s=[2], alpha=0.5)
            # ax.view_init(4, -72)

    if title is not None:
        plt.title(title)
    plt.legend([label_str_dict[item] for item in label_str_dict],
               loc="upper right", title="Environment Labels")
    plt.savefig(os.path.join(save_path, "latent_data%s-%s_method%s-%dD_epoch%d.png"
                             % (dataset_name, level, method, n_components, epoch)))
    plt.savefig(os.path.join(save_path, "latent_data%s-%s_method%s-%dD_epoch%d.pdf"
                             % (dataset_name, level, method, n_components, epoch)))
    plt.show()
    plt.close()


def visualize_data(result_path, data_test, method="umap", n_components=2,
                   dataset_name="ewine", dataset_env="nlos"):
    if dataset_name == 'ewine':

        cir_arr, err_arr, label_arr = data_test
        features_arr = reduce_latents(cir_arr, method=method, n_components=n_components)

        visualize_latents(features_arr, label_arr, result_path, epoch=00,
                          dataset_name=dataset_name,  # neglect env and room
                          method=method, n_components=n_components)

    elif dataset_name == 'zenodo':

        cir_arr, err_arr, olabel_arr, rlabel_arr = data_test
        features_arr = reduce_latents(cir_arr, method=method, n_components=n_components)

        visualize_latents(features_arr, olabel_arr, result_path, epoch=00,
                          dataset_name=dataset_name, dataset_env=dataset_env, level="obstacle",
                          method=method, n_components=n_components)
        visualize_latents(features_arr, rlabel_arr, result_path, epoch=00,
                          dataset_name=dataset_name, dataset_env=dataset_env, level="room",
                          method=method, n_components=n_components)

    else:
        raise ValueError("Unknown dataset %s." % dataset_name)


def visualize_data_batch(result_path, dataloader_test, method="umap", n_components=2,
                         dataset_name="ewine", dataset_env="nlos"):

    # For data visualization the above function is enough,
    # this is for the convenience of training and visualize latents.
    if dataset_name == 'ewine':

        for i, batch in enumerate(dataloader_test):
            cir_gt = batch["CIR"]
            label_gt = batch["Label"]
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                label_gt = label_gt.cuda()
                # label_gt = label_gt.to(device=device, dtype=torch.int64)

            # cir_arr = cir_gt.view(cir_gt.size(0), -1)  # not necessary
            cir_arr = cir_arr.cpu().numpy()
            # don't reduce dim batch-wise, method will consider in the speciality within the batch
            # reduced_data = reduce_latents(cir_arr, method=method, n_components=n_components)
            labels = label_gt.cpu().numpy()
            if i == 0:
                features_arr = cir_arr  # reduced_data
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, cir_arr))
                labels_arr = np.vstack((labels_arr, labels))

        features_arr = reduce_latents(features_arr, method=method, n_components=n_components)
        visualize_latents(features_arr, labels_arr, result_path, epoch=00,
                          dataset_name=dataset_name,  # neglect env and room
                          method=method, n_components=n_components)

    elif dataset_name == 'zenodo':

        for i, batch in enumerate(dataloader_test):
            cir_gt = batch["CIR"]
            obs_label_gt = batch["Label"]
            room_label_gt = batch["RoomLabel"]
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                obs_label_gt = obs_label_gt.cuda()
                room_label_gt = room_label_gt.cuda()
                # obs_label_gt = obs_label_gt.to(device=device, dtype=torch.int64)
                # room_label_gt = room_label_gt.to(device=device, dtype=torch.int64)

            # don't reduce dim batch-wise, method will consider in the speciality within the batch
            # reduced_data = reduce_latents(cir_gt, method=method, n_components=n_components)
            obs_labels = obs_label_gt.cpu().numpy()
            room_labels = room_label_gt.cpu().numpy()
            if i == 0:
                features_arr = cir_gt  # reduced_data
                obs_labels_arr = obs_labels
                room_labels_arr = room_labels
            else:
                features_arr = np.vstack((features_arr, cir_gt))
                obs_labels_arr = np.vstack((obs_labels_arr, obs_labels))
                room_labels_arr = np.vstack((room_labels_arr, room_labels))

        # print("Test feature arry length: ", len(obs_labels_arr))
        features_arr = reduce_latents(features_arr, method=method, n_components=n_components)
        visualize_latents(features_arr, obs_labels_arr, result_path, epoch=00,
                          dataset_name=dataset_name, dataset_env=dataset_env, level="obstacle",
                          method=method, n_components=n_components)
        visualize_latents(features_arr, room_labels_arr, result_path, epoch=00,
                          dataset_name=dataset_name, dataset_env=dataset_env, level="room",
                          method=method, n_components=n_components)

    else:
        raise ValueError("Unknown dataset %s." % dataset_name)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ewine", help="dataset name")
    parser.add_argument("--dataset_env", type=str, default="nlos", help="different dataset environment labels")
    parser.add_argument("--split_factor", type=float, default=0.8, help="split for training and testing data")

    parser.add_argument("--visual_method", type=str, default="umap", help="umap or t-sne to visualize cir data.")
    parser.add_argument("--visual_dim", type=int, default=2, help="2D/3D for data visualization")
    opt = parser.parse_args()

    # assign different roots of each dataset
    if opt.dataset_name == 'zenodo':
        root = './data/data_zenodo/dataset.pkl'
    elif opt.dataset_name == 'ewine':
        folderpaths = [
            './data/data_ewine/dataset1/',
            './data/data_ewine/dataset2'
        ]
        root = folderpaths

    # assign data for training and testing
    data_train, data_test = err_mitigation_dataset(
        root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env,
        split_factor=opt.split_factor, scaling=True
    )
    print("%s dataset check length: %d/%d" % (opt.dataset_name, len(data_train[0]), len(data_test[0])))
    # zenodo: 9078/2270, ewine: 27990/6998

    ## 1 test visualize data
    result_path = "saved_results/visual_data/"
    os.makedirs(result_path, exist_ok=True)
    visualize_data(result_path, data_test, method="umap", n_components=2,
                   dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    visualize_data(result_path, data_test, method="umap", n_components=3,
                   dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    visualize_data(result_path, data_test, method="t-sne", n_components=2,
                   dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    visualize_data(result_path, data_test, method="t-sne", n_components=3,
                   dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    print("Finish Test Phase 1.")

    ## 2 test visualize data batch-wise for training process
    # dataset
    if opt.dataset_name == "ewine":
        # dataset_train = UWBDatasetEwine(data_train)
        dataset_test = UWBDatasetEwine(data_test)
    elif opt.dataset_name == "zenodo":
        # dataset_train = UWBDatasetZenodo(data_train)
        dataset_test = UWBDatasetZenodo(data_test)

    # dataloaders
    # dataloader = DataLoader(
    #     dataset=dataset_train,
    #     batch_size=1,
    #     shuffle=True,
    #     num_workers=8,
    # )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=100,
        shuffle=True,
        num_workers=1,
    )

    result_path = "saved_results/visual_data_batch/"
    os.makedirs(result_path, exist_ok=True)
    visualize_data_batch(result_path, dataloader_test,
                         method="umap", n_components=2,
                         dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    visualize_data_batch(result_path, dataloader_test,
                         method="umap", n_components=3,
                         dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    visualize_data_batch(result_path, dataloader_test,
                         method="t-sne", n_components=2,
                         dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    visualize_data_batch(result_path, dataloader_test,
                         method="t-sne", n_components=3,
                         dataset_name=opt.dataset_name, dataset_env=opt.dataset_env)
    print("Finish Test Phase 2.")