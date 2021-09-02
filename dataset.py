import numpy as np
import os
import glob
import random

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from data_tools import *


def err_mitigation_dataset(root, dataset_name='zenodo', dataset_env=None, split_factor=0.8, scaling=False, mode='paper', feature_flag=False):
    err = []
    cir = []
    label = []

    # get data from files
    if dataset_name == 'zenodo':
        if not dataset_env:
            dataset_env = 'nlos'
        cir, err, label, room_label = load_pkl_data(filepath=root, option=dataset_env)

        # split and reshape data
        # note that if we want to assign as paper, we have to return additional
        # label from 'load_pkl_data' to give label of the rooms
        if mode == 'full':
            train_err, test_err = np.split(err, np.array([int(np.size(err, 0) * split_factor)]))
            train_cir, test_cir = np.split(cir, np.array([int(np.size(cir, 0) * split_factor)]))
            train_label, test_label = np.split(label, np.array([int(np.size(label, 0) * split_factor)]))
        elif mode == 'paper':
            data_module = np.hstack((cir, err, label, room_label))
            test_cir = None
            test_err = None
            test_label = None
            train_cir = None
            train_err = None
            train_label = None
            
            for i in range(data_module.shape[0]):
                if data_module[i, -1] == 2:  # room_label=2 for medium room
                    test_cir = data_module[i, 0:len(cir[0])] if test_cir is None else np.vstack((test_cir, data_module[i, 0:len(cir[0])]))
                    test_err = data_module[i, len(cir[0]):len(cir[0])+1] if test_err is None else np.vstack((test_err, data_module[i, len(cir[0]):len(cir[0])+1]))
                    test_label = data_module[i, len(cir[0])+1:len(cir[0])+2] if test_label is None else np.vstack((test_label, data_module[i, len(cir[0])+1:len(cir[0])+2]))
                else:
                    train_cir = data_module[i, 0:len(cir[0])] if train_cir is None else np.vstack((train_cir, data_module[i, 0:len(cir[0])]))
                    train_err = data_module[i, len(cir[0]):len(cir[0])+1] if train_err is None else np.vstack((train_err, data_module[i, len(cir[0]):len(cir[0])+1]))
                    train_label = data_module[i, len(cir[0])+1:len(cir[0])+2] if train_label is None else np.vstack((train_label, data_module[i, len(cir[0])+1:len(cir[0])+2]))
    
    elif dataset_name == 'ewine':
        cir, err, label = load_reg_data(folderpaths=root)
        train_err, test_err = np.split(err, np.array([int(np.size(err, 0) * split_factor)]))
        train_cir, test_cir = np.split(cir, np.array([int(np.size(cir, 0) * split_factor)]))
        train_label, test_label = np.split(label, np.array([int(np.size(label, 0) * split_factor)]))

    train_err = np.reshape(train_err, (len(train_err), 1))
    test_err = np.reshape(test_err, (len(test_err), 1))
    train_label = np.reshape(train_label, (len(train_label), 1))
    test_label = np.reshape(test_label, (len(test_label), 1))

    # extract features
    if feature_extraction:
        train_features = feature_extraction(train_cir)
        test_features = feature_extraction(test_cir)
    else:
        train_features = None
        test_features = None

    # scale cir data to N(0, 1)
    if scaling:
        scaler = StandardScaler()
        train_cir = scaler.fit_transform(train_cir)
        test_cir = scaler.transform(test_cir)

    train = train_cir, train_err, train_label
    test = test_cir, test_err, test_label

    return train, test, train_features, test_features


class UWBDataset(Dataset):
    def __init__(self, data):
        # basic data
        self.data = data
        cir_arr, err_arr, label_arr = self.data  # train or test
        self.cir = cir_arr
        self.err = err_arr
        self.label = label_arr

    def __getitem__(self, index):
        # get basic data
        cir_item = torch.Tensor(self.cir[index % len(self.cir)])
        err_item = torch.Tensor(self.err[index % len(self.err)])
        label_item = torch.Tensor(self.label[index % len(self.label)])

        return {"CIR": cir_item, "Err": err_item, "Label": label_item}

    def __len__(self):
        return len(self.cir)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ewine", help="dataset for usage, ewine or zenodo")
    parser.add_argument("--dataset_env", type=str, default="nlos", help="different envs for data samples")
    parser.add_argument("--mode", type=str, default="paper", help="method to split train and test data")
    parser.add_argument("--split_factor", type=float, default=0.8, help="split factor for train and test data")
    opt = parser.parse_args()

    # assign different roots of each dataset
    if opt.dataset_name == 'zenodo':
        root = './data/data_zenodo/dataset.pkl'
    elif opt.dataset_name == 'ewine':
        folderpaths = ['./data/data_ewine/dataset1/',
                     './data/data_ewine/dataset2/',
                     './data/data_ewine/dataset2/tag_room1/']
        root = folderpaths

    # assign data for training and testing
    data_train, data_test, feature_train, feature_test = err_mitigation_dataset(
        root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env, split_factor=opt.split_factor,
        scaling=True, mode=opt.mode, feature_flag=False
    )
    print("check length: ", len(data_train[0]), len(data_test[0]))
    # zenodo: 9078/2270(full), 7401/3947(paper)
    # ewine: 27990/6998

    # dataset (not use features)
    dataset_train = UWBDataset(data_train)
    dataset_test = UWBDataset(data_test)

    # dataloaders
    dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=8,
        shuffle=True,
        num_workers=8,
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    for i, batch in enumerate(dataloader_test):
        cir_gt = batch["CIR"]
        err_gt = batch["Err"]
        label_gt = batch["Label"]

        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()
            label_gt = label_gt.cuda()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        env = Variable(torch.randn(cir_gt.size(0), 8, 1, 1).type(Tensor))
        if i == 1:
            print("cir type: ", cir_gt.type)
            print("env type: ", env.type)
            print("cir shape: ", cir_gt.shape)
            print("cir max is {} and min is {}".format(max(cir_gt[0]), min(cir_gt[0])))
            print("cir waveform: {}, range error: {}, env label: {}".format(cir_gt, err_gt, label_gt))