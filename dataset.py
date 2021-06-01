import numpy as np
import os
import glob
import random
import time

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from data_tools import *


def err_mitigation_dataset(root, dataset_name='zenodo', dataset_env=None, split_factor=0.8, scaling=False, mode='paper', feature_flag=False):
    # datasets_reg = collections.namedtuple('Datasets', ['train', 'test'])
    err = []
    cir = []
    label = []
    # get data from files
    if dataset_name == 'ewine':
        # (not used)
        err, cir = load_reg_data(filepaths=root)  # list
        label, cir = load_cls_data(filepaths=root)
    elif dataset_name == 'zenodo':
        if not dataset_env:
            dataset_env = 'room_full'
        err, cir, label = load_pkl_data(filepath=root, option=dataset_env)
        # cir = cir[:, :-1]
        # cir = np.concatenate((cir, np.zeros((len(cir), 1))), axis=1)  # 158 instead of 157 for net
    
    # split and reshape data
    if mode == 'full':  # split factor
        train_err, test_err = np.split(err, np.asarray([int(np.size(err, 0) * split_factor)]))
        train_cir, test_cir = np.split(cir, np.array([int(np.size(cir, 0) * split_factor)]))
        train_label, test_label = np.split(label, np.array([int(np.size(label, 0) * split_factor)]))
    elif mode == 'paper':  # medium room as test and rest as train
        data_env = 'full_room'
        data_module = np.hstack((cir, err, label))
        # label=2 for medium room, separate it from training set and for testing set
        test_cir = None
        test_err = None
        test_label = None
        train_cir = None
        train_err = None
        train_label = None
                
        for i in range(data_module.shape[0]):
            if data_module[i, -1] == 2:
                test_cir = data_module[i, 0:len(cir[0])] if test_cir is None else np.vstack((test_cir, data_module[i, 0:len(cir[0])]))
                test_err = data_module[i, len(cir[0]):len(cir[0])+1] if test_err is None else np.vstack((test_err, data_module[i, len(cir[0]):len(cir[0])+1]))
                test_label = data_module[i, len(cir[0])+1:] if test_label is None else np.vstack((test_label, data_module[i, len(cir[0])+1:]))
            else:
                train_cir = data_module[i, 0:len(cir[0])] if train_cir is None else np.vstack((train_cir, data_module[i, 0:len(cir[0])]))
                train_err = data_module[i, len(cir[0]):len(cir[0])+1] if train_err is None else np.vstack((train_err, data_module[i, len(cir[0]):len(cir[0])+1]))
                train_label = data_module[i, len(cir[0])+1:] if train_label is None else np.vstack((train_label, data_module[i, len(cir[0])+1:]))
    
    # print(train_label.shape)
    train_err = np.reshape(train_err, (len(train_err), 1))  # (n, 1)
    test_err = np.reshape(test_err, (len(test_err), 1))  # (n, 1)
    train_label = np.reshape(train_label, (len(train_label), 1))
    test_label = np.reshape(test_label, (len(test_label), 1))

    # extract features
    if feature_flag:
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

    # # save to file
    # save_path_train = os.path.join('./data/dataset_for_usage/%s', dataset_name)
    # save_path_test = os.path.join('./data/dataset_for_usage/%s', dataset_name)
    # os.makedirs(save_path_train, exist_ok=True)
    # os.makedirs(save_path_test, exist_ok=True)
    # df.to_csv(os.path.join(save_path_train, "train.csv"))

    train = train_cir, train_err, train_label  # (0.8*n, 152), (0.8*n, 1)
    test = test_cir, test_err, test_label  # (0.2*n, 152), (0.2*n, 1)


    return train, test, train_features, test_features  # datasets_reg(train=train, test=test)
    # data = err_mitigation_dataset()


class UWBDataset(Dataset):
    def __init__(self, data):
        # basic data
        self.data = data
        cir_arr, err_arr, label_arr = self.data  # train/test
        self.cir = cir_arr
        self.err = err_arr
        self.label = label_arr

        # extract features
        # Er, T_EMD, T_RMS, Kur, R_T, M_AMP = feature_extraction(cir_arr)
        # self.er = Er
        # self.temd = T_EMD
        # self.trms = T_RMS
        # self.kur = Kur
        # self.rt = R_T
        # self.mamp = M_AMP

        # define transform to make data Tensor
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     # transforms.Normalize()
        # ])
        # self.transform = transforms.Compose(transforms_)

    def __getitem__(self, index):
        # get basic data
        cir_item = torch.Tensor(self.cir[index % len(self.cir)])  # (152)
        # cir_item = self.transform(cir_item)
        err_item = torch.Tensor(self.err[index % len(self.err)])
        label_item = torch.Tensor(self.label[index % len(self.label)])
        
        # get extracted features
        # er_item = torch.Tensor(self.er[index % len(self.er)])
        # temd_item = torch.Tensor(self.temd[index % len(self.temd)])
        # trms_item = torch.Tensor(self.trms[index % len(self.trms)])
        # kur_item = torch.Tensor(self.kur[index % len(self.kur)])
        # rt_item = torch.Tensor(self.rt[index % len(self.rt)])
        # mamp_item = torch.Tensor(self.mamp[index % len(self.mamp)])

        return {"CIR": cir_item, "Err": err_item, "Label": label_item}  # "Er": er_item, "T_EMD": temd_item, "T_RMS": trms_item, "Kur": kur_item, "R_T": rt_item, "M_AMP": mamp_item}

    def __len__(self):
        return len(self.cir)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="zenodo",
                        help="dataset for usage, ewine or zenodo")
    # parser.add_argument("--dataset_use", type=str, default='regression',
    #                     help="dataset (ewine) of different usage, including classification and regression")
    parser.add_argument("--dataset_env", type=str, default='room_full',
                        help="dataset (zenodo) of different environments, including rooms and obstacles")
    parser.add_argument("--mode", type=str, default="paper", help="simulated mode train/test for data usage, paper or full")
    parser.add_argument("--split_factor", type=float, default=0.8, help="ratio to split data for training and testing")
    opt = parser.parse_args()

    # assign different roots of each dataset
    root = []
    if opt.dataset_name == 'ewine':  # load_reg_data(filepaths=root)
        print("Loading data for regression.")
        root = ['./data/data_ewine/dataset_reg1/tag_room0.csv',
                './data/data_ewine/dataset_reg1/tag_room1.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part0.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part1.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part2.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part3.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part4.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part5.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part6.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part7.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part8.csv',
                './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part9.csv']
        err_reg, cir_reg = load_reg_data(root)
        print("Shape of error: ", err_reg.shape)  # (31489, 1)
        print("Shape of cir sample: ", cir_reg.shape)  # (31489, 152)

    elif opt.dataset_name == 'zenodo':  # load_pkl_data(filepath=root)

        print("Loading data for regression.")
        root = './data/data_zenodo/dataset.pkl'
        # err_reg, cir_reg, label_reg = load_pkl_data(root, option=opt.dataset_env)
        # print("Shape of error: ", err_reg.shape)  # (55158, 1)
        # print("Shape of cir sample: ", cir_reg.shape)  # (55158, 157)
        # print("Shape of env label: ", label_reg.shape)

    else:
        raise RuntimeError("Unknown dataset for usage.")

    # assign data for training and testing
    start_time = time.time()
    data_train, data_test, feature_train, feature_test = err_mitigation_dataset(
        root=root, dataset_env=opt.dataset_env, split_factor=opt.split_factor, scaling=True, mode=opt.mode, feature_flag=False
    )  # feature_extraction=False
    time = time.time() - start_time
    print("time: ", time)  # around 270 for no feature and 290 for feature
    print("check length: ", len(data_train[0]), len(data_test[0]))  # train - 25191/44126, test - 6298/11032

    # datasets
    dataset_train = UWBDataset(data_train)  # feature_train
    dataset_test = UWBDataset(data_test)

    # datasetloaders
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

    # Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    for i, batch in enumerate(dataloader_test):
        # cir_gt = Variable(batch["CIR"].type(Tensor))
        # err_gt = Variable(batch["Err"].type(Tensor))
        cir_gt = batch["CIR"]  # (B, 157)
        err_gt = batch["Err"]  # (B, 1)
        label_gt = batch['Label']  # (B, 1)
        
        # test an extra feature
        # Er = batch["Er"]

        # print(cir_gt.shape)
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()
            label_gt = label_gt.cuda()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        env = Variable(torch.randn(cir_gt.size(0), 8, 1, 1).type(Tensor))
        if i == 1:
            print("cir type: ", cir_gt.type)
            print("env type: ", env.type)
            print('cir shape: ', cir_gt.shape)
            print("cir max is {} and min is {}".format(max(cir_gt[0]), min(cir_gt[0])))
            # ewine: 51.7314, 0.1088; zenodo: 17592.9344, 67.6240 (scaling=False)
            # weine: 3.9475, -1.0568; zenodo: 4.6199, -2.0656
            print("cir waveform is {}, ranging error is {}, env label is {}".format(cir_gt, err_gt, label_gt))
            # tensor([[...152dim]], device='cuda:0'), tensor([[0.1700]], device='cuda:0')
            