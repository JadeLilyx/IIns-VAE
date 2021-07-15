import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as FloatTensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from utils import *
from dataset import *
from model_sep import *
from baseline import svm_regressor
from train import *
from test import *

import setproctitle
import logging


setproctitle.setproctitle("UWB_EM")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Get arguments
parser = argparse.ArgumentParser()
parser = get_args(parser)
opt = parser.parse_args()
print(opt)

# Initialize network
if opt.dataset_name == 'zenodo':
    opt.cir_len = 157
    if opt.dataset_env == 'room_full':
        opt.num_classes = 5
    elif opt.dataset_env == 'obstacle_full':
        opt.num_classes = 10
    elif opt.dataset_env == 'nlos':
        opt.num_classes = 2
    elif opt.dataset_env == 'room_part':
        opt.num_classes = 3
    elif opt.data_env == 'obstacle_part':
        opt.num_classes = 4
elif opt.dataset_name == 'ewine':
    opt.cir_len = 152
    opt.dataset_env = 'nlos'
    opt.num_classes = 2
ENet = IdentifierSep(cir_len=opt.cir_len, num_classes=opt.num_classes, env_dim=opt.env_dim,
    filters=opt.filters, enet_type=opt.identifier_type).to(device)
MNet = RegressorSep(cir_len=opt.cir_len, num_classes=opt.num_classes, env_dim=opt.env_dim,
    filters=opt.filters, mnet_type=opt.regressor_type).to(device)

# Create sample and checkpoint directories
model_path = "./saved_models_sep/data_%s_%s_mode_%s/enet%d_mnet%d" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.identifier_type, opt.regressor_type)
train_path = "./saved_results_sep/data_%s_%s_mode_%s/enet%d_mnet%d" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.identifier_type, opt.regressor_type)
os.makedirs(model_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
test_path = "./saved_results_sep/test/data_%s_%s_mode_%s/enet%d_mnet%d" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.identifier_type, opt.regressor_type)
os.makedirs(test_path, exist_ok=True)

# Optimizers
optimizerE = torch.optim.Adam(
    ENet.parameters(),
    lr=opt.lr,
    betas=(opt.b1, opt.b2)
)

optimizerM = torch.optim.Adam(
    MNet.parameters(),
    lr=opt.lr,
    betas=(opt.b1, opt.b2)
)


# Get data
print("Loading dataset from %s_%s for training." % (opt.dataset_name, opt.dataset_env))
if opt.dataset_name == 'zenodo':
    root = './data/data_zenodo/dataset.pkl'
elif opt.dataset_name == 'ewine':
    filepaths = ['./data/data_ewine/dataset1/tag_room0.csv',
                 './data/data_ewine/dataset1/tag_room1.csv',
                 './data/data_ewine/dataset2/tag_room0.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part0.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part1.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part2.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part3.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part4.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part5.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part6.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part7.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part8.csv',
                 './data/data_ewine/dataset2/tag_room1/tag_room1_part9.csv']
    root = filepaths

data_train, data_test = assign_train_test(opt, root)

# Configure dataloaders
dataloader_train = DataLoader(
    dataset=UWBDataset(data_train),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8,
)

dataloader_test = DataLoader(
    dataset=UWBDataset(data_test),
    batch_size=500,
    shuffle=True,
    num_workers=1,
)


# ------------- Training --------------

data = data_train, data_test
train_gem_sepE(
    opt, device=device, tensor=Tensor, result_path=train_path, model_path=model_path, 
    dataloader=dataloader_train, val_dataloader=dataloader_test,
    optimizer_e=optimizerE, enet=ENet, data_raw=data
)

train_gem_sepM(
    opt, device=device, tensor=Tensor, result_path=train_path, model_path=model_path, 
    dataloader=dataloader_train, val_dataloader=dataloader_test,
    optimizer_m=optimizerM, enet=ENet, mnet=MNet, data_raw=data
)

# ------------- Testing --------------

test_gem_sepE(
    opt=opt, device=device, tensor=Tensor, result_path=test_path, model_path=model_path, 
    dataloader=dataloader_test, enet=ENet, epoch=opt.test_epoch, daat_raw=data
)  # epoch for val and opt.test_epoch for test

test_gem_sepEM(
    opt=opt, device=device, tensor=Tensor, result_path=test_path, model_path=model_path, 
    dataloader=dataloader_test, enet=ENet, mnet=MNet, epoch=opt.test_epoch, data_raw=data
)  # epoch for val and opt.test_epoch for test
