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
from models import *
from baseline import svm_regressor

import setproctitle
import logging


setproctitle.setproctitle("UWB_VL")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
Enc = Encoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim).to(device)
Dec = Decoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_upsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim, out_dim=opt.cir_len).to(device)
Res = Restorer(use_soft=opt.use_soft, layer_type=opt.restorer_type, conv_type=opt.ae_type, range_dim=opt.range_dim, n_downsample=opt.n_downsample).to(device)
Cly = Classifier(env_dim=opt.env_dim, num_classes=opt.num_classes, filters=16, layer_type=opt.classifier_type).to(device)

# Create sample and checkpoint directories
if opt.use_semi:
    model_path = "./saved_models/data_%s_%s_mode_%s/ae%d_res%d_cly%d" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.ae_type, opt.regressor_type, opt.classifier_type)
    train_path = "./saved_results/data_%s_%s_mode_%s/ae%d_res%d_cly%d" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.ae_type, opt.regressor_type, opt.classifier_type)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    test_path = "./saved_results/test/data_%s_%s_mode_%s/ae%d_res%d_cly%d" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.ae_type, opt.regressor_type, opt.classifier_type)
    os.makedirs(test_path, exist_ok=True)
else:
    model_path = "./saved_models_semi%d/data_%s_%s_mode_%s/ae%d_res%d_cly%d" % (opt.supervision_rate, opt.dataset_name, opt.dataset_env, opt.mode, opt.ae_type, opt.regressor_type, opt.classifier_type)
    train_path = "./saved_results_semi%d/data_%s_%s_mode_%s/ae%d_res%d_cly%d" % (opt.supervision_rate, opt.dataset_name, opt.dataset_env, opt.mode, opt.ae_type, opt.regressor_type, opt.classifier_type)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    test_path = "./saved_results_semi%d/test/data_%s_%s_mode_%s/ae%d_res%d_cly%d" % (opt.supervision_rate, opt.dataset_name, opt.dataset_env, opt.mode, opt.ae_type, opt.regressor_type, opt.classifier_type)
    os.makedirs(test_path, exist_ok=True)


# Optimizers
optimizer = torch.optim.Adam(
    Network.parameters(),
    lr=opt.lr,
    betas=(opt.b1, opt.b2)
)

# Learning rate update schedulers
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

# Get data
print("Loading dataset from %s_%s for training." % (opt.dataset_name, opt.dataset_env))
if opt.dataset_name == 'zenodo':
    root = './data/data_zenodo/dataset.pkl'
elif opt.dataset_name == 'ewine':
    folderpaths = ['./data/data_ewine/dataset1/',
                 './data/data_ewine/dataset2/',
                 './data/data_ewine/dataset2/tag_room1/']
    root = folderpaths
else:
    raise ValueError("Unknown dataset for usage.")

data_train, data_test = assign_train_test(opt, root)
data = data_train, data_test


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

# use test_vl() for validation in checkpoint epochs
if opt.use_semi:
    train_vl_semi(
        opt, enc=Enc, dec=Dec, res=Res, cly=Cly,
        device=device, result_path=train_path, model_path=model_path, 
        dataloader=dataloader_train, val_dataloader=dataloader_test,
        optimizer=optimizer, lr_scheduler=lr_scheduler, data=data
    )
else:
    train_vl(
        opt, enc=Enc, dec=Dec, res=Res, cly=Cly,
        device=device, result_path=train_path, model_path=model_path, 
        dataloader=dataloader_train, val_dataloader=dataloader_test,
        optimizer=optimizer, lr_scheduler=lr_scheduler, data=data
    )


# ------------- Testing --------------
# if opt.use_semi:
#     test_vl_semi(
#     opt=opt, network=Network,
#     device=device, result_path=test_path, model_path=model_path, 
#     dataloader=dataloader_test,
#     epoch=opt.test_epoch, data=data
# )
# else:
#     test_vl(
#         opt=opt, network=Network,
#         device=device, result_path=test_path, model_path=model_path, 
#         dataloader=dataloader_test,
#         epoch=opt.test_epoch, data=data
#     )  # epoch for val and opt.test_epoch for test

test_vl(
        opt=opt, enc=Enc, res=Res, cly=Cly,
        device=device, result_path=test_path, model_path=model_path, 
        dataloader=dataloader_test,
        epoch=opt.test_epoch, data=data
    )  # epoch for val and opt.test_epoch for test
