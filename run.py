import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from utils import *
from dataset import *
from model import *
from train import *
from test import *


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Get arguments
parser = argparse.ArgumentParser()
parser = get_args(parser)
opt = parser.parse_args()
print(opt)

# Initialize networks
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
    elif opt.dataset_env == 'obstacle_part':
        opt.num_classes = 4
elif opt.dataset_name == 'ewine':
    opt.cir_len = 152
    opt.dataset_env = 'nlos'
    opt.num_classes = 2


# Select neural module arrangement
scale_factor = 2 ** opt.n_downsample
if opt.conv_type == 1:
    range_code_shape = (opt.range_dim, 128 // (2 ** opt.n_downsample))
elif opt.conv_type == 2:
    range_code_shape = (opt.range_dim, 128 // (2 ** opt.n_downsample), 128 // (2 ** opt.n_downsample))
Enc = Encoder(
    conv_type=opt.conv_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample,
    env_dim=opt.env_dim, range_dim=opt.range_dim
).to(device)
Dec = Decoder(
    conv_type=opt.conv_type, filters=opt.filters, n_residual=opt.n_residual, n_upsample=opt.n_downsample,
    env_dim=opt.env_dim, range_dim=opt.range_dim, out_dim=opt.cir_len
).to(device)
Idy = Classifier(
    env_dim=opt.env_dim, num_classes=opt.num_classes, filters=opt.filters, layer_type=opt.identifier_type
).to(device)
Reg = Restorer(
    use_soft=opt.use_soft, layer_type=opt.regressor_type, conv_type=opt.conv_type,
    range_dim=opt.range_dim, n_downsample=opt.n_downsample
).to(device)

# Create sample and checkpoint directories
model_path = "saved_models_ab%d/data_%s_%s_mode_%s/ae%didy%dreg%d/" % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type)
train_path = "saved_results_ab%d/data_%s_%s_mode_%s/ae%didy%dreg%d/" % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type)
os.makedirs(model_path, exist_ok=True)
os.makedirs(train_path, exist_ok=True)
test_path = "saved_results_ab%d/test/data_%s_%s_mode_%s/ae%didy%dreg%d/" % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type)
os.makedirs(test_path, exist_ok=True)

# Optimizers
optimizer = torch.optim.Adam(
    itertools.chain(Enc.parameters(), Dec.parameters(), Idy.parameters(), Reg.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2)
)

# Learning rate update schedulers (optional)
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
    raise RuntimeError("Unknown dataset for usage.")

data_train, data_test = assign_train_test(opt, root)

# Configure dataloaders
dataloader_train = DataLoader(
    dataset=UWBDataset(data_train),
    batch_size=opt.batch_size,  # 500
    shuffle=True,
    num_workers=8,
)

dataloader_test = DataLoader(
    dataset=UWBDataset(data_test),
    batch_size=500,
    shuffle=True,
    num_workers=1,
)


# ------------- Training and Testing ----------------

data = data_train, data_test
# choose ablation cases
if opt.ablation_type == 0:
    train_vl_naive(
        opt, device=device, result_path=train_path, model_path=model_path,
        dataloader=dataloader_train, val_dataloader=dataloader_test, optimizer=optimizer, scheduler=lr_scheduler,
        enc=Enc, dec=Dec, idy=Idy, reg=Reg, data_raw=data
    )
    
    test_vl_naive(
        opt=opt, device=device, result_path=test_path, model_path=model_path,
        dataloader=dataloader_test, enc=Enc, dec=Dec, idy=Idy, reg=Reg,
        test_epoch=opt.test_epoch, data_raw=data, test_flag=True
    )

elif opt.ablation_type == 1:
    # same as type 0 if opt.sup_rate_r=opt.sup_rate_e=1, more general and can be merged
    train_vl_semi(
        opt, device=device, result_path=train_path, model_path=model_path,
        dataloader=dataloader_train, val_dataloader=dataloader_test, optimizer=optimizer, scheduler=lr_scheduler,
        enc=Enc, dec=Dec, idy=Idy, reg=Reg, data_raw=data
    )

    test_vl_semi(
        opt=opt, device=device, result_path=test_path, model_path=model_path,
        dataloader=dataloader_test, enc=Enc, dec=Dec, idy=Idy, reg=Reg,
        test_epoch=opt.test_epoch, data_raw=data,
        test_flag=True
    )

elif opt.ablation_type == 2:
    train_vl_DeIdy(
        opt, device=device, result_path=train_path, model_path=model_path,
        dataloader=dataloader_train, val_dataloader=dataloader_test, optimizer=optimizer, scheduler=lr_scheduler,
        enc=Enc, dec=Dec, reg=Reg, data_raw=data
    )
    # read additional identifier for testing
    idy_path = "saved_models_ab0/data_%s_%s_mode_%s/ae%didy%dreg%d/" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type)
    test_vl_DeIdy_variant(
        opt=opt, device=device, result_path=test_path, model_path=model_path, model_path_ref=idy_path,
        dataloader=dataloader_test, enc=Enc, dec=Dec, reg=Reg, idy_ref=Idy,
        test_epoch=opt.test_epoch, data_raw=data, test_flag=True
    )

elif opt.ablation_type == 3:
    train_vl_DeReg(
        opt, device=device, result_path=train_path, model_path=model_path,
        dataloader=dataloader_train, val_dataloader=dataloader_test, optimizer=optimizer, scheduler=lr_scheduler,
        enc=Enc, dec=Dec, idy=Idy, data_raw=data
    )
    # read additional identifier for testing
    reg_path = "saved_models_ab0/data_%s_%s_mode_%s/ae%didy%dreg%d/" % (opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type)
    test_vl_DeReg_variant(
        opt=opt, device=device, result_path=test_path, model_path=model_path, model_path_ref=reg_path,
        dataloader=dataloader_test, enc=Enc, dec=Dec, idy=Idy, reg_ref=Reg,
        test_epoch=opt.test_epoch, data_raw=data, test_flag=True
    )

elif opt.ablation_type == 4:
    train_vl_DeDec(
        opt, device=device, result_path=train_path, model_path=model_path,
        dataloader=dataloader_train, val_dataloader=dataloader_test, optimizer=optimizer, scheduler=lr_scheduler,
        enc=Enc, idy=Idy, reg=Reg, data_raw=data
    )
    test_vl_DeRec(
        opt=opt, device=device, result_path=test_path, model_path=model_path,
        dataloader=dataloader_test, enc=Enc, idy=Idy, reg=Reg,
        test_epoch=opt.test_epoch, data_raw=data, test_flag=True
    )
else:
    raise ValueError("Unknown ablation study type, choices 0~4.")

