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
# train_path = "saved_results_ab%d/data_%s_%s_mode_%s/ae%didy%dreg%d/" % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type)
# os.makedirs(model_path, exist_ok=True)
# os.makedirs(train_path, exist_ok=True)
test_path = "saved_results_ab%d/test/"
os.makedirs(test_path, exist_ok=True)

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

for rate_e in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
    test_vl_multisemi(
        opt=opt, device=device, result_path=test_path, model_path=model_path,
        dataloader=dataloader_test, enc=Enc, dec=Dec, idy=Idy, reg=Reg,
        test_epoch=opt.test_epoch, data_raw=data,
        sup_rate_e=0.8, sup_rate_r=rate_r,
        test_flag=True
    )
# data_train, data_test = data_raw
res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
# accuracy, svc_test_time = svm_classifier(data_train, data_test)
res_svm = np.asarray(res_svm)
CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
plt.legend(
    [
        "Unmitigated Error",
        "GEM ($\eta_e=0.1$)", "GEM ($\eta_e=0.2$)", "GEM ($\eta_e=0.4$)",
        "GEM ($\eta_e=0.6$)", "GEM ($\eta_e=0.8$)", "GEM ($\eta_e=1.0$)",
        "SVM"
    ]
)
plt.savefig(os.path.join(result_path, "CDFMulti_%d.png" % test_epoch))
plt.savefig(os.path.join(result_path, "CDFMulti_%d.pdf" % test_epoch))
plt.close()


for rate_r in []:
    test_vl_multisemi(
        opt=opt, device=device, result_path=test_path, model_path=model_path,
        dataloader=dataloader_test, enc=Enc, dec=Dec, idy=Idy, reg=Reg,
        test_epoch=opt.test_epoch, data_raw=data,
        sup_rate_e=rate_e, sup_rate_r=0.8,
        test_flag=True
    )
# data_train, data_test = data_raw
res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
# accuracy, svc_test_time = svm_classifier(data_train, data_test)
res_svm = np.asarray(res_svm)
CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
plt.legend(
    [
        "Unmitigated Error",
        "GEM ($\eta_r=0.1$)", "GEM ($\eta_r=0.2$)", "GEM ($\eta_r=0.4$)",
        "GEM ($\eta_r=0.6$)", "GEM ($\eta_r=0.8$)", "GEM ($\eta_r=1.0$)",
        "SVM"
    ],
    loc='lower right'
)
plt.savefig(os.path.join(result_path, "CDFMulti_%d.png" % test_epoch))
plt.savefig(os.path.join(result_path, "CDFMulti_%d.pdf" % test_epoch))
plt.close()