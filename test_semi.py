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
from models import *

import setproctitle
import logging


setproctitle.setproctitle("UWB_VL")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Get arguments
parser = argparse.ArgumentParser()
parser = get_args(parser)
opt = parser.parse_args()
print(opt)

# Set model and result paths
# model_path_2 = "./saved_models_semi/%s_mode_%s/SEMI%f_AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)
model_path_1 = "./saved_models_semi1/data_ewine_nlos_mode_full/ae2_res1_cly1"
model_path_2 = "./saved_models_semi2/data_ewine_nlos_mode_full/ae2_res1_cly1"
model_path_4 = "./saved_models_semi4/data_ewine_nlos_mode_full/ae2_res1_cly1"
model_path_6 = "./saved_models_semi6/data_ewine_nlos_mode_full/ae2_res1_cly1"
model_path_8 = "./saved_models_semi8/data_ewine_nlos_mode_full/ae2_res1_cly1"
model_path_10 = "./saved_models_semi10/data_ewine_nlos_mode_full/ae2_res1_cly1"
test_path = "./saved_results_semi/test_compare"

# Load encoders, decoders and restorers
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

# semi 0.1
Enc1 = Encoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim).to(device)
Res1 = Restorer(use_soft=opt.use_soft, layer_type=opt.restorer_type, conv_type=opt.ae_type, range_dim=opt.range_dim, n_downsample=opt.n_downsample).to(device)
# semi 0.2
Enc2 = Encoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim).to(device)
Res2 = Restorer(use_soft=opt.use_soft, layer_type=opt.restorer_type, conv_type=opt.ae_type, range_dim=opt.range_dim, n_downsample=opt.n_downsample).to(device)
# Cls2 = Classifier(env_dim=opt.env_dim, num_classes=opt.num_classes, filters=16, net_type=opt.classifier_type).to(device)
# semi 0.4
Enc4 = Encoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim).to(device)
Res4 = Restorer(use_soft=opt.use_soft, layer_type=opt.restorer_type, conv_type=opt.ae_type, range_dim=opt.range_dim, n_downsample=opt.n_downsample).to(device)
# Cls4 = Classifier(env_dim=opt.env_dim, num_classes=opt.num_classes, filters=16, net_type=opt.classifier_type).to(device)
# semi 0.6
Enc6 = Encoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim).to(device)
Res6 = Restorer(use_soft=opt.use_soft, layer_type=opt.restorer_type, conv_type=opt.ae_type, range_dim=opt.range_dim, n_downsample=opt.n_downsample).to(device)
# Cls6 = Classifier(env_dim=opt.env_dim, num_classes=opt.num_classes, filters=16, net_type=opt.classifier_type).to(device)
# semi 0.8
Enc8 = Encoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim).to(device)
Res8 = Restorer(use_soft=opt.use_soft, layer_type=opt.restorer_type, conv_type=opt.ae_type, range_dim=opt.range_dim, n_downsample=opt.n_downsample).to(device)
# Cls8 = Classifier(env_dim=opt.env_dim, num_classes=opt.num_classes, filters=16, net_type=opt.classifier_type).to(device)
# semi 1.0
Enc10 = Encoder(conv_type=opt.ae_type, filters=opt.filters, n_residual=opt.n_residual, n_downsample=opt.n_downsample, env_dim=opt.env_dim, range_dim=opt.range_dim).to(device)
Res10 = Restorer(use_soft=opt.use_soft, layer_type=opt.restorer_type, conv_type=opt.ae_type, range_dim=opt.range_dim, n_downsample=opt.n_downsample).to(device)
# Cls10 = Classifier(env_dim=opt.env_dim, num_classes=opt.num_classes, filters=16, net_type=opt.classifier_type).to(device)


if opt.test_epoch != 0:
    Enc1.load_state_dict(torch.load(os.path.join(model_path_10, "Enc_%d.pth" % opt.test_epoch)))
    Res1.load_state_dict(torch.load(os.path.join(model_path_10, "Res_%d.pth" % opt.test_epoch)))
    Enc1.eval()
    Res1.eval()
    Enc2.load_state_dict(torch.load(os.path.join(model_path_2, "Enc_%d.pth" % opt.test_epoch)))
    Res2.load_state_dict(torch.load(os.path.join(model_path_2, "Res_%d.pth" % opt.test_epoch)))
    Enc2.eval()
    Res2.eval()
    Enc4.load_state_dict(torch.load(os.path.join(model_path_4, "Enc_%d.pth" % opt.test_epoch)))
    Res4.load_state_dict(torch.load(os.path.join(model_path_4, "Res_%d.pth" % opt.test_epoch)))
    Enc4.eval()
    Res4.eval()
    Enc6.load_state_dict(torch.load(os.path.join(model_path_6, "Enc_%d.pth" % opt.test_epoch)))
    Res6.load_state_dict(torch.load(os.path.join(model_path_6, "Res_%d.pth" % opt.test_epoch)))
    Enc6.eval()
    Res6.eval()
    Enc8.load_state_dict(torch.load(os.path.join(model_path_8, "Enc_%d.pth" % opt.test_epoch)))
    Res8.load_state_dict(torch.load(os.path.join(model_path_8, "Res_%d.pth" % opt.test_epoch)))
    Enc8.eval()
    Res8.eval()
    Enc10.load_state_dict(torch.load(os.path.join(model_path_10, "Enc_%d.pth" % opt.test_epoch)))
    Res10.load_state_dict(torch.load(os.path.join(model_path_10, "Res_%d.pth" % opt.test_epoch)))
    Enc10.eval()
    Res10.eval()
else:
    print("No saved models in dir.")

# Save experimental results
os.makedirs(test_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(test_path, 'test_log_semi_compare.log'), level=logging.INFO)
logging.info("Started")

# Assign data for testing
root = './data/data_zenodo/dataset.pkl'
data_train, data_test, _, _ = err_mitigation_dataset(
    root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env, split_factor=0.8, scaling=True, mode=opt.mode
)

# Configure dataloader for testing
dataloader_test = DataLoader(
    dataset=UWBDataset(data_test),
    batch_size=500,
    shuffle=True,
    num_workers=1,
)

# Evaluation initialization
rmse_error = 0.0
abs_error = 0.0
accuracy = 0.0


# ============================
#        Testing
# ============================

prev_time = time.time()

for i, batch in enumerate(dataloader_test):

    # Set model input
    cir_gt = batch["CIR"]
    err_gt = batch["Err"]
    # label_gt = batch["Label"]
    if torch.cuda.is_available():
        cir_gt = cir_gt.cuda()
        err_gt = err_gt.cuda()
        # label_gt = label_gt.to(device=device, dtype=torch.int64)

    with torch.no_grad():
        # semi 0.1
        range_code_1, env_code_1, _, _ = Enc1(cir_gt)
        err_fake_1 = Res1(range_code_1)
        # semi 0.2
        range_code_2, env_code_2, _, _ = Enc2(cir_gt)
        err_fake_2 = Res2(range_code_2)
        # semi 0.4
        range_code_4, env_code_4, _, _ = Enc4(cir_gt)
        err_fake_4 = Res4(range_code_4)
        # semi 0.6
        range_code_6, env_code_6, _, _ = Enc6(cir_gt)
        err_fake_6 = Res6(range_code_6)
        # semi 0.8
        range_code_8, env_code_8, _, _ = Enc8(cir_gt)
        err_fake_8 = Res8(range_code_8)
        # semi 1.0
        range_code_10, env_code_10, _, _ = Enc10(cir_gt)
        err_fake_10 = Res10(range_code_10)

        err_gt = err_gt.cpu().numpy()
        err_fake_1 = err_fake_1.cpu().numpy()
        err_fake_2 = err_fake_2.cpu().numpy()
        err_fake_4 = err_fake_4.cpu().numpy()
        err_fake_6 = err_fake_6.cpu().numpy()
        err_fake_8 = err_fake_8.cpu().numpy()
        err_fake_10 = err_fake_10.cpu().numpy()
        if i == 0:
            err_gt_arr = err_gt
            err_fake_arr_1 = err_fake_1
            err_fake_arr_2 = err_fake_2
            err_fake_arr_4 = err_fake_4
            err_fake_arr_6 = err_fake_6
            err_fake_arr_8 = err_fake_8
            err_fake_arr_10 = err_fake_10
        else:
            err_gt_arr = np.vstack((err_gt_arr, err_gt))
            err_fake_arr_1 = np.vstack((err_fake_arr_1, err_fake_1))
            err_fake_arr_2 = np.vstack((err_fake_arr_2, err_fake_2))
            err_fake_arr_4 = np.vstack((err_fake_arr_4, err_fake_4))
            err_fake_arr_6 = np.vstack((err_fake_arr_6, err_fake_6))
            err_fake_arr_8 = np.vstack((err_fake_arr_8, err_fake_8))
            err_fake_arr_10 = np.vstack((err_fake_arr_10, err_fake_10))

        CDF_plot_semi_test(opt, root, test_path, opt.test_epoch, err_gt_arr, err_fake_arr_1, err_fake_arr_2, err_fake_arr_4, err_fake_arr_6, err_fake_arr_8, err_fake_arr_10)

