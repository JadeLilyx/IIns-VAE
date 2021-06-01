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


setproctitle.setproctitle("UWB_AE")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Get arguments
parser = argparse.ArgumentParser()
parser = get_args(parser)
opt = parser.parse_args()
print(opt)

# Set model and result paths
# opt.supervision_rate = 1.0
# model_path = "./saved_models_semi/%s_mode_%s/SEMI%f_AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.supervision_rate, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)
# test_path = "./saved_results_semi/test/%s_mode_%s/SEMI%d_AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.supervision_rate, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)

model_path = "./saved_models/%s_mode_%s/AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)
test_path = "./saved_results/test/%s_mode_%s/AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)

# Load encoders, decoders and restorers
len_cir = 157
if opt.dataset_env == 'room_full':
    opt.num_classes = 5
elif opt.dataset_env == 'obstacle_full':
    opt.num_classes = 10
elif opt.dataset_env == 'room_part':
    opt.num_classes = 3
elif opt.dataset_env == 'room_full_rough':
    opt.num_classes = 3
elif opt.dataset_env == 'obstacle_part':
    opt.num_classes = 4
elif opt.dataset_env == 'obstacle_part2':
    opt.num_classes = 2
elif opt.dataset_env == 'room_full_rough2':
    opt.num_classes = 2
else:
    print("Unknown environment.")

scale_factor = 2 ** opt.n_downsample

opt.if_expand = False if opt.conv_type == 1 else True
if opt.conv_type == 1:
    range_code_shape = (opt.range_dim, 128 // (2 ** opt.n_downsample))
else:
    range_code_shape = (opt.range_dim, 128 // (2 ** opt.n_downsample), 128 // (2 ** opt.n_downsample)) if opt.if_expand \
        else (opt.range_dim, 128 // (2 ** opt.n_downsample), 1)  # (2, 8)
# if opt.if_expand == False and opt.conv_type == 2:  # benefit recording
#     opt.conv_type = 3  # conv2d without expansion

Enc = Encoder(conv_type=opt.conv_type, dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual,
              style_dim=opt.env_dim, out_dim=opt.range_dim, expand=opt.if_expand).to(device)
Dec = Decoder(conv_type=opt.conv_type, dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual,
              style_dim=opt.env_dim, in_dim=len_cir, out_dim=opt.range_dim, expand=opt.if_expand).to(device)
Res = Restorer(code_shape=range_code_shape, soft=False, filters=opt.dim, conv_type=opt.conv_type, expand=opt.if_expand, net_type=opt.restorer_type).to(device)
Cls = Classifier(env_dim=opt.env_dim, num_classes=opt.num_classes, filters=16, net_type=opt.classifier_type).to(device)

if opt.test_epoch != 0:
    Enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % opt.test_epoch)))
    Dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % opt.test_epoch)))
    Res.load_state_dict(torch.load(os.path.join(model_path, "Res_%d.pth" % opt.test_epoch)))
    Cls.load_state_dict(torch.load(os.path.join(model_path, "Cls_%d.pth" % opt.test_epoch)))
    Enc.eval()
    Dec.eval()
    Res.eval()
    Cls.eval()
else:
    print("No saved models in dir.")

# Save experimental results
os.makedirs(test_path, exist_ok=True)
logging.basicConfig(filename=os.path.join(test_path, 'test_log.log'), level=logging.INFO)
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
    label_gt = batch["Label"]
    if torch.cuda.is_available():
        cir_gt = cir_gt.cuda()
        err_gt = err_gt.cuda()
        label_gt = label_gt.to(device=device, dtype=torch.int64)

    with torch.no_grad():

        # Get latent representation
        range_code, env_code, env_code_rv, kl_div = Enc(cir_gt)

        # 1) Reconstructed cir
        cir_gen = Dec(range_code, env_code)

        # 2) Estimated range error
        err_fake = Res(range_code)

        # 3) Estimated env label
        label_fake = Cls(env_code)

        # Evaluate
        # range mitigation error
        rmse_error += (torch.mean((err_fake - err_gt) ** 2)) ** 0.5
        abs_error += torch.mean(torch.abs(err_fake - err_gt))
        time_test = (time.time() - prev_time) / 500  # batch_size
        rmse_avg = rmse_error / (i + 1)
        abs_avg = abs_error / (i + 1)
        time_avg = time_test / (i + 1)
        # env classification error
        label_gt = label_gt.squeeze()
        label_gt = label_gt.to(device=device, dtype=torch.int64)
        if opt.dataset_env == 'room_full':  # 0~4
            prediction = torch.argmax(label_fake, dim=1)
        else:
            prediction = torch.argmax(label_fake, dim=1) + 1
        accuracy += torch.sum(prediction == label_gt).float() / label_gt.shape[0]
        accuracy_avg = accuracy / (i + 1)

# Print log
sys.stdout.write(
    "\r[Model Name: AE%d_%s_%s] [Test Env: %s] [Test Epoch: %d] [RMSE: %f] [ABS ERROR: %f] [Accuracy: %f] [Test Time: %f]"
    % (opt.conv_type, opt.restorer_type, opt.classifier_type, opt.dataset_env, opt.test_epoch, rmse_avg, abs_avg, accuracy_avg, time_avg)
)
logging.info(
    "\r[Model Name: AE%d_%s_%s] [Test Env: %s] [Test Epoch: %d] [RMSE: %f] [ABS ERROR: %f] [Accuracy: %f] [Test Time: %f]"
    % (opt.conv_type, opt.restorer_type, opt.classifier_type, opt.dataset_env, opt.test_epoch, rmse_avg, abs_avg, accuracy_avg, time_avg)
)

# Qualitative results
with torch.no_grad():
    # visualize_recon(test_path, opt.test_epoch, dataloader_test, Enc, Dec)
    CDF_plot(opt, root, test_path, opt.test_epoch, dataloader_test, Enc, Res, use_competitor=True)
    visualize_latent(test_path, opt.test_epoch, dataloader_test, opt.dataset_env, Enc, Cls)