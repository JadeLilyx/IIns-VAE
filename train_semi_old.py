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


setproctitle.setproctitle("UWB_SEMI")


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Get arguments
parser = argparse.ArgumentParser()
parser = get_args(parser)
parser.add_argument("--supervision_rate", type=float, default=0.8, help="Rate of labeled data to pure cir data.")
opt = parser.parse_args()
print(opt)


# Set loss function
criterion_recon = torch.nn.L1Loss().to(device)
criterion_code = torch.nn.CrossEntropyLoss().to(device)

# Initialize encoders, decoders and restorers
# Initialize encoders, decoders and restorers
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
elif opt.dataset_env == 'paper':
    opt.num_classes = 4
else:
    print("Unknown environment.")

scale_factor = 2 ** opt.n_downsample

opt.if_expand = False if opt.conv_type == 1 else True
if opt.conv_type == 1:
    # range_code_shape = (opt.dim * scale_factor, len_cir // scale_factor)
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


# Create sample and checkpoint directories
print("conv type: ", opt.conv_type)
model_path = "saved_models_semi/%s_mode_%s/SEMI%f_AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.supervision_rate, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)
result_path = "saved_results_semi/%s_mode_%s/SEMI%f_AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.supervision_rate, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)
os.makedirs(model_path, exist_ok=True)
os.makedirs(result_path, exist_ok=True)

# Save training log
logging.basicConfig(filename=os.path.join(result_path, 'train_log.log'), level=logging.INFO)
logging.info("Started")


# Load pre-trained model or initialize weights
if opt.epoch != 0:
    Enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % epoch)))
    Dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % epoch)))
    Res.load_state_dict(torch.load(os.path.join(model_path, "Res_%d.pth" % epoch)))
    Cls.load_state_dict(torch.load(os.path.join(model_path, "Cls_%d.pth" % epoch)))
else:
    Enc.apply(weights_init_normal)
    Dec.apply(weights_init_normal)
    Res.apply(weights_init_normal)
    Cls.apply(weights_init_normal)


# Loss weights
lambda_ae = 1
lambda_res = 10
lambda_range = 1
lambda_env = 1
mask = 0 if np.random.randn(1) > opt.supervision_rate else 1

# Optimizers
optimizer = torch.optim.Adam(
    itertools.chain(Enc.parameters(), Dec.parameters(), Res.parameters(), Cls.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)


# Learning rate update schedulers
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)


# Assign different roots of each dataset (only zenodo)
print("Loading dataset %s and env %s for semi-supervised training." % (opt.dataset_name, opt.dataset_env))
root = './data/data_zenodo/dataset.pkl'


# Assign data for training
data_train, data_test, _, _ = err_mitigation_dataset(
    root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env, split_factor=0.8, scaling=True, mode=opt.mode
)

# Configure dataloaders
dataloader = DataLoader(
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


# ===================================
#           Training
# ===================================

prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):

    # initialization evaluation metrics
    rmse_error = 0.0
    abs_error = 0.0
    accuracy = 0.0
    start_time = time.time()

    for i, batch in enumerate(dataloader):

        # Set model input
        cir_gt = batch["CIR"]
        err_gt = batch["Err"]
        label_gt = batch["Label"]
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()
            label_gt = label_gt.to(device=device, dtype=torch.int64)

        # Start training
        optimizer.zero_grad()

        # Get latent representation
        range_code, env_code, env_code_rv, kl_div = Enc(cir_gt)

        # 1) Reconstructed cir
        cir_gen = Dec(range_code, env_code)

        # 2) Estimated ranging error
        err_fake = Res(range_code)

        # 3) Estimated env label (try rv later)
        label_fake = Cls(env_code)

        # Losses for semi-supervised learning (exist supervision with p=supervision_rate)
        # unsupervied terms
        loss_ae = lambda_ae * criterion_recon(cir_gt, cir_gen)
        loss_range = lambda_range * kl_div
        # supervised terms
        label_gt = label_gt.squeeze()
        loss_res = mask * lambda_res * criterion_recon(err_gt, err_fake)
        if opt.dataset_env == 'room_full':  # 0~4
            loss_env = mask * lambda_env * criterion_code(label_fake, label_gt)
        else:  # 0~9 for cross_entropy
            loss_env = mask * lambda_env * criterion_code(label_fake - 1, label_gt - 1)

        # Total loss
        loss = loss_ae + loss_range + loss_range + loss_env

        loss.backward()
        optimizer.step()

        # -------------------
        #   Log Progress
        # -------------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

        # Evaluation
        with torch.no_grad():
            # ranging mitigation error
            rmse_error += (torch.mean((err_fake - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_fake - err_gt))
            time_train = (time.time() - prev_time) / opt.batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_train / (i + 1)
            # env classification error
            if opt.dataset_env == 'room_full':  # 0~4
                prediction = torch.argmax(label_fake, dim=1)  # (b, num_classes) -> (b, 1)
            else:
                prediction = torch.argmax(label_fake, dim=1) + 1
            accuracy += torch.sum(prediction == label_gt).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

        # Print log
        sys.stdout.write(
            "\r[Model Name: C%d_%s_semi%f] [Epoch: %d/%d] [Batch %d/%d] [RMSE: %F] [ABS ERROR: %F] [Accuracy: %f] [Train Time: %f] \
            [Total loss: %f] [Supervised loss: ae %f, kl %f] [Unsup loss: res %f, cls %f] ETA: %s"
            % (opt.conv_type, opt.restorer_type, opt.supervision_rate,epoch, opt.n_epochs, i, len(dataloader), rmse_avg, abs_avg, accuracy_avg, time_avg, \
            loss.item(), loss_ae.item(), loss_range.item(), loss_res.item(), loss_env.item(), time_left)
        )
        logging.info(
            "\r[Model Name: C%d_%s_semi%f] [Epoch: %d/%d] [Batch %d/%d] [RMSE: %F] [ABS ERROR: %F] [Accuracy: %f] [Train Time: %f] \
            [Total loss: %f] [Supervised loss: ae %f, kl %f] [Unsup loss: res %f, cls %f] ETA: %s"
            % (opt.conv_type, opt.restorer_type, opt.supervision_rate,epoch, opt.n_epochs, i, len(dataloader), rmse_avg, abs_avg, accuracy_avg, time_avg, \
            loss.item(), loss_ae.item(), loss_range.item(), loss_res.item(), loss_env.item(), time_left)
        )

    # Update learning rates
    lr_scheduler.step()

    # If at sample interval visulaize results
    # if epoch % opt.sample_interval == 0:
    #     with torch.no_grad():
    #         visualize_recon(result_path, epoch, dataloader_test, Enc, Dec)
    #         CDF_plot(opt, root, result_path, epoch, dataloader_test, Enc, Res, num=200, use_competitor=True)
    #         visualize_latent(result_path, epoch, dataloader_test, opt.dataset_env, Enc, Cls, title=None)

    # If at check interval save models
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(Enc.state_dict(), os.path.join(model_path, "Enc_%d.pth" % epoch))
        torch.save(Dec.state_dict(), os.path.join(model_path, "Dec_%d.pth" % epoch))
        torch.save(Res.state_dict(), os.path.join(model_path, "Res_%d.pth" % epoch))
        torch.save(Cls.state_dict(), os.path.join(model_path, "Cls_%d.pth" % epoch))
