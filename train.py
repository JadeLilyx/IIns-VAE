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


# Set loss function
criterion_recon = torch.nn.L1Loss().to(device)
criterion_code = torch.nn.CrossEntropyLoss().to(device)

# Loss weights
lambda_ae = 1
lambda_res = 10
lambda_range = 1
lambda_env = 1


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


# Create sample and checkpoint directories (later may add opt.dataset_env)
print("conv type: ", opt.conv_type)
model_path = "./saved_models/%s_mode_%s/AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)
result_path = "./saved_results/%s_mode_%s/AE%d_Res%s_Cls%s_Rdim%dEdim%d" % (opt.dataset_env, opt.mode, opt.conv_type, opt.restorer_type, opt.classifier_type, opt.range_dim, opt.env_dim)
os.makedirs(result_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

# Save training log
logging.basicConfig(filename=os.path.join(result_path, 'train_log.log'), level=logging.INFO)
logging.info("Started")

# Load pre-trained model or initialize weights
if opt.epoch != 0:
    # Enc.load_state_dict(torch.load("saved_models/AEType%d_ResType%s_%s_dim%d/Enc_%d.pth" % (opt.conv_type, opt.restorer_type, opt.dataset_name, opt.range_dim, opt.epoch)))
    Enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % opt.epoch)))
    Dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % opt.epoch)))
    Res.load_state_dict(torch.load(os.path.join(model_path, "Res_%d.pth" % opt.epoch)))
    Cls.load_state_dict(torch.load(os.path.join(model_path, "Cls_%d.pth" % opt.epoch)))
else:
    Enc.apply(weights_init_normal)
    Dec.apply(weights_init_normal)
    Res.apply(weights_init_normal)
    Cls.apply(weights_init_normal)


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


# Assign different roots of each dataset
# (recommend to only use zenodo temporally)
root = []
if opt.dataset_name == 'ewine':  # load_reg_data(filepaths=root)
    print("Loading dataset %s for training." % opt.dataset_name)
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
    # err_reg, cir_reg = load_reg_data(root)
    # print("Shape of error: ", err_reg.shape)  # (31489, 1)
    # print("Shape of cir sample: ", cir_reg.shape)  # (31489, 152)
elif opt.dataset_name == 'zenodo':  # load_pkl_data(filepath=root)
    print("Loading dataset %s and env %s for training." % (opt.dataset_name, opt.dataset_env))
    root = './data/data_zenodo/dataset.pkl'
    # err_reg, cir_reg, label_reg = load_pkl_data(root, option=opt.dataset_env)
    # print("Shape of error: ", err_reg.shape)  # (55158, 1)
    # print("Shape of cir sample: ", cir_reg.shape)  # (55158, 152)
else:
    raise RuntimeError("Unknown dataset for usage.")


# Assign data for training
data_train, data_test, _, _ = err_mitigation_dataset(
    root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env, split_factor=opt.split_factor, scaling=True, mode=opt.mode, feature_flag=False
)
# print("check length: ", len(data_train[0]))  # train - 25191/44126, test - 6298/11032


# Configure dataloaders
dataloader = DataLoader(
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


# ===============================
#          Training
# ===============================


prev_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs):
    
    # Initialization evaluation metrics
    rmse_error = 0.0
    abs_error = 0.0
    accuracy = 0.0  
    start_time = time.time()

    for i, batch in enumerate(dataloader):

        # Set model input
        cir_gt = batch["CIR"]  # (B, 152/157)
        err_gt = batch["Err"]  # (B, 1) ~ ranging error
        label_gt = batch["Label"]  # (B, 1) ~ env label
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()
            # label_gt = label_gt.cuda()
            label_gt = label_gt.to(device=device, dtype=torch.int64)

        # Sampled rnv codes
        # env_rand = Variable(torch.randn(cir_gt.size(0), opt.env_dim, 1, 1).type(Tensor))

        # Start training
        optimizer.zero_grad()

        # Get latent representation (env_code_rv not used in training)
        range_code, env_code, env_code_rv, kl_div = Enc(cir_gt)

        # 1) Reconetructed cir
        cir_gen = Dec(range_code, env_code)  # env_rand

        # 2) Estimated ranging error
        # assert range_code.shape == range_code_shape, \
        #     "Wrong shape for range code, get {} but desire {}.".format(range_code.shape, range_code_shape)
        err_fake = Res(range_code)  # (b, 256, 38/39) -> (b, 1)

        # 3) Estimated env label (try env_code_rv)
        label_fake = Cls(env_code)  # (b, num_classes), one-hot encoding
        # print(label_fake.shape)
        # prediction = torch.argmax(label_fake, dim=1)
        # accuracy = torch.sum(prediction == label_gt).float() / label_gt.shape[0]

        # Losses
        loss_ae = lambda_ae * criterion_recon(cir_gt, cir_gen)  # data space
        loss_res = lambda_res * criterion_recon(err_gt, err_fake)  # error space
        loss_range = lambda_range * kl_div  # latent space
        # label_gt = torch.tensor(label_gt, dtype=torch.long, device=device)        
        label_gt = label_gt.squeeze()  # use [0, 1, 2] instead of [[0], [1], [2]]
        # print(label_fake[0], label_gt[0])
        # print(label_fake[0], label_gt[0])
        if opt.dataset_env == 'room_full':  # 0~4
            loss_env = lambda_env * criterion_code(label_fake, label_gt)
        else:  # 0~9 for cross_entropy
            loss_env = lambda_env * criterion_code(label_fake - 1, label_gt - 1)


        # Total loss
        loss = loss_ae + loss_res + loss_range + loss_env

        loss.backward()
        optimizer.step()

        # ---------------
        #  Log Progress
        # ---------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        
        # Evaluation
        with torch.no_grad():
            # ranging mitigation error
            rmse_error += (torch.mean((err_fake - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_fake - err_gt))
            time_train = time.time() - start_time
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
            # print("prediction: ", prediction[0: 10])
            # print("label_gt: ", label_gt[0: 10])
            # print("label_fake: ", label_fake[0: 10])
        
        # Print log
        sys.stdout.write(
            "\r[Model Name: C%d_%s] [Epoch: %d/%d] [Batch %d/%d] [RMSE: %f] [ABS ERROR: %f] [Accuracy: %f] [Train Time: %f] \
             [AE loss: %f] [Res loss: %f] [Range loss: %f, Env loss: %f] ETA: %s"
            % (opt.conv_type, opt.restorer_type, epoch, opt.n_epochs, i, len(dataloader), rmse_avg, abs_avg, accuracy_avg, time_avg, \
            loss_ae.item(), loss_res.item(), loss_range.item(), loss_env.item(), time_left)
        )
        logging.info(
            "\r[Model Name: C%d_%s] [Epoch: %d/%d] [Batch %d/%d] [RMSE: %f] [ABS ERROR: %f] [Accuracy: %f] [Train Time: %f] \
             [AE loss: %f] [Res loss: %f] [Range loss: %f, Env loss: %f] ETA: %s"
            % (opt.conv_type, opt.restorer_type, epoch, opt.n_epochs, i, len(dataloader), rmse_avg, abs_avg, accuracy_avg, time_avg, \
            loss_ae.item(), loss_res.item(), loss_range.item(), loss_env.item(), time_left)
        )

    # Update learning rates
    lr_scheduler.step()
    
    # If at sample interval visualize reconstructed cir
    if epoch % opt.sample_interval == 0:
        with torch.no_grad():
            visualize_recon(result_path, epoch, dataloader_test, Enc, Dec)
            CDF_plot(opt, root, result_path, epoch, dataloader_test, Enc, Res, num=200, use_competitor=True)
            visualize_latent(result_path, epoch, dataloader_test, opt.dataset_env, Enc, Cls, title=None)

    # If at check interval save models
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(Enc.state_dict(), os.path.join(model_path, "Enc_%d.pth" % epoch))
        torch.save(Dec.state_dict(), os.path.join(model_path, "Dec_%d.pth" % epoch))
        torch.save(Res.state_dict(), os.path.join(model_path, "Res_%d.pth" % epoch))
        torch.save(Cls.state_dict(), os.path.join(model_path, "Cls_%d.pth" % epoch))
