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


def train_vl(opt, network, device, result_path, model_path, dataloader, val_dataloader, optimizer, lr_scheduler, data):
    network.train()

    # Save training log
    logging.basicConfig(filename=os.path.join(result_path, 'training_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "Network_%d.pth" % epoch)))
    else:
        network.apply(weights_init_normal)

    # Set loss function
    criterion_ae = torch.nn.L1Loss().to(device)
    criterion_reg = torch.nn.L1Loss().to(device)
    criterion_idy = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_ae = 1
    lambda_res = 10
    lambda_idy = 1
    lambda_reg = 1

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
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

            # Generate estimations
            label_est, env_latent, err_est = network(cir_gt)

            # Loss terms
            label_gt = label_gt.squeeze()  # 0~n-1 for cross-entropy
            loss_idy = lambda_idy * criterion_idy(label_est, label_gt)
            loss_reg = lambda_reg * criterion_reg(err_est, err_gt)

            # Total loss
            loss = loss_idy + loss_reg

            loss.backward()
            optimizer.step()

            # ------ log process ------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = time.time() - start_time
                rmse_avg = rmse_error / (i + 1)
                abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Data Env: %s] [Model Type: Identifier%s_Regressor%s] [Epoch: %d/%d] [Batch: %d/%d] \
                [Total Loss: %f, Idy Loss: %f, Reg Loss: %f] [Error: rmse %f, abs %f, accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, opt.regressor_type, epoch, opt.n_epochs, i, len(dataloader), \
                loss.item(), loss_idy.item(), loss_reg.item(), rmse_avg, abs_avg, accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "\r[Data Env: %s] [Model Type: Identifier%s_Regressor%s] [Epoch: %d/%d] [Batch: %d/%d] \
                [Total Loss: %f, Idy Loss: %f, Reg Loss: %f] [Error: rmse %f, abs %f, accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, opt.regressor_type, epoch, opt.n_epochs, i, len(dataloader), \
                loss.item(), loss_idy.item(), loss_reg.item(), rmse_avg, abs_avg, accuracy_avg, time_avg, time_left)
            )

        # Update learning rate
        lr_scheduler.step()

        # Illustrate results on test set
        if epoch % opt.sample_interval == 0:
            test_vl(opt, network, device, result_path, model_path, 
                val_dataloader, epoch, data
        )  # epoch for val and opt.test_epoch for test
        
        # Save models at checkpoint
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(network.load_state_dict(), os.path.join(model_path, "Network_%d.pth" % epoch))


def train_vl_semi(opt, network, device, result_path, model_path, dataloader, val_dataloader, optimizer, lr_scheduler, data):
    network.train()

    # Save training log
    logging.basicConfig(filename=os.path.join(result_path, 'semi_training_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "Network_%d.pth" % epoch)))
    else:
        network.apply(weights_init_normal)

    # Set loss function
    criterion_ae = torch.nn.L1Loss().to(device)
    criterion_reg = torch.nn.L1Loss().to(device)
    criterion_idy = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_ae = 1
    lambda_res = 10
    lambda_idy = 1
    lambda_reg = 1

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
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

            # Generate estimations
            label_est, env_latent, err_est = network(cir_gt)

            # Loss terms
            label_gt = label_gt.squeeze()  # 0~n-1 for cross-entropy
            loss_idy = lambda_idy * criterion_idy(label_est, label_gt)
            loss_reg = lambda_reg * criterion_reg(err_est, err_gt)

            # Total loss
            loss = loss_idy + loss_reg

            loss.backward()
            optimizer.step()

            # ------ log process ------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = time.time() - start_time
                rmse_avg = rmse_error / (i + 1)
                abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Data Env: %s] [Model Type: Identifier%s_Regressor%s] [Epoch: %d/%d] [Batch: %d/%d] \
                [Total Loss: %f, Idy Loss: %f, Reg Loss: %f] [Error: rmse %f, abs %f, accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, opt.regressor_type, epoch, opt.n_epochs, i, len(dataloader), \
                loss.item(), loss_idy.item(), loss_reg.item(), rmse_avg, abs_avg, accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "\r[Data Env: %s] [Model Type: Identifier%s_Regressor%s] [Epoch: %d/%d] [Batch: %d/%d] \
                [Total Loss: %f, Idy Loss: %f, Reg Loss: %f] [Error: rmse %f, abs %f, accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, opt.regressor_type, epoch, opt.n_epochs, i, len(dataloader), \
                loss.item(), loss_idy.item(), loss_reg.item(), rmse_avg, abs_avg, accuracy_avg, time_avg, time_left)
            )

        # Update learning rate
        lr_scheduler.step()

        # Illustrate results on test set
        if epoch % opt.sample_interval == 0:
            test_vl(opt, network, device, result_path, model_path, 
                val_dataloader, epoch, data
        )  # epoch for val and opt.test_epoch for test
        
        # Save models at checkpoint
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(network.load_state_dict(), os.path.join(model_path, "Network_%d.pth" % epoch))
