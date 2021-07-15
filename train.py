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
from model import *
from baseline import svm_regressor
from test import *

import setproctitle
import logging


def train_gem(opt, device, tensor, result_path, model_path, dataloader, val_dataloader, optimizer, network, data_raw):
    
    # Save training log
    logging.basicConfig(filename=os.path.join(result_path, 'training_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load pre-trained model or initialize weights
    # if opt.net_ablation == 'detach':
    #     if opt.epoch != 0:
    #         enet.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % opt.epoch)))
    #         mnet.load_state_dict(torch.load(os.path.join(model_path, "MNet_%d.pth" % opt.epoch)))
    #     else:
    #         enet.apply(weights_init_normal)
    #         mnet.apply(weights_init_normal)
    # else:
    if opt.epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "Network_%d.pth" % opt.epoch)))
    else:
        network.apply(weights_init_normal)

    # Set loss function
    criterion_reg = torch.nn.L1Loss().to(device)
    criterion_idy = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
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
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer.zero_grad()

            # Generate estimations
            # if opt.net_ablation == 'detach':
            #     label_est, env_latent = enet(cir_gt)
            #     err_est = mnet(env_latent, cir_gt)
            # else:
            label_est, env_latent, err_est = network(cir_gt)

            # Loss terms
            label_gt = label_gt.squeeze()  # 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
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
        # lr_scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # if opt.net_ablation == 'detach':
            #     torch.save(enet.state_dict(), os.path.join(model_path, "ENet_%d.pth" % epoch))
            #     torch.save(mnet.state_dict(), os.path.join(model_path, "MNet_%d.pth" % epoch))
            # else:
            torch.save(network.state_dict(), os.path.join(model_path, "Network_%d.pth" % epoch))

        # # Illustrate results on test set if at sample interval epoch
        if epoch % opt.sample_interval == 0:
            # with torch.no_grad():
            #     visualize_latents()
            #     CDF_plot()
            # if opt.net_ablation == 'detach':
            #     test_gem(
            #         opt=opt, device=device, tensor=tensor, result_path=result_path, model_path=model_path, 
            #         dataloader=val_dataloader, enet=enet, mnet=mnet, epoch=epoch, data_raw=data_raw
            #     )  # epoch for val and opt.test_epoch for test
            # else:
            test_gem(
                opt=opt, device=device, tensor=tensor, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, network=network, epoch=epoch, data_raw=data_raw
            )  # epoch for val and opt.test_epoch for test


def train_gem_sepE(opt, device, tensor, result_path, model_path, dataloader, val_dataloader, optimizer_e, data_raw, enet):
    
    # Save training log
    logging.basicConfig(filename=os.path.join(result_path, 'training_logE.log'), level=logging.INFO)
    logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        enet.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % opt.epoch)))
    else:
        enet.apply(weights_init_normal)

    # Set loss function
    # criterion_reg = torch.nn.L1Loss().to(device)
    criterion_idy = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_idy = 1
    # lambda_reg = 1

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
        # rmse_error = 0.0
        # abs_error = 0.0
        accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            
            # Set model input
            cir_gt = batch["CIR"]
            # err_gt = batch["Err"]
            label_gt = batch["Label"]
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                # err_gt = err_gt.cuda()
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer_e.zero_grad()

            # Generate estimations
            label_est = enet(cir_gt)
            # err_est = mnet(env_latent, cir_gt)

            # Loss terms
            label_gt = label_gt.squeeze()  # 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            loss_idy = lambda_idy * criterion_idy(label_est, label_gt)
            # loss_reg = lambda_reg * criterion_reg(err_est, err_gt)

            # Total loss
            loss_e = loss_idy  # + loss_reg

            loss_e.backward()
            optimizer_e.step()

            # ------ log process ------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                # rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                # abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = time.time() - start_time
                # rmse_avg = rmse_error / (i + 1)
                # abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Sep: Data Env: %s] [Model Type: Identifier%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Sep Idy Loss: %f] [accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, epoch, opt.n_epochs, i, len(dataloader),
                loss_e.item(), accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "\r[Sep: Data Env: %s] [Model Type: Identifier%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Sep Idy Loss: %f] [accuracy %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.identifier_type, epoch, opt.n_epochs, i, len(dataloader),
                loss_e.item(), accuracy_avg, time_avg, time_left)
            )
        
        # Update learning rate
        # lr_scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(enet.state_dict(), os.path.join(model_path, "ENet_%d.pth" % epoch))

        # # Illustrate results on test set if at sample interval epoch
        if epoch % opt.sample_interval == 0:
            # with torch.no_grad():
            #     visualize_latents()
            #     CDF_plot()
            test_gem_sepE(
                opt=opt, device=device, tensor=tensor, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, enet=enet, epoch=epoch, data_raw=data_raw
            )  # epoch for val and opt.test_epoch for test
            

def train_gem_sepM(opt, device, tensor, result_path, model_path, dataloader, val_dataloader, optimizer_m, enet, mnet, data_raw):
    
    # Save training log
    logging.basicConfig(filename=os.path.join(result_path, 'training_logM.log'), level=logging.INFO)
    logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        mnet.load_state_dict(torch.load(os.path.join(model_path, "MNet_%d.pth" % opt.epoch)))
    else:
        mnet.apply(weights_init_normal)

    # Set loss function
    criterion_reg = torch.nn.L1Loss().to(device)
    # criterion_idy = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    # lambda_idy = 1
    lambda_reg = 1

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
        rmse_error = 0.0
        abs_error = 0.0
        # accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):
            
            # Set model input
            cir_gt = batch["CIR"]
            err_gt = batch["Err"]
            # label_gt = batch["Label"]
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                err_gt = err_gt.cuda()
                # label_gt = label_gt.to(device=device, dtype=torch.LongTensor)  # torch.int64

            # Start training
            optimizer_m.zero_grad()

            # Generate estimations
            # label_est, env_latent = enet(cir_gt)
            err_est = mnet(cir_gt, label_gt)  # instead of none or label_est/env_latent
            # not that in testing use 0/1 instead

            # Loss terms
            # label_gt = label_gt.squeeze()  # 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            # loss_idy = lambda_idy * criterion_idy(label_est, label_gt)
            loss_reg = lambda_reg * criterion_reg(err_est, err_gt)

            # Total loss
            loss_m = loss_reg  # loss_idy

            loss_m.backward()
            optimizer_m.step()

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
                # prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                # accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]
                # accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Sep Data Env: %s] [Model Type: Regressor%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Reg Loss: %f] [Error: rmse %f, abs %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.regressor_type, epoch, opt.n_epochs, i, len(dataloader),
                loss_m.item(), loss_reg.item(), rmse_avg, abs_avg, time_avg, time_left)
            )
            logging.info(
                "\r[Sep Data Env: %s] [Model Type: Regressor%s] [Epoch: %d/%d] [Batch: %d/%d]"
                "[Reg Loss: %f] [Error: rmse %f, abs %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_env, opt.regressor_type, epoch, opt.n_epochs, i, len(dataloader),
                loss_m.item(), loss_reg.item(), rmse_avg, abs_avg, time_avg, time_left)
            )
        
        # Update learning rate
        # lr_scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(mnet.state_dict(), os.path.join(model_path, "MNet_%d.pth" % epoch))

        # # Illustrate results on test set if at sample interval epoch
        if epoch % opt.sample_interval == 0:
            # with torch.no_grad():
            #     visualize_latents()
            #     CDF_plot()
            test_gem_sepEM(
                opt=opt, device=device, tensor=tensor, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, enet=enet, mnet=mnet, epoch=epoch, data_raw=data_raw
            )  # epoch for val and opt.test_epoch for test
            
