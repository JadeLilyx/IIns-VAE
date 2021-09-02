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


def train_vl_naive(opt, device, result_path, model_path, dataloader, val_dataloader, optimizer, scheduler, enc, dec, idy, reg, data_raw):

    # Save training log
    setproctitle.setproctitle("UWB_VL_normal")
    logging.basicConfig(filename=os.path.join(result_path, 'training_AE%dIdy%dReg%d.log' % (opt.conv_type, opt.identifier_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % opt.epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % opt.epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % opt.epoch)))
        reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % opt.epoch)))
    else:
        enc.apply(weights_init_normal)
        dec.apply(weights_init_normal)
        idy.apply(weights_init_normal)
        reg.apply(weights_init_normal)

    # Set loss function
    criterion_recon = torch.nn.L1Loss().to(device)
    criterion_code = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_ae = 1
    lambda_reg = 10
    lambda_env = 1
    lambda_kl = 1

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
        rmse_error = 0.0
        abs_error = 0.0
        accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):

            # Set model input
            cir_gt = batch["CIR"]  # (b, 152/157)
            err_gt = batch["Err"]  # (b, 1)
            label_gt = batch["Label"]  # (b, 1)
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                err_gt = err_gt.cuda()
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer.zero_grad()

            # Generate latent representations
            range_code, env_code, env_code_rv, kl_div = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Loss terms
            loss_ae = lambda_ae * criterion_recon(cir_gt, cir_gen)
            loss_reg = lambda_reg * criterion_recon(err_gt, err_est)
            label_gt = label_gt.squeeze()  # value 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            loss_env = lambda_env * criterion_code(label_est, label_gt)
            loss_kl = lambda_kl * kl_div  # env latent space

            # Total loss
            loss = loss_ae + loss_reg + loss_env + loss_kl

            loss.backward()
            optimizer.step()

            # -------------- log process -------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = (time.time() - start_time)
                rmse_avg = rmse_error / (i + 1)
                abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]  # already squeeze gt label above
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Data: %s/%s/%s] [Model Type: AE%d/Idy%d/Reg%d] "
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %3.3f, Range loss: %.3f, Env loss: %.3f] "
                "[RMSE: %f, MAE: %f, ACC: %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type,
                epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_reg.item(), loss_env.item(),
                rmse_avg, abs_avg, accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %.3f, Range loss: %.3f, Env loss: %.3f] "
                "[RMSE: %f, MAE: %f, ACC: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_reg.item(), loss_env.item(),
                rmse_avg, abs_avg, accuracy_avg)
            )

        # Update learning rate (Optional)
        scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(enc.state_dict(), os.path.join(model_path, "Enc_%d.pth" % (epoch + 1)))
            torch.save(dec.state_dict(), os.path.join(model_path, "Dec_%d.pth" % (epoch + 1)))
            torch.save(idy.state_dict(), os.path.join(model_path, "Idy_%d.pth" % (epoch + 1)))
            torch.save(reg.state_dict(), os.path.join(model_path, "Reg_%d.pth" % (epoch + 1)))
        
        # Illustrate results on test data if at sample epoch
        if (epoch + 1) % opt.sample_interval == 0:
            test_vl_naive(
                opt=opt, device=device, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, enc=enc, dec=dec, idy=idy, reg=reg, test_epoch=epoch + 1, data_raw=data_raw,
                test_flag=False
            )  # epoch for val and test_epoch for test


def train_vl_semi(opt, device, result_path, model_path, dataloader, val_dataloader, optimizer, scheduler, enc, dec, idy, reg, data_raw):

    # Save training log
    setproctitle.setproctitle("UWB_VL_semi")
    logging.basicConfig(filename=os.path.join(result_path, 'training_semi_AE%dIdy%dReg%d.log' % (opt.conv_type, opt.identifier_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "EncE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, opt.epoch))))
        dec.load_state_dict(torch.load(os.path.join(model_path, "DecE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, opt.epoch))))
        idy.load_state_dict(torch.load(os.path.join(model_path, "IdyE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, opt.epoch))))
        reg.load_state_dict(torch.load(os.path.join(model_path, "RegE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, opt.epoch))))
    else:
        enc.apply(weights_init_normal)
        dec.apply(weights_init_normal)
        idy.apply(weights_init_normal)
        reg.apply(weights_init_normal)

    # Set loss function
    criterion_recon = torch.nn.L1Loss().to(device)
    criterion_code = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_ae = 1
    lambda_reg = 10
    lambda_env = 1
    lambda_kl = 1

    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
        rmse_error = 0.0
        abs_error = 0.0
        accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):

            # Set model input
            cir_gt = batch["CIR"]  # (b, 152/157)
            err_gt = batch["Err"]  # (b, 1)
            label_gt = batch["Label"]  # (b, 1)
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                err_gt = err_gt.cuda()
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer.zero_grad()

            # Generate latent representations
            range_code, env_code, env_code_rv, kl_div = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Loss terms
            # 1) unsupervised loss
            loss_vae = lambda_ae * criterion_recon(cir_gt, cir_gen) + lambda_kl * kl_div  # env latent space
            # mask part of the data
            mask_e = 0 if np.random.randn(1) > opt.sup_rate_e else 1
            mask_r = 0 if np.random.randn(1) > opt.sup_rate_r else 1
            # if mask == 0:  # dont use labels in this batch
            #     # Warm-up (vae loss only)
            #     loss_vae.backward()
            #     optimizer.step()
            #     continue

            # 2) supervised loss
            loss_reg = lambda_reg * criterion_recon(err_gt, err_est)
            label_gt = label_gt.squeeze()  # value 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            loss_env = lambda_env * criterion_code(label_est, label_gt)

            # Total loss with mask
            loss = loss_vae + loss_reg * mask_r + loss_env * mask_e

            loss.backward()
            optimizer.step()

            # -------------- log process -------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = (time.time() - start_time)
                rmse_avg = rmse_error / (i + 1)
                abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]  # already squeeze gt label above
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Data: %s/%s/%s] [Model Type: AE%d/Idy%d/Reg%d Semi: R%.2fE%.2f] "
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %.3f, Unsup loss: %.3f, Sup loss: %.3f] "
                "[RMSE: %f, MAE: %f, ACC: %f] [Train Time: %f, ETA: %s]"
                % (opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type,
                opt.sup_rate_r, opt.sup_rate_e,
                epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_vae.item(), (loss_reg+loss_env).item(),
                rmse_avg, abs_avg, accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %.3f, Unsup loss: %.3f, Sup loss: %.3f] "
                "[RMSE: %f, MAE: %f, ACC: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_vae.item(), (loss_reg+loss_env).item(),
                rmse_avg, abs_avg, accuracy_avg)
            )

        # Update learning rate (Optional)
        scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(enc.state_dict(), os.path.join(model_path, "EncE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, (epoch + 1))))
            torch.save(dec.state_dict(), os.path.join(model_path, "DecE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, (epoch + 1))))
            torch.save(idy.state_dict(), os.path.join(model_path, "IdyE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, (epoch + 1))))
            torch.save(reg.state_dict(), os.path.join(model_path, "RegE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, (epoch + 1))))
        
        # Illustrate results on test data if at sample epoch
        if (epoch + 1) % opt.sample_interval == 0:
            test_vl_semi(
                opt=opt, device=device, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, enc=enc, dec=dec, idy=idy, reg=reg, test_epoch=epoch + 1, data_raw=data_raw,
                test_flag=False
            )  # epoch for val and test_epoch for test


def train_vl_DeIdy(opt, device, result_path, model_path, dataloader, val_dataloader, optimizer, scheduler, enc, dec, reg, data_raw):

    # Save training log
    setproctitle.setproctitle("UWB_VL_abtype2")
    logging.basicConfig(filename=os.path.join(result_path, 'trainingAb2_AE%dReg%d.log' % (opt.conv_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % opt.epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % opt.epoch)))
        # idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % opt.epoch)))
        reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % opt.epoch)))
    else:
        enc.apply(weights_init_normal)
        dec.apply(weights_init_normal)
        # idy.apply(weights_init_normal)
        reg.apply(weights_init_normal)

    # Set loss function
    criterion_recon = torch.nn.L1Loss().to(device)
    criterion_code = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_ae = 1
    lambda_kl = 1
    lambda_reg = 10
    # lambda_env = 1
    
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
        rmse_error = 0.0
        abs_error = 0.0
        accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):

            # Set model input
            cir_gt = batch["CIR"]  # (b, 152/157)
            err_gt = batch["Err"]  # (b, 1)
            label_gt = batch["Label"]  # (b, 1)
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                err_gt = err_gt.cuda()
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer.zero_grad()

            # Generate latent representations
            range_code, env_code, env_code_rv, kl_div = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            # label_est = idy(env_code_rv)

            # Loss terms
            loss_vae = lambda_ae * criterion_recon(cir_gt, cir_gen) + lambda_kl * kl_div  # env latent space
            loss_reg = lambda_reg * criterion_recon(err_gt, err_est)
            label_gt = label_gt.squeeze()  # value 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            # loss_env = lambda_env * criterion_code(label_est, label_gt)

            # Total loss
            loss = loss_vae + loss_reg   # + loss_env

            loss.backward()
            optimizer.step()

            # -------------- log process -------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = (time.time() - start_time)
                rmse_avg = rmse_error / (i + 1)
                abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # # env label estimation
                # prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                # accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]  # already squeeze gt label above
                # accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/IdyNone/Reg%d] "
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %3.3f, Range loss: %.3f, VAE loss: %.3f] "
                "[RMSE: %f, MAE: %f] [Train Time: %f, ETA: %s]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.regressor_type,
                epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_reg.item(), loss_vae.item(),
                rmse_avg, abs_avg, time_avg, time_left)
            )
            logging.info(
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %.3f, Range loss: %.3f, VAE loss: %.3f] "
                "[RMSE: %f, MAE: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_reg.item(), loss_vae.item(),
                rmse_avg, abs_avg)
            )

        # Update learning rate (Optional)
        scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(enc.state_dict(), os.path.join(model_path, "Enc_%d.pth" % (epoch + 1)))
            torch.save(dec.state_dict(), os.path.join(model_path, "Dec_%d.pth" % (epoch + 1)))
            # torch.save(idy.state_dict(), os.path.join(model_path, "Idy_%d.pth" % (epoch + 1)))
            torch.save(reg.state_dict(), os.path.join(model_path, "Reg_%d.pth" % (epoch + 1)))
        
        # Illustrate results on test data if at sample epoch
        if (epoch + 1) % opt.sample_interval == 0:
            test_vl_DeIdy(
                opt=opt, device=device, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, enc=enc, dec=dec, idy=idy, reg=reg, test_epoch=epoch + 1, data_raw=data_raw,
                test_flag=False
            )  # epoch for val and test_epoch for test


def train_vl_DeReg(opt, device, result_path, model_path, dataloader, val_dataloader, optimizer, scheduler, enc, dec, idy, data_raw):

    # Save training log
    setproctitle.setproctitle("UWB_VL_abtype3")
    logging.basicConfig(filename=os.path.join(result_path, 'trainingAb3_AE%dIdy%d.log' % (opt.conv_type, opt.identifier_type)), level=logging.INFO)
    # logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % opt.epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % opt.epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % opt.epoch)))
        # reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % opt.epoch)))
    else:
        enc.apply(weights_init_normal)
        dec.apply(weights_init_normal)
        idy.apply(weights_init_normal)
        # reg.apply(weights_init_normal)

    # Set loss function
    criterion_recon = torch.nn.L1Loss().to(device)
    criterion_code = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_ae = 1
    lambda_kl = 1
    # lambda_reg = 10
    lambda_env = 1
    
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
        rmse_error = 0.0
        abs_error = 0.0
        accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):

            # Set model input
            cir_gt = batch["CIR"]  # (b, 152/157)
            err_gt = batch["Err"]  # (b, 1)
            label_gt = batch["Label"]  # (b, 1)
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                err_gt = err_gt.cuda()
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer.zero_grad()

            # Generate latent representations
            range_code, env_code, env_code_rv, kl_div = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            # err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Loss terms
            loss_vae = lambda_ae * criterion_recon(cir_gt, cir_gen) + lambda_kl * kl_div  # env latent space
            # loss_reg = lambda_reg * criterion_recon(err_gt, err_est)
            label_gt = label_gt.squeeze()  # value 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            loss_env = lambda_env * criterion_code(label_est, label_gt)

            # Total loss
            loss = loss_vae + loss_env   # + loss_reg

            loss.backward()
            optimizer.step()

            # -------------- log process -------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                # rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                # abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = (time.time() - start_time)
                # rmse_avg = rmse_error / (i + 1)
                # abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]  # already squeeze gt label above
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/Idy%d/RegNone] "
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %3.3f, Env loss: %.3f, VAE loss: %.3f] "
                "[ACC: %f] [Train Time: %f, ETA: %s]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type,
                epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_env.item(), loss_vae.item(),
                accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %.3f, Env loss: %.3f, VAE loss: %.3f] "
                "[ACC: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_env.item(), loss_vae.item(),
                accuracy_avg)
            )

        # Update learning rate (Optional)
        scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(enc.state_dict(), os.path.join(model_path, "Enc_%d.pth" % (epoch + 1)))
            torch.save(dec.state_dict(), os.path.join(model_path, "Dec_%d.pth" % (epoch + 1)))
            torch.save(idy.state_dict(), os.path.join(model_path, "Idy_%d.pth" % (epoch + 1)))
            # torch.save(reg.state_dict(), os.path.join(model_path, "Reg_%d.pth" % (epoch + 1)))
        
        # Illustrate results on test data if at sample epoch
        if (epoch + 1) % opt.sample_interval == 0:
            test_vl_DeReg(
                opt=opt, device=device, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, enc=enc, dec=dec, idy=idy, test_epoch=epoch + 1, data_raw=data_raw,
                test_flag=False
            )  # epoch for val and test_epoch for test


def train_vl_DeDec(opt, device, result_path, model_path, dataloader, val_dataloader, optimizer, scheduler, enc, idy, reg, data_raw):

    # Save training log
    setproctitle.setproctitle("UWB_VL_abtype4")
    logging.basicConfig(filename=os.path.join(result_path, 'trainingAb4_AE%dIdy%dReg%d.log' % (opt.conv_type, opt.identifier_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load pre-trained model or initialize weights
    if opt.epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % opt.epoch)))
        # dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % opt.epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % opt.epoch)))
        reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % opt.epoch)))
    else:
        enc.apply(weights_init_normal)
        # dec.apply(weights_init_normal)
        idy.apply(weights_init_normal)
        reg.apply(weights_init_normal)

    # Set loss function
    criterion_recon = torch.nn.L1Loss().to(device)
    criterion_code = torch.nn.CrossEntropyLoss().to(device)

    # Loss weights
    lambda_ae = 1
    lambda_kl = 1
    lambda_reg = 10
    lambda_env = 1
    
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        # Initialization evaluation metrics
        rmse_error = 0.0
        abs_error = 0.0
        accuracy = 0.0
        start_time = time.time()

        for i, batch in enumerate(dataloader):

            # Set model input
            cir_gt = batch["CIR"]  # (b, 152/157)
            err_gt = batch["Err"]  # (b, 1)
            label_gt = batch["Label"]  # (b, 1)
            if torch.cuda.is_available():
                cir_gt = cir_gt.cuda()
                err_gt = err_gt.cuda()
                label_gt = label_gt.to(device=device, dtype=torch.int64)  # torch.LongTensor

            # Start training
            optimizer.zero_grad()

            # Generate latent representations
            range_code, env_code, env_code_rv, kl_div = enc(cir_gt)
            # 1) reconstruct cir
            # cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Loss terms
            # loss_vae = lambda_ae * criterion_recon(cir_gt, cir_gen) + lambda_kl * kl_div  # env latent space
            loss_enc = lambda_kl * kl_div
            loss_reg = lambda_reg * criterion_recon(err_gt, err_est)
            label_gt = label_gt.squeeze()  # value 0~n-1 for cross-entropy
            # label_gt = label_gt.type(torch.LongTensor)
            loss_env = lambda_env * criterion_code(label_est, label_gt)

            # Total loss
            loss = loss_enc + loss_env + loss_reg

            loss.backward()
            optimizer.step()

            # -------------- log process -------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time))

            # Evaluation
            with torch.no_grad():
                # range error estimation
                rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
                abs_error += torch.mean(torch.abs(err_est - err_gt))
                time_train = (time.time() - start_time)
                rmse_avg = rmse_error / (i + 1)
                abs_avg = abs_error / (i + 1)
                time_avg = time_train / (i + 1)
                # env label estimation
                prediction = torch.argmax(label_est, dim=1)  # (b, num_classes)
                accuracy += torch.sum(prediction==label_gt).float() / label_gt.shape[0]  # already squeeze gt label above
                accuracy_avg = accuracy / (i + 1)

            # Print log
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/Idy%d/Reg%d] "
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %3.3f, Env loss: %.3f, Range loss: %.3f] "
                "[RMSE: %f, MAE: %f, ACC: %f] [Train Time: %f, ETA: %s]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type,
                epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_env.item(), loss_reg.item(),
                rmse_avg, abs_avg, accuracy_avg, time_avg, time_left)
            )
            logging.info(
                "[Epoch: %d/%d] [Batch: %d/%d] [Total loss: %.3f, Env loss: %.3f, Range loss: %.3f] "
                "[RMSE: %f, MAE: %f, ACC: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss.item(), loss_env.item(), loss_reg.item(),
                rmse_avg, abs_avg, accuracy_avg)
            )

        # Update learning rate (Optional)
        scheduler.step()

        # Save models if at checkpoint epoch
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            torch.save(enc.state_dict(), os.path.join(model_path, "Enc_%d.pth" % (epoch + 1)))
            # torch.save(dec.state_dict(), os.path.join(model_path, "Dec_%d.pth" % (epoch + 1)))
            torch.save(idy.state_dict(), os.path.join(model_path, "Idy_%d.pth" % (epoch + 1)))
            torch.save(reg.state_dict(), os.path.join(model_path, "Reg_%d.pth" % (epoch + 1)))
        
        # Illustrate results on test data if at sample epoch
        if (epoch + 1) % opt.sample_interval == 0:
            test_vl_DeDec(
                opt=opt, device=device, result_path=result_path, model_path=model_path, 
                dataloader=val_dataloader, enc=enc, idy=idy, reg=Reg, test_epoch=epoch + 1, data_raw=data_raw,
                test_flag=False
            )  # epoch for val and test_epoch for test