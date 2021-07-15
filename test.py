import os
import numpy as np
import math
import itertools
import datetime
import time
import sys
import scipy.io as io

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets

from utils import *
from dataset import *
from model import *
from baseline import *

import setproctitle
import logging


def test_gem(opt, device, tensor, result_path, model_path, dataloader, network, epoch, data_raw): 
    # different for val and test: result_path, epoch

    # Save experimental results (though also in train dir, different log file from 'train_log.log' nevertheless)
    logging.basicConfig(filename=os.path.join(result_path, 'val_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load models from path
    # if opt.net_ablation == 'detach':
    #     if epoch != 0:
    #         enet.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % opt.epoch)))
    #         mnet.load_state_dict(torch.load(os.path.join(model_path, "MNet_%d.pth" % opt.epoch)))
    #         enet.eval()
    #         mnet.eval()
    #     else:
    #         print("No saved models in dirs.")
    # else:
    if epoch != 0:
        network.load_state_dict(torch.load(os.path.join(model_path, "Network_%d.pth" % epoch)))
        network.eval()
    else:
        print("No saved models in dirs.")

    # Evaluation initialization
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

        with torch.no_grad():

            # Generate estimations
            # if opt.net_ablation == 'detach':
            #     label_est, env_latent = enet(cir_gt)
            #     err_est = mnet(env_latent, cir_gt)
            # else:
            label_est, env_latent, err_est = network(cir_gt)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500 # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            ## Illustrate figures
            # latent env arrays
            reduced_latents, labels = reduce_latents(env_latent, label_gt)  # not label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            # elif i != len(dataloader) - 1:
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # else:  # drop the last batch
            #     continue
            # range residual error arrays
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))
        
        # Print log
        sys.stdout.write(
            "\r[Data: %s/%s] [Model Type: Identifier%s_Regressor%s] [Test Epoch: %d] [Batch: %d/%d] \
            [Error: rmse %f, abs %f, accuracy %f] [Test Time: %f]"
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, opt.regressor_type, epoch, i, len(dataloader),
            rmse_avg, abs_avg, accuracy_avg, time_avg)
        )
        logging.info(
            "\r[Data: %s/%s] [Model Type: Identifier%s_Regressor%s] [Test Epoch: %d] [Batch: %d/%d] \
            [Error: rmse %f, abs %f, accuracy %f] [Test Time: %f]"
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, opt.regressor_type, epoch, i, len(dataloader),
            rmse_avg, abs_avg, accuracy_avg, time_avg)
        )

    # latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, epoch, opt.dataset_env)
    # CDF ploting of residual error
    res_em = np.abs(err_real_arr - err_fake_arr)
    data_train, data_test = data_raw
    err_svm, err_gt, _, _ = svm_regressor(data_train, data_test)
    res_svm = np.abs(err_svm - err_gt)
    accuracy, _, _ = svm_classifier(data_train, data_test)
    # print residual errors for CDF plotting
    # print("residual errors: ", res_em)
    # print(res_svm)
    # print(err_real_arr)
    CDF_plot(err_arr=err_real_arr, num=200, color='y')
    CDF_plot(err_arr=res_em, num=200, color='purple')
    CDF_plot(err_arr=res_svm, num=200, color='c')
    plt.legend(["Original error", "Our method", "SVM"], loc='lower right')
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, epoch)))
    plt.close()
    io.savemat(os.path.join(result_path, "residual_em_%s_%s_%d" % (opt.dataset_name, opt.dataset_env, epoch)),
               {'residual_em': res_em})
    io.savemat(os.path.join(result_path, "residual_svm_%s_%s_%d" % (opt.dataset_name, opt.dataset_env, epoch)),
               {'residual_em': res_svm})
    io.savemat(os.path.join(result_path, "original_%s_%s_%d" % (opt.dataset_name, opt.dataset_env, epoch)),
               {'residual_em': err_real_arr})


def test_gem_sepE(opt, device, tensor, result_path, model_path, dataloader, enet, epoch, data_raw): 
    # different for val and test: result_path, epoch

    # Save experimental results (though also in train dir, different log file from 'train_log.log' nevertheless)
    logging.basicConfig(filename=os.path.join(result_path, 'val_logE.log'), level=logging.INFO)
    logging.info("Started")

    # Load models from path
    if epoch != 0:
        enet.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % opt.epoch)))
        enet.eval()
    else:
        print("No saved models in dirs.")

    # Evaluation initialization
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
            label_gt = label_gt.to(device=device, dtype=torch.int64)

        with torch.no_grad():

            # Generate estimations
            label_est = enet(cir_gt)
            # err_est = mnet(env_latent, cir_gt)
            
            # Evaluation metrics
            # rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            # abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500 # batch_size
            # rmse_avg = rmse_error / (i + 1)
            # abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # ## Illustrate figures
            # # latent env arrays
            # reduced_latents, labels = reduce_latents(env_latent, label_gt)
            # if i == 0:
            #     features_arr = reduced_latents
            #     labels_arr = labels
            # elif i != len(dataloader) - 1:
            #     features_arr = np.vstack((features_arr, reduced_latents))
            #     labels_arr = np.vstack((labels_arr, labels))
            # else:  # drop the last batch
            #     continue
            # # range residual error arrays
            # err_real = err_gt.cpu().numpy()
            # err_fake = err_est.cpu().numpy()
            # if i == 0:
            #     err_real_arr = err_real
            #     err_fake_arr = err_fake
            # else:
            #     err_real_arr = np.vstack((err_real_arr, err_real))
            #     err_fake_arr = np.vstack((err_fake_arr, err_fake))
        
        # Print log
        sys.stdout.write(
            "\r[Sep Data: %s/%s] [Model Type: Identifier%s] [Test Epoch: %d] [Batch: %d/%d] [accuracy %f] [Test Time: %f]"
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, epoch, i, len(dataloader),
            accuracy_avg, time_avg)
        )
        logging.info(
            "\r[Data: %s/%s] [Model Type: Identifier%s] [Test Epoch: %d] [Batch: %d/%d] [accuracy %f] [Test Time: %f]"
            % (opt.dataset_name, opt.dataset_env, opt.identifier_type, epoch, i, len(dataloader),
            accuracy_avg, time_avg)
        )

    # # latent space visualization
    # visualize_latents(features_arr, labels_arr, result_path, epoch, opt.dataset_env)
    # # CDF ploting of residual error
    # res_em = np.abs(err_real_arr - err_fake_arr)
    # data_train, data_test = data_raw
    # err_svm, err_gt, _, _ = svm_regressor(data_train, data_test)
    # res_svm = np.abs(err_svm - err_gt)
    # CDF_plot(err_arr=err_real_arr, num=200, color='y')
    # CDF_plot(err_arr=res_em, num=200, color='purple')
    # CDF_plot(err_arr=res_svm, num=200, color='c')
    # plt.legend(["Original error", "Our method", "SVM"], loc='lower right')
    # plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, epoch)))
    # plt.close()


def test_gem_sepEM(opt, device, tensor, result_path, model_path, dataloader, enet, mnet, epoch, data_raw): 
    # different for val and test: result_path, epoch

    # Save experimental results (though also in train dir, different log file from 'train_log.log' nevertheless)
    logging.basicConfig(filename=os.path.join(result_path, 'val_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load models from path
    if epoch != 0:
        mnet.load_state_dict(torch.load(os.path.join(model_path, "MNet_%d.pth" % opt.epoch)))
        mnet.eval()
        enet.load_state_dict(torch.load(os.path.join(model_path, "ENet_%d.pth" % opt.epoch)))
        enet.eval()
    else:
        print("No saved models in dirs.")

    # Evaluation initialization
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

        with torch.no_grad():

            # Generate estimations p(k|r) p(\Delta d|r, k)
            label_est, env_latent = enet(cir_gt)
            ##
            zeros = np.zeros((cir_gt.shape[0], 1)) # as in GAN to build labels
            ones = np.ones((cir_gt.shape[0], 1))
            err_est_0 = mnet(cir_gt, zeros)
            err_est_1 = mnet(cir_gt, ones)
            # calculate estimated SRI p(\Delta d|r)
            err_est = label_est[:, 0] * err_est_0 + label_est[:, 1] * err_est_1

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            ##
            log_likelihood = err_est
            time_test = (time.time() - start_time) / 500 # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            ## Illustrate figures
            # latent env arrays
            # reduced_latents, labels = reduce_latents(env_latent, label_gt)
            # if i == 0:
            #     features_arr = reduced_latents
            #     labels_arr = labels
            # elif i != len(dataloader) - 1:
            #     features_arr = np.vstack((features_arr, reduced_latents))
            #     labels_arr = np.vstack((labels_arr, labels))
            # else:  # drop the last batch
            #     continue
            # range residual error arrays
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))
        
        # Print log
        sys.stdout.write(
            "\r[Sep Data: %s/%s] [Model Type: Regressor%s] [Test Epoch: %d] [Batch: %d/%d] [Error: rmse %f, abs %f, likelihood %f, accuracy %f] [Test Time: %f]"
            % (opt.dataset_name, opt.dataset_env, opt.regressor_type, epoch, i, len(dataloader),
            rmse_avg, abs_avg, log_likelihood, accuracy_avg, time_avg)
        )
        logging.info(
            "\r[Sep Data: %s/%s] [Model Type: Regressor%s] [Test Epoch: %d] [Batch: %d/%d] [Error: rmse %f, abs %f, likelihood %f, accuracy %f] [Test Time: %f]"
            % (opt.dataset_name, opt.dataset_env, opt.regressor_type, epoch, i, len(dataloader),
            rmse_avg, abs_avg, log_likelihood, accuracy_avg, time_avg)
        )

    # latent space visualization
    # visualize_latents(features_arr, labels_arr, result_path, epoch, opt.dataset_env)
    # CDF ploting of residual error
    res_em = np.abs(err_real_arr - err_fake_arr)
    data_train, data_test = data_raw
    err_svm, err_gt, _, _ = svm_regressor(data_train, data_test)
    res_svm = np.abs(err_svm - err_gt)
    CDF_plot(err_arr=err_real_arr, num=200, color='y')
    CDF_plot(err_arr=res_em, num=200, color='purple')
    CDF_plot(err_arr=res_svm, num=200, color='c')
    plt.legend(["Original error", "Our method", "SVM"], loc='lower right')
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, epoch)))
    plt.close()