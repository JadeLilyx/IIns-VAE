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


def test_vl_naive(opt, device, result_path, model_path, dataloader, enc, dec, idy, reg, test_epoch, data_raw, test_flag=False):
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'val_AE%dIdy%dReg%d' % (opt.conv_type, opt.identifier_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % test_epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % test_epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % test_epoch)))
        reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % test_epoch)))
        enc.eval()
        dec.eval()
        idy.eval()
        reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Data: %s/%s/%s] [Model Type: AE%d/Idy%d/Reg%d] [Test Epoch: %d] [Batch: %d/%d] "
                "[RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type, 
                test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )
            logging.info(
                "[Epoch: %d] [Batch: %d/%d] [RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )

    # 1) Latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    data_train, data_test = data_raw
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    accuracy, svc_test_time = svm_classifier(data_train, data_test)
    res_svm = np.asarray(res_svm)
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
    CDF_plot(err_arr=res_vl, num=50, color='c', marker='*')
    plt.legend(["Unmitigated Error", "SVM", "GEM (Our Method)"])
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.pdf" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.close()


def test_vl_semi(opt, device, result_path, model_path, dataloader, enc, dec, idy, reg, test_epoch, data_raw, test_flag=False):
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'val_AE%dIdy%dReg%d' % (opt.conv_type, opt.identifier_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "EncE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, test_epoch))))
        dec.load_state_dict(torch.load(os.path.join(model_path, "DecE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, test_epoch))))
        idy.load_state_dict(torch.load(os.path.join(model_path, "IdyE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, test_epoch))))
        reg.load_state_dict(torch.load(os.path.join(model_path, "RegE%.2fR%.2f_%d.pth" % (opt.sup_rate_e, opt.sup_rate_r, test_epoch))))
        enc.eval()
        dec.eval()
        idy.eval()
        reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Data: %s/%s/%s] [Model Type: AE%d/Idy%d/Reg%d Semi: R%.2fE%.2f] [Test Epoch: %d] [Batch: %d/%d] "
                "[RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type,
                opt.sup_rate_r, opt.sup_rate_e,
                test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )
            logging.info(
                "[Epoch: %d] [Batch: %d/%d] [RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )

    # 1) Latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    data_train, data_test = data_raw
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    accuracy, svc_test_time = svm_classifier(data_train, data_test)
    res_svm = np.asarray(res_svm)
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
    CDF_plot(err_arr=res_vl, num=50, color='c', marker='*')
    plt.legend(["Unmitigated Error", "SVM", "GEM (Our Method)"])
    plt.savefig(os.path.join(result_path, "CDFE%.2fR%.2f_%d.png" % (opt.sup_rate_e, opt.sup_rate_r, test_epoch)))
    plt.savefig(os.path.join(result_path, "CDFE%.2fR%.2f_%d.pdf" % (opt.sup_rate_e, opt.sup_rate_r, test_epoch)))
    plt.close()


def test_vl_multisemi(opt, device, result_path, model_path, dataloader, enc, dec, idy, reg, test_epoch, data_raw, sup_rate_e, sup_rate_r, test_flag=False):
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'testmulti_AE%dIdy%dReg%d' % (opt.conv_type, opt.identifier_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "EncE%.2fR%.2f_%d.pth" % (sup_rate_e, sup_rate_r, test_epoch))))
        dec.load_state_dict(torch.load(os.path.join(model_path, "DecE%.2fR%.2f_%d.pth" % (sup_rate_e, sup_rate_r, test_epoch))))
        idy.load_state_dict(torch.load(os.path.join(model_path, "IdyE%.2fR%.2f_%d.pth" % (sup_rate_e, sup_rate_r, test_epoch))))
        reg.load_state_dict(torch.load(os.path.join(model_path, "RegE%.2fR%.2f_%d.pth" % (sup_rate_e, sup_rate_r, test_epoch))))
        enc.eval()
        dec.eval()
        idy.eval()
        reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            # reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            # if i == 0:
            #     features_arr = reduced_latents
            #     labels_arr = labels
            # else:
            #     features_arr = np.vstack((features_arr, reduced_latents))
            #     labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Data: %s/%s/%s] [Model Type: AE%d/Idy%d/Reg%d Semi: R%.2fE%.2f] [Test Epoch: %d] [Batch: %d/%d] "
                "[RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type,
                sup_rate_r, sup_rate_e,
                test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )
            logging.info(
                "[Semi: R%.2fE%.2f Epoch: %d] [Batch: %d/%d] [RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (sup_rate_r, sup_rate_e, test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )

    # 1) Latent space visualization
    # visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_vl, num=50, marker='*')


def test_vl_DeIdy(opt, device, result_path, model_path, dataloader, enc, dec, reg, test_epoch, data_raw, test_flag=False):
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'valAb2_AE%dReg%d' % (opt.conv_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % test_epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % test_epoch)))
        # idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % test_epoch)))
        reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % test_epoch)))
        enc.eval()
        dec.eval()
        # idy.eval()
        reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            # label_est = idy(env_code_rv)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            # label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            # prediction = torch.argmax(label_est, dim=1)
            # accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            # accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/IdyNone/Reg%d] [Test Epoch: %d] [Batch: %d/%d] "
                "[RMSE: %f, MAE: %f, Test Time: %f]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.regressor_type, 
                test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, time_avg)
            )
            logging.info(
                "[Epoch: %d] [Batch: %d/%d] [RMSE: %f, MAE: %f, Test Time: %f]"
                % (test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, time_avg)
            )

    # 1) Latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    data_train, data_test = data_raw
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    accuracy, svc_test_time = svm_classifier(data_train, data_test)
    res_svm = np.asarray(res_svm)
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
    CDF_plot(err_arr=res_vl, num=50, color='c', marker='*')
    plt.legend(["Unmitigated Error", "SVM", "GEM (Our Method)"])
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.pdf" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.close()


def test_vl_DeIdy_variant(opt, device, result_path, model_path, model_path_ref, dataloader, enc, dec, idy_ref, reg, test_epoch, data_raw, test_flag=False):
    # advanced test: introduce another identifier (separatedly trained) to test accuracy
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'valAb2_AE%dReg%d' % (opt.conv_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % test_epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % test_epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path_ref, "Idy_500.pth")))
        reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % test_epoch)))
        enc.eval()
        dec.eval()
        idy.eval()
        reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/IdyNone/Reg%d] [Test Epoch: %d] [Batch: %d/%d] "
                "[RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.regressor_type, 
                test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )
            logging.info(
                "[Epoch: %d] [Batch: %d/%d] [RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )

    # 1) Latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    data_train, data_test = data_raw
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    accuracy, svc_test_time = svm_classifier(data_train, data_test)
    res_svm = np.asarray(res_svm)
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
    CDF_plot(err_arr=res_vl, num=50, color='c', marker='*')
    plt.legend(["Unmitigated Error", "SVM", "GEM (Our Method)"])
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.pdf" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.close()



def test_vl_DeReg(opt, device, result_path, model_path, dataloader, enc, dec, reg, test_epoch, data_raw, test_flag=False):
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'valAb3_AE%dIdy%d' % (opt.conv_type, opt.identifier_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % test_epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % test_epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % test_epoch)))
        # reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % test_epoch)))
        enc.eval()
        dec.eval()
        idy.eval()
        # reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            # err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Evaluation metrics
            # rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            # abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            # rmse_avg = rmse_error / (i + 1)
            # abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/Idy%d/RegNone] [Test Epoch: %d] [Batch: %d/%d] "
                "[ACC: %f, Test Time: %f]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, 
                test_epoch, i, len(dataloader), accuracy_avg, time_avg)
            )
            logging.info(
                "[Epoch: %d] [Batch: %d/%d] [ACC: %f, Test Time: %f]"
                % (test_epoch, i, len(dataloader), accuracy_avg, time_avg)
            )

    # 1) Latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    data_train, data_test = data_raw
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    accuracy, svc_test_time = svm_classifier(data_train, data_test)
    res_svm = np.asarray(res_svm)
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
    CDF_plot(err_arr=res_vl, num=50, color='c', marker='*')
    plt.legend(["Unmitigated Error", "SVM", "GEM (Our Method)"])
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.pdf" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.close()


def test_vl_DeReg_variant(opt, device, result_path, model_path, model_path_ref, dataloader, enc, dec, idy, reg_ref, test_epoch, data_raw, test_flag=False):
    # advanced test: introduce another regressor (separatedly trained) to test accuracy
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'valAb3_AE%dReg%d' % (opt.conv_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % test_epoch)))
        dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % test_epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % test_epoch)))
        reg.load_state_dict(torch.load(os.path.join(model_path_ref, "Reg_500.pth")))
        enc.eval()
        dec.eval()
        idy.eval()
        reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/Idy%d/RegNone] [Test Epoch: %d] [Batch: %d/%d] "
                "[RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, 
                test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )
            logging.info(
                "[Epoch: %d] [Batch: %d/%d] [RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (test_epoch, i, len(dataloader),
                rmse_avg, abs_avg, accuracy_avg, time_avg)
            )

    # 1) Latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    data_train, data_test = data_raw
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    accuracy, svc_test_time = svm_classifier(data_train, data_test)
    res_svm = np.asarray(res_svm)
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
    CDF_plot(err_arr=res_vl, num=50, color='c', marker='*')
    plt.legend(["Unmitigated Error", "SVM", "GEM (Our Method)"])
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.pdf" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.close()


def test_vl_DeDecoder(opt, device, result_path, model_path, dataloader, enc, idy, reg, test_epoch, data_raw, test_flag=False):
    # different for val and test: result_path, test_epoch

    # Save experimental results
    logging.basicConfig(filename=os.path.join(result_path, 'valAb4_AE%dIdy%dReg%d' % (opt.conv_type, opt.identifier_type, opt.regressor_type)), level=logging.INFO)
    # logging.info("Started")

    # Load models from path
    if test_epoch != 0:
        enc.load_state_dict(torch.load(os.path.join(model_path, "Enc_%d.pth" % test_epoch)))
        # dec.load_state_dict(torch.load(os.path.join(model_path, "Dec_%d.pth" % test_epoch)))
        idy.load_state_dict(torch.load(os.path.join(model_path, "Idy_%d.pth" % test_epoch)))
        reg.load_state_dict(torch.load(os.path.join(model_path, "Reg_%d.pth" % test_epoch)))
        enc.eval()
        # dec.eval()
        idy.eval()
        reg.eval()
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

            # Generate latent representations
            range_code, env_code, env_code_rv, _ = enc(cir_gt)
            # 1) reconstruct cir
            # cir_gen = dec(range_code, env_code)
            # 2) estimated range error
            err_est = reg(range_code)
            # 3) estimated env label
            label_est = idy(env_code_rv)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500   # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gtl = label_gt.squeeze()  # no squeeze affect accuracy but squeeze will affect latent visualization
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gtl).float() / label_gt.shape[0]
            accuracy_avg = accuracy / (i + 1)

            # Illustrate figures
            # 1) latent env arrays (numpy, reduce, append)
            reduced_latents, labels = reduce_latents(env_code_rv, label_gt)  # not suqeezed label_gtl
            if i == 0:
                features_arr = reduced_latents
                labels_arr = labels
            else:
                features_arr = np.vstack((features_arr, reduced_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # 2) range residual error arrays (numpy, append)
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))

        # Print log
        if test_flag:
            sys.stdout.write(
                "\r[Ablation: %d, Data: %s/%s/%s] [Model Type: AE%d/Idy%d/Reg%d] [Test Epoch: %d] [Batch: %d/%d] "
                "[RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (opt.ablation_type, opt.dataset_name, opt.dataset_env, opt.mode, opt.conv_type, opt.identifier_type, opt.regressor_type, 
                test_epoch, i, len(dataloader), rmse_avg, abs_avg, accuracy_avg, time_avg)
            )
            logging.info(
                "[Epoch: %d] [Batch: %d/%d] [RMSE: %f, MAE: %f, ACC: %f, Test Time: %f]"
                % (test_epoch, i, len(dataloader), rmse_avg, abs_avg, accuracy_avg, time_avg)
            )

    # 1) Latent space visualization
    visualize_latents(features_arr, labels_arr, result_path, test_epoch, opt.dataset_env)
    
    # 2) CDF plotting of residual error
    res_ori = np.asarray([item[0] for item in err_real_arr])
    res_vl_arr = np.abs(err_real_arr - err_fake_arr)
    res_vl = np.asarray([item[0] for item in res_vl_arr])
    
    data_train, data_test = data_raw
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    accuracy, svc_test_time = svm_classifier(data_train, data_test)
    res_svm = np.asarray(res_svm)
    
    CDF_plot(err_arr=res_ori, num=50, color='y', marker='x')
    CDF_plot(err_arr=res_svm, num=50, color='purple', marker='o')
    CDF_plot(err_arr=res_vl, num=50, color='c', marker='*')
    plt.legend(["Unmitigated Error", "SVM", "GEM (Our Method)"])
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.png" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.savefig(os.path.join(result_path, "CDF_%s_%s_%d.pdf" % (opt.dataset_name, opt.dataset_env, test_epoch)))
    plt.close()