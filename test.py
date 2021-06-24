import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets

from utils import *
from dataset import *
from models import *
from baseline import svm_regressor

import setproctitle
import logging


def test_vl(opt, network, device, result_path, model_path, dataloader, epoch, data):
    # different for val and test: result_path, epoch
    network.eval()
    
    # Save experimental results (though also in train dir)
    logging.basicConfig(filename=os.path.join(result_path, 'val_log.log'), level=logging.INFO)
    logging.info("Started")

    # Load models from path
    if epoch = 0:
        Network.load_state_dict(torch.load(os.path.join(model_path, "Network_%d.pth" % epoch)))
        Network.eval()
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

            # Get estimation results
            label_est, err_est, env_latent = network(cir_gt)

            # Evaluation metrics
            rmse_error += (torch.mean((err_est - err_gt) ** 2)) ** 0.5
            abs_error += torch.mean(torch.abs(err_est - err_gt))
            time_test = (time.time() - start_time) / 500 # batch_size
            rmse_avg = rmse_error / (i + 1)
            abs_avg = abs_error / (i + 1)
            time_avg = time_test / (i + 1)
            label_gt = label_gt.squeeze()
            prediction = torch.argmax(label_est, dim=1)
            accuracy += torch.sum(prediction == label_gt).float()
            accuracy_avg = accuracy / (i + 1)

            # Illustration figures
            # latent env arrays
            reduced_latents, labels = reduce_latents(env_latent, label_gt)
            if i == 0:
                latents_arr = reduce_latents
                labels_arr = labels
            else:
                latents_arr = np.vstack((latents_arr, reduce_latents))
                labels_arr = np.vstack((labels_arr, labels))
            # range residual error arrays
            err_real = err_gt.cpu().numpy()
            err_fake = err_est.cpu().numpy()
            if i == 0:
                err_real_arr = err_real
                err_fake_arr = err_fake
            else:
                err_real_arr = np.vstack((err_real_arr, err_real))
                err_fake_arr = np.vstack((err_fake_arr, err_fake))
        
    # Print log (avg anyway here)
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

    # Latent code visualization
    visualize_latents(latents_arr, labels_arr, opt.dataset_env)
    plt.savefig(os.path.join(save_path, "latent_env_epoch%d.png" % epoch))
    plt.close()

    # CDF plotting of range error
    res_vl = np.abs(err_real_arr - err_fake_arr)
    data_train, data_test = data
    res_svm, err_gt, _ = svm_regressor(data_train, data_test)
    CDF_plot(err_arr=err_real_arr, num=200, color='y')
    CDF_plot(err_arr=res_em, num=200, color='purple')
    CDF_plot(err_arr=res_svm, num=200, color='c')
    plt.legend(["Original error", "Our method", "SVM"], loc='lower right')
    plt.savefig(os.path.join(result_path, "CDF_epoch%d.png" % epoch))
    plt.close()