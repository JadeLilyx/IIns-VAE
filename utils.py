import argparse
import numpy as np
import math
import os
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision.utils import save_image, make_grid

# import pytorch_lightning as pl
import umap
from baselines import svm_regressor  # for CDF comparison


def get_args(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=501, help="number of epochs of training")
    parser.add_argument("--test_epoch", type=int, default=500, help="epoch to test model performance")

    parser.add_argument("--dataset_name", type=str, default="zenodo", help="name of the dataset, ewine or zenodo")
    parser.add_argument("--dataset_env", type=str, default='room_full',
                        help="dataset (zenodo) of different environments, including rooms and obstacles")
    parser.add_argument("--mode", type=str, default="full", help="simulated mode train/test for data usage, paper or full")
    parser.add_argument("--split_factor", type=float, default=0.8, help="ratio to split data for training and testing")

    parser.add_argument("--batch_size", type=int, default=500, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

    parser.add_argument("--n_downsample", type=int, default=4, help="number downsampling layers in encoder")
    parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder/decoder")
    parser.add_argument("--dim", type=int, default=4, help="number of filters in first encoder layer")
    parser.add_argument("--env_dim", type=int, default=10, help="dimensionality of the env code")
    
    parser.add_argument("--range_dim", type=int, default=10, help="dimensionality of the range code")
    parser.add_argument("--conv_type", type=int, default=2, help="use 1 dim or 2 dim for convolution")
    # parser.add_argument("--if_expand", type=bool, default=False, help="Expand the cir signal to square or not")
    parser.add_argument("--restorer_type", type=str, default='Linear', help="structure for restorer net: linear, conv1d, or conv2d")
    parser.add_argument("--classifier_type", type=str, default='Conv2d', help="structure for classifier to estimate label")
    parser.add_argument("--use_soft", type=bool, default=False, help="estimate soft range estimation or hard one")

    parser.add_argument("--sample_interval", type=int, default=20, help="epoch interval saving generator samples")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="epoch interval between saving model checkpoint")

    return parser


# -------------------------------------------
#       Visualization Figures
# -------------------------------------------


# use encoder, decoder (cir)
def visualize_recon(save_path, epoch, dataloader, encoder, decoder):
    """Save a generated sample from the validation or testing set"""
    cir_samples = next(iter(dataloader))
    # encoder.eval()
    # decoder.eval()
    # restorer.eval()
    # for i, samples in enumerate(dataloader):
    cir_gt = cir_samples["CIR"]  # (B, 152/156)
    if torch.cuda.is_available():
        cir_gt = cir_gt.cuda()

    # Get latent variables for reconstruction
    range_code, env_code, _, _ = encoder(cir_gt)
    cir_gen = decoder(range_code, env_code)
        
        # if i == 0:
        #     cir_gen_arr = cir_gen
        #     cir_gt_arr = cir_gt
        # else:
        #     cir_gen_arr = np.vstack((cir_gen_arr, cir_gen))
        #     cir_gt_arr = np.vstack((cir_gt_arr, cir_gt))

    # visualize results
    # plt.title("Reconstruction of CIR waveforms (epoch %d)." % (epoch))
    # plt.xlabel("Time Interval")
    # plt.ylabel("CIR")
    plt.plot(cir_gt[0].cpu().detach().numpy(), color='blue', linestyle='--')
    plt.plot(cir_gen[0].cpu().detach().numpy(), color='red')
    plt.legend(["Real Waveform", "Reconstructed Waveform"], loc='upper right')
    plt.savefig(os.path.join(save_path, "recon_epoch%d.png" % epoch))
    # plt.savefig("saved_results/AEType%d_ResType%s_%s_dim%d/%d.png" % (conv_type, restorer_type, dataset_name, range_dim, epoch))
    # dataset_name, conv_type, restorer_type, range_dim
    plt.close()


# use encoder, classifier (cir, label), full dataset
def visualize_latent(save_path, epoch, dataloader, dataset_env, encoder, classifier, title=None):
    """Visualize latent space disentanglement"""
    # samples = next(iter(dataloader))
    for i, samples in enumerate(dataloader):
        cirs = samples["CIR"]
        labels = samples["Label"]
        if torch.cuda.is_available():
            cirs = cirs.cuda()
            labels = labels.cuda()
        
        # range_code, env_code, env_code_rv, kl_div = encoder(cirs)
        latents = encoder(cirs)[1].view(encoder(cirs)[1].size(0), -1)
        # try latter
        latents_rv = encoder(cirs)[2].view(encoder(cirs)[2].size(0), -1)
        
        # use all 8 dims
        latents = latents.cpu().numpy()
        labels = labels.cpu().numpy()
        # print(latents.shape)  # (b, 8)
        if latents.shape[1] > 2:
            reduced_latents = umap.UMAP().fit_transform(latents)
        else:
            reduced_latents = latents
        if i == 0:
            features_arr = reduced_latents
            labels_arr = labels
            # print(i, features_arr.shape)
        else:
            features_arr = np.vstack((features_arr, reduced_latents))
            labels_arr = np.vstack((labels_arr,labels))
            # print(i, features_arr.shape)
        
        # ----------- use 4 dim rv -------------
        latents_rv = latents_rv.cpu().numpy()
        labels_rv = labels
        # print(latents.shape)  # (b, 8)
        if latents_rv.shape[1] > 2:
            reduced_latents_rv = umap.UMAP().fit_transform(latents_rv)
        else:
            reduced_latents_rv = latents_rv
        if i == 0:
            features_arr_rv = reduced_latents_rv
            labels_arr_rv = labels_rv
            # print(i, features_arr.shape)
        else:
            features_arr_rv = np.vstack((features_arr_rv, reduced_latents_rv))
            labels_arr_rv = np.vstack((labels_arr_rv, labels_rv))
            # print(i, features_arr.shape)
    
    # print(features_arr.shape)
    data_module = dict()
    labels_list = labels_arr.tolist()
    reduced_latents_list = features_arr.tolist()
    classes = set()
    for i in range(len(labels_list)):
        label_item = labels_list[i][0]
        if label_item not in data_module:
            data_module[label_item] = list()
            classes.add(label_item)
        data_module[label_item].append(reduced_latents_list[i])

    # ----------- use 4 dim rv -------------
    data_module_rv = dict()
    labels_list_rv = labels_arr_rv.tolist()
    reduced_latents_list_rv = features_arr_rv.tolist()
    classes_rv = set()
    for i in range(len(labels_list_rv)):
        label_item_rv = labels_list_rv[i][0]
        if label_item_rv not in data_module_rv:
            data_module_rv[label_item_rv] = list()
            classes.add(label_item_rv)
        data_module_rv[label_item_rv].append(reduced_latents_list_rv[i])

    # classes = np.unique(labels)
    # print("classes:", classes)
    if dataset_env == 'room_full':
        label_str_dict = {
            0: 'cross-room', 1: 'big room', 2: 'medium room', 3: 'small room', 4: 'outdoor'
        }
    elif dataset_env == 'obstacle_full':
        label_str_dict = {
            1: 'metal window', 2: 'glass plate', 3: 'wood door', 4: 'metal plate', 5: 'LCD TV',
            6: 'cardboard box', 7: 'plywood plate', 8: 'plastic', 9: 'polystyrene plate', 10: 'wall'
        }
    elif dataset_env == 'room_part':
        label_str_dict = {
            1: 'big room', 2: 'medium room', 3: 'small room'
        }
    elif dataset_env == 'room_full_rough':
        label_str_dict = {
            1: 'cross-room', 2: 'outdoor', 3: 'rooms'
        }
    elif dataset_env == 'obstacle_part':
        label_str_dict = {
            1: 'metal (plate/window)', 2: 'wood', 3: 'plastic', 4: 'glass'
        }
    elif dataset_env == 'obstacle_part2':
        label_str_dict = {
            1: 'heavy', 2: 'light'
        }
    elif dataset_env == 'room_full_rough2':
        label_str_dict = {
            1: 'indoor', 2: 'outdoor'
        }
    elif dataset_env == 'paper':
        label_str_dict = {
            1: 'cross-room', 2: 'big room', 3: 'medium room', 4: 'small room'
        }
    for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
        class_features = np.asarray(data_module[cls])
        # if dataset_env == 'room_full':
        #     label_str_dict = {
        #         0: 'cross-room', 1: 'big room', 2: 'medium room', 3: 'small room', 4: 'outdoor'
        #     }
        # else:
        #     label_str_dict = {
        #         1: 'metal window', 2: 'glass plate', 3: 'wood door', 4: 'metal plate', 5: 'LCD TV',
        #         6: 'cardboard box', 7: 'plywood plate', 8: 'plastic', 9: 'polystyrene plate', 10: 'wall'
        #     }
        plt.scatter(class_features[:, 0], class_features[:, 1], c=[color], label=label_str_dict[cls], s=[10], alpha=0.5)
        # print(cls)
    if title is not None:
        plt.set_title(title)
    plt.legend([label_str_dict[item] for item in label_str_dict], loc='upper right')
    plt.savefig(os.path.join(save_path, "latent_env_rv_epoch%d.png" % epoch))
    plt.close()

    # ----------- use 4 dim rv -------------
    for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
        class_features_rv = np.asarray(data_module_rv[cls])
        plt.scatter(class_features_rv[:, 0], class_features_rv[:, 1], c=[color], label=label_str_dict[cls], s=[2], alpha=0.5)
        # print(cls)
    if title is not None:
        plt.set_title(title)
    plt.legend([label_str_dict[item] for item in label_str_dict], loc='upper right')
    plt.savefig(os.path.join(save_path, "latent_env_epoch%d.png" % epoch))
    plt.close()


# use encoder, restorer (cir, err), use full dataset
def CDF_plot(opt, root, save_path, epoch, dataloader, encoder, restorer, num=20, title=None, use_competitor=True):
    # cir_samples = next(iter(dataloader))
    encoder.eval()
    for i, samples in enumerate(dataloader):
        cir_gt = samples["CIR"]
        err_gt = samples["Err"]
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()
    
        # Get latent variables for estimation
        range_code, env_code, _, _ = encoder(cir_gt)
        err_fake = restorer(range_code)
        
        err_gt = err_gt.cpu().numpy()
        err_fake = err_fake.cpu().numpy()
        if i == 0:
            err_gt_arr = err_gt
            err_fake_arr = err_fake
        else:
            err_gt_arr = np.vstack((err_gt_arr, err_gt))
            err_fake_arr = np.vstack((err_fake_arr, err_fake))
    
    # ------------------------------------
    #           Learning Method
    # ------------------------------------
    data = np.abs(err_fake_arr - err_gt_arr).squeeze()
    print(data)
    CDF_plot_old(save_path, data)
    data = np.abs(data)
    blocks_num = num
    pred_error_max = np.max(data)
    pred_error_cnt = np.zeros((blocks_num + 1,))
    step = pred_error_max / blocks_num

    # normalize to (0, 1) by dividing max
    for i in range(data.shape[0]):
        index = int(data[i] / step)
        pred_error_cnt[index] = pred_error_cnt[index] + 1
    pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)

    # accumulate error at each point to CDF
    CDF = np.zeros((blocks_num + 1,))
    for i in range(blocks_num + 1):
        if i == 0:
            CDF[i] = pred_error_cnt[i]
        else:
            CDF[i] = CDF[i - 1] + pred_error_cnt[i]
        
    plt.plot(np.linspace(0, pred_error_max, num=blocks_num + 1), CDF)
    
    # -------------------------------
    #          SVM Method
    # -------------------------------

    if use_competitor:  # True for testing and False for validation
        error_svm, error_gt = svm_regressor(root=root, dataset_env=opt.dataset_env, mode=opt.mode, split_factor=opt.split_factor)
        print(error_svm)
        CDF_plot_old(save_path, error_svm)
        data_com = np.abs(error_svm)
        blocks_num_com = num
        pred_error_max_com = np.max(data_com)
        pred_error_cnt_com = np.zeros((blocks_num_com + 1,))
        step_com = pred_error_max_com / blocks_num_com

        # normalize to (0, 1) by dividing max
        for i in range(data_com.shape[0]):
            index_com = int(data_com[i] / step_com)
            pred_error_cnt_com[index_com] = pred_error_cnt_com[index_com] + 1
        pred_error_cnt_com = pred_error_cnt_com / np.sum(pred_error_cnt_com)

        # accumulate error at each point to CDF
        CDF_com = np.zeros((blocks_num_com + 1,))
        for i in range(blocks_num_com + 1):
            if i == 0:
                CDF_com[i] = pred_error_cnt_com[i]
            else:
                CDF_com[i] = CDF_com[i - 1] + pred_error_cnt_com[i]
        
        plt.plot(np.linspace(0, pred_error_max_com, num=blocks_num_com + 1), CDF_com, color='red')
        
        # original range error
        data_org = np.abs(error_gt)
        print(data_org)
        CDF_plot_old(save_path, data_org)
        blocks_num_org = num
        pred_error_max_org = np.max(data_org)
        pred_error_cnt_org = np.zeros((blocks_num_org + 1,))
        step_org = pred_error_max_org / blocks_num_org

        # normalize to (0, 1) by dividing max
        for i in range(data_org.shape[0]):
            index_org = int(data_org[i] / step_org)
            pred_error_cnt_org[index_org] = pred_error_cnt_org[index_org] + 1
        pred_error_cnt_org = pred_error_cnt_org / np.sum(pred_error_cnt_org)

        # accumulate error at each point to CDF
        CDF_org = np.zeros((blocks_num_org + 1,))
        for i in range(blocks_num_org + 1):
            if i == 0:
                CDF_org[i] = pred_error_cnt_org[i]
            else:
                CDF_org[i] = CDF_org[i - 1] + pred_error_cnt_org[i]
            
        plt.plot(np.linspace(0, pred_error_max_org, num=blocks_num_org + 1), CDF_org, color='black')

        plt.legend(["Our method", "SVM", "Original error"], loc='lower right')

    # (here can plot another line from svr)
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join(save_path, "CDF_epoch%d.png" % epoch))
    plt.close()


def CDF_plot_old(save_path, data, num=20, figure_title=""):
    # pred_abs_error = np.abs(predict_y - test_label)
    data = np.abs(data)
    blocks_num = num
    pred_error_max = np.max(data)
    pred_error_cnt = np.zeros((blocks_num + 1,))
    step = pred_error_max / blocks_num

    for i in range(data.shape[0]):
        index = int(data[i] / step)
        pred_error_cnt[index] = pred_error_cnt[index] + 1

    pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)
    CDF = np.zeros((blocks_num + 1,))

    for i in range(blocks_num+1):
        if i==0:
            CDF[i] = pred_error_cnt[i]
        else:
            CDF[i] = CDF[i-1] + pred_error_cnt[i]

    plt.plot(np.linspace(0, pred_error_max, num=blocks_num+1), CDF)
    plt.title(figure_title)

    plt.savefig(os.path.join(save_path, "new2.png"))
    plt.close()


def CDF_plot_test(opt, root, save_path, epoch, dataloader, encoder, restorer, num=20, title=None, use_competitor=True):
    encoder.eval()
    for i, samples in enumerate(dataloader):
        cir_gt = samples["CIR"]
        err_gt = samples["Err"]
        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()

        # Get latent variables for estimation
        range_code, _, _, _ = encoder(cir_gt)
        err_fake = restorer(range_code)

        err_gt = err_gt.cpu().numpy()
        err_fake = err_fake.cpu().numpy()
        if i == 0:
            err_gt_arr = err_gt
            err_fake_arr = err_fake
        else:
            err_gt_arr = np.vstack((err_gt_arr, err_gt))
            err_fake_arr = np.vstack((err_fake_arr, err_fake))

    # ---------------------------
    #     Learning Method
    # ---------------------------
    data = np.abs(err_fake_arr - err_gt_arr)
    blocks_num = num
    pred_error_max = np.max(data)
    pred_error_cnt = np.zeros((blocks_num + 1,))
    step = pred_error_max / blocks_num

    # normalize to (0, 1) by dividing max
    for i in range(data.shape[0]):
        index = int(data[i] / step)
        pred_error_cnt[index] = pred_error_cnt[index] + 1
    pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)

    # accumulate error at each point to CDF
    CDF = np.zeros((blocks_num +1,))
    for i in range(blocks_num + 1):
        if i == 0:
            CDF[i] = pred_error_cnt[i]
        else:
            CDF[i] = CDF[i - 1] + pred_error_cnt[i]

    plt.plot(np.linspace(0, pred_error_max, num=blocks_num + 1), CDF, color='purple', marker='*')
    # plt.vlines(0, 0, 0.75, colors = "purple", linestyles='solid')

    # -----------------------------
    #         SVM Method
    # -----------------------------

    if use_competitor:
        error_svm, error_gt = svm_regressor(root=root, dataset_env=opt.dataset_env, mode=opt.mode, split_factor=opt.split_factor)
        data_com = np.abs(error_svm)
        blocks_num_com = num
        pred_error_max_com = np.max(data_com)
        pred_error_cnt_com = np.zeros((blocks_num_com + 1,))
        step_com = pred_error_max_com / blocks_num_com

        # normalize to (0, 1) by dividing max
        for i in range(data_com.shape[0]):
            index_com = int(data_com[i] / step_com)
            pred_error_cnt_com[index_com] = pred_error_cnt_com[index_com] + 1
        pred_error_cnt_com = pred_error_cnt_com / np.sum(pred_error_cnt_com)
        
        # accumulate error at each point to CDF
        CDF_com = np.zeros((blocks_num_com + 1,))
        for i in range(blocks_num_com + 1):
            if i == 0:
                CDF_com[i] = pred_error_cnt_com[i]
            else:
                CDF_com[i] = CDF_com[i - 1] + pred_error_cnt_com[i]

        plt.plot(np.linspace(0, pred_error_max_com, num=blocks_num_com + 1), CDF_com, color='c', marker='x')

        # original range error
        data_org = np.abs(error_gt)
        blocks_num_org = num
        pred_error_max_org = np.max(data_org)
        pred_error_cnt_org = np.zeros((blocks_num_org + 1,))
        step_org = pred_error_max_org / blocks_num_org

        # normalize to (0, 1) by dividing max
        for i in range(data_org.shape[0]):
            index_org = int(data_org[i] / step_org)
            pred_error_cnt_org[index_org] = pred_error_cnt_org[index_org] + 1
        pred_error_cnt_org = pred_error_cnt_org / np.sum(pred_error_cnt_org)

        # accumulate error at each point to CDF
        CDF_org = np.zeros((blocks_num_org + 1,))
        for i in range(blocks_num_org + 1):
            if i == 0:
                CDF_org[i] = pred_error_cnt_org[i]
            else:
                CDF_org[i] = CDF_org[i - 1] + pred_error_cnt_org[i]
            
        plt.plot(np.linspace(0, pred_error_max_org, num=blocks_num_org + 1), CDF_org, color='y', marker='o')

    plt.legend(["DGM", "SVM", "Unmitigated"], loc='lower right')
    plt.xlim((0.0, 0.6))
    plt.ylim((0.0, 1.0))
    plt.xlabel('Range Error (m)')
    plt.ylabel('CDF')
    x_ticks = np.arange(0.0, 0.5, 0.1)
    y_ticks = np.arange(0.0, 1.0, 0.2)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # (here can plot another line from svr)
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join(save_path, "CDF_epoch%d.png" % epoch))
    plt.close()


def CDF_plot_semi_test(opt, root, save_path, epoch, err_gt_arr, err_fake_arr_1, err_fake_arr_2, err_fake_arr_4, err_fake_arr_6, err_fake_arr_8, err_fake_arr_10, use_competitor=True):

    # ---------------------------
    #     Learning Methods
    # ---------------------------
    
    CDF_plot_net(err_gt_arr, err_fake_arr_1, color='midnightblue') #, marker='o')
    CDF_plot_net(err_gt_arr, err_fake_arr_2, color='deepskyblue')  #, marker='o')
    CDF_plot_net(err_gt_arr, err_fake_arr_4, color='lightsteelblue')  #, marker='x')
    CDF_plot_net(err_gt_arr, err_fake_arr_6, color='lightgreen')  #, marker='*')
    CDF_plot_net(err_gt_arr, err_fake_arr_8, color='mediumseagreen')  #, marker='+')
    CDF_plot_net(err_gt_arr, err_fake_arr_10, color='darkgreen')  #, linestyle='-o')

    # plt.plot(np.linspace(0, pred_error_max, num=blocks_num + 1), CDF, color='purple', linestyle='-o')

    # -----------------------------
    #         SVM and GT
    # -----------------------------
    # require: root, opt, use_competitor
    # better use CDF_plot_net() also
    num = 200
    if use_competitor:
        error_svm, error_gt = svm_regressor(root=root, dataset_env=opt.dataset_env, mode=opt.mode, split_factor=opt.split_factor)
        data_com = np.abs(error_svm)
        blocks_num_com = num
        pred_error_max_com = np.max(data_com)
        pred_error_cnt_com = np.zeros((blocks_num_com + 1,))
        step_com = pred_error_max_com / blocks_num_com

        # normalize to (0, 1) by dividing max
        for i in range(data_com.shape[0]):
            index_com = int(data_com[i] / step_com)
            pred_error_cnt_com[index_com] = pred_error_cnt_com[index_com] + 1
        pred_error_cnt_com = pred_error_cnt_com / np.sum(pred_error_cnt_com)
        
        # accumulate error at each point to CDF
        CDF_com = np.zeros((blocks_num_com + 1,))
        for i in range(blocks_num_com + 1):
            if i == 0:
                CDF_com[i] = pred_error_cnt_com[i]
            else:
                CDF_com[i] = CDF_com[i - 1] + pred_error_cnt_com[i]

        plt.plot(np.linspace(0, pred_error_max_com, num=blocks_num_com + 1), CDF_com, color='rosybrown', marker='v')

        # original range error
        data_org = np.abs(error_gt)  # from svm not before
        blocks_num_org = num
        pred_error_max_org = np.max(data_org)
        pred_error_cnt_org = np.zeros((blocks_num_org + 1,))
        step_org = pred_error_max_org / blocks_num_org

        # normalize to (0, 1) by dividing max
        for i in range(data_org.shape[0]):
            index_org = int(data_org[i] / step_org)
            pred_error_cnt_org[index_org] = pred_error_cnt_org[index_org] + 1
        pred_error_cnt_org = pred_error_cnt_org / np.sum(pred_error_cnt_org)

        # accumulate error at each point to CDF
        CDF_org = np.zeros((blocks_num_org + 1,))
        for i in range(blocks_num_org + 1):
            if i == 0:
                CDF_org[i] = pred_error_cnt_org[i]
            else:
                CDF_org[i] = CDF_org[i - 1] + pred_error_cnt_org[i]
            
        plt.plot(np.linspace(0, pred_error_max_org, num=blocks_num_org + 1), CDF_org, color='orchid', marker='v')

    plt.legend(["DGM(0.1)", "DGM(0.2)", "DGM(0.4)", "DGM(0.6)", "DGM(0.8)", "DGM(1.0)", "SVM", "Unmitigated"], loc='lower right')
    plt.xlim((0.0, 0.75))
    plt.ylim((0.0, 1.0))
    plt.xlabel('Range Error (m)')
    plt.ylabel('CDF')
    x_ticks = np.arange(0.0, 0.6, 0.1)
    y_ticks = np.arange(0.0, 1.0, 0.2)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.savefig(os.path.join(save_path, "CDF_epoch%d.png" % epoch))
    plt.close()


def CDF_plot_net(err_gt_arr, err_fake_arr, num=200, color='purple'):  #, marker='-o'):
    data = np.abs(err_fake_arr - err_gt_arr)
    blocks_num = num
    pred_error_max = np.max(data)
    pred_error_cnt = np.zeros((blocks_num + 1,))
    step = pred_error_max / blocks_num

    # normalize to (0, 1) by dividing max
    for i in range(data.shape[0]):
        index = int(data[i] / step)
        pred_error_cnt[index] = pred_error_cnt[index] + 1
    pred_error_cnt = pred_error_cnt / np.sum(pred_error_cnt)

    # accumulate error at each point to CDF
    CDF = np.zeros((blocks_num + 1,))
    for i in range(blocks_num + 1):
        if i == 0:
            CDF[i] = pred_error_cnt[i]
        else:
            CDF[i] = CDF[i - 1] + pred_error_cnt[i]

    plt.plot(np.linspace(0, pred_error_max, num=blocks_num + 1), CDF, color=color)  #, marker=marker)


# backup
def visualize_latent_origin(save_path, epoch, dataloader, encoder, classifier, title=None):
    """Visualize latent space disentanglement"""
    # cir_samples = next(iter(dataloader))
    # encoder.eval()
    # cir_gt = cir_samples["CIR"]
    # label_gt = cir_samples["Label"]
    # if torch.cuda.is_available():
    #     cir_gt = cir_gt.cuda()
    #     label_gt = label_gt.cuda()

    # # Get latent variable for env
    # range_code, env_code, env_code_rv, _ = encoder(cir_gt)
    # label_fake = classifier(env_code)

    # latents = []
    # labels = []
    # for cir, label in zip(samples["CIR"], samples["Label"]):
    #     print(cir.shape)  # (157)
    #     _, latent, _, _ = encoder(cir)
    #     print(latent.shape)
    #     latents += [latent.view(latent.size(0), -1)]
    #     labels += [label]

    samples = next(iter(dataloader))
    cirs = samples["CIR"]
    labels = samples["Label"]
    if torch.cuda.is_available():
        cirs = cirs.cuda()
        labels = labels.cuda()
    
    # latents = [encoder(x)[1].view(encoder(x)[1].size(0), -1) for x in cir]  # use the full dataloader
    # latents_rv = [encoder(x)[2].view(encoder(x)[2].size(0), -1) for _, x, _ in dataloader]
    # labels = [label for label in label]
    latents = encoder(cirs)[1].view(encoder(cirs)[1].size(0), -1)

    latents = latents.cpu().numpy()
    labels = labels.cpu().numpy()
    print(latents.shape)  # (b, 8)
    if latents.shape[1] > 2:
        reduced_latents = umap.UMAP().fit_transform(latents)
    else:
        reduced_latents = latents

    # return reduced_latents and labels for disp
    data_module = np.load(file_path)
    reduced_latents = data_module['arr_0']
    labels = data_module['arr_1']
    classes = np.unique(labels)
    for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
        class_features = reduced_latents[labels == cls]
        plt.scatter(class_features[:, 0], class_features[:, 1], c=[color], label=cls, s=[2], alpha=0.5)
    if title is not None:
        plt.set_title(title)
    plt.savefig(os.path.join(save_path, "latent_env_epoch%d.png" % epoch))
    plt.close()


if __name__ == '__main__':
    y_gt = np.array([0, 1, 0, 2, 6, 4, 2, 1, 0])
    y_est = np.array([0, 5, 1, 2, 5, 6, 2, 0, 1])
    err = np.abs(y_gt - y_est)
    CDF_plot(err, 200)
