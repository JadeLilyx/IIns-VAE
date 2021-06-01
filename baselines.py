import numpy as np
import time
import os
import scipy.io
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

from data_tools import *
from dataset import *


def svm_regressor(root, dataset_env='room_full', mode='full', split_factor=0.8):
    
    # assign train and test data (feature_extraction)
    data_train, data_test, feature_train, feature_test = err_mitigation_dataset(
        root=root, dataset_env=dataset_env, split_factor=split_factor, scaling=True, mode=mode, feature_flag=True
    )

    # assign data for svm baseline
    _, err_train, label_train = data_train  # cir_data not used
    _, err_test, label_test = data_test
    
    # error regression
    train_time = time.time()
    clf_reg = make_pipeline(StandardScaler(), SVR(gamma='auto'))
    clf_reg.fit(feature_train, err_train)
    svr_train_time_r = time.time() - train_time

    test_time = time.time()
    err_est = clf_reg.predict(feature_test)
    svr_test_time_r = time.time() - test_time

    # reshape
    err_test = err_test.reshape(err_test.shape[0])
    unmitigated_error = np.sum(err_test) / err_test.shape[0]
    rmse_error = (np.sum((err_est - err_test) ** 2) / err_test.shape[0]) ** 0.5
    abs_error = (np.sum(np.abs(err_est - err_test)) / err_test.shape[0])
    print("SVM Regression Results: rmse %f, abs %f \n" % (rmse_error, abs_error))
    print("Unmitigated Error: abs %f" % unmitigated_error)

    return np.abs(err_est - err_test), np.abs(err_test)


def svm_classifier(root, dataset_env='room_full', mode='full', split_factor=0.8):
    
    # assign train and test data (feature_extraction)
    data_train, data_test, feature_train, feature_test = err_mitigation_dataset(
        root=root, dataset_env=dataset_env, split_factor=split_factor, scaling=True, mode=mode, feature_flag=True
    )

    # assign data for svm baseline
    _, err_train, label_train = data_train  # cir_data not used
    _, err_test, label_test = data_test
    # print(label_train)
    
    # error regression
    train_time = time.time()
    clf_cls = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_cls.fit(feature_train, label_train.squeeze())
    svr_train_time_c = time.time() - train_time

    test_time = time.time()
    label_est = clf_cls.predict(feature_test)
    svr_test_time_c = time.time() - test_time

    # reshape
    label_test = label_test.reshape(label_test.shape[0])
    accuracy = np.sum(label_est == label_test) / label_test.shape[0]
    # print("SVM Classification Result: accuracy %f" % accuracy)

    return accuracy, label_est, label_test


def CDF_plot_base(opt, root, save_path, num=200, title=None):
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
        
    plt.plot(np.linspace(0, pred_error_max_com, num=blocks_num_com + 1), CDF_com, color='red')
    
    # original error
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
        
    plt.plot(np.linspace(0, pred_error_max_org, num=blocks_num_org + 1), CDF_org, color='black')

    plt.legend(["SVM Mitigated Error", "Original Range Error"], loc='lower right')
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join(save_path, "CDF_svm_%s.png" % opt.dataset_env))
    plt.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_env", type=str, default="room_full", help="dataset (zenodo) of different environments for generality, room_full/half/quater")
    parser.add_argument("--mode", type=str, default="full", help="method to assign train and test data")
    parser.add_argument("--split_factor", type=float, default=0.8, help="split factor for train data")
    opt = parser.parse_args()

    # range error regression using svm
    root = './data/data_zenodo/dataset.pkl'
    error_svm = svm_regressor(root=root, dataset_env=opt.dataset_env, mode=opt.mode, split_factor=opt.split_factor)
    save_path = "saved_results/%s_%s/SVR_reg" % (opt.dataset_env, opt.mode)
    os.makedirs(save_path, exist_ok=True)
    CDF_plot_base(opt, root, save_path, 200)

    # env label classification using svm
    accuracy, label_est, label_test = svm_classifier(root=root, dataset_env=opt.dataset_env, mode=opt.mode, split_factor=opt.split_factor)
    print("SVM Classification Result: accuracy %f\n" % accuracy)
    print("label_est: ", label_est)
    print("label_test: ", label_test)


    