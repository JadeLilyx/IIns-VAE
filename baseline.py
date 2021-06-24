import numpy as np
import time
import os
import scipy.import
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

from data_tools import *
from dataset import *


# ------------- SVM for comparison ---------------

def svm_regressor(data_train, data_test):

    # assign data for svm baseline
    cir_train, err_train, label_train = data_train
    cir_test, err_test, label_test = data_test

    # extract feature
    train_time = time.time()
    features_train = feature_extraction(cir_train)

    # error regression
    clf_reg = make_pipeline(StandardScaler(), SVR(gamma='auto'))
    clf_reg.fit(features_train, err_train)
    svr_train_time = time.time() - train_time

    test_time = time.time()
    features_test = feature_extraction(cir_test)
    err_est = clf_reg.predict(features_test)
    svr_test_time = time.time() - test_time

    # reshape
    err_test = err_test.reshape(err_test.shape[0])
    rmse_error = (np.sum((err_est - err_test) ** 2) / err_test.shape[0]) ** 0.5
    abs_error = (np.sum(np.abs(err_est - err_test)) / err_test.shape[0])
    print("SVM Regression Results: rmse %f, abs %f, time %f/%f" % (rmse_error, abs_error, svr_train_time, svr_test_time))

    return np.abs(err_est - err_test), np.abs(err_test), svr_test_time


# use the same cir and label as network
def svm_classifier(data_train, data_test):

    # assign data for svm baseline
    cir_train, err_train, label_train = data_train
    cir_test, err_test, label_test = data_test

    # extract feature
    train_time = time.time()
    feature_train = feature_extraction(cir_train)

    # label classification
    clf_cls = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_cls.fit(features_train, label_train.squeeze())
    svc_train_time = time.time() - train_time

    test_time = time.time()
    features_test = feature_extraction(cir_test)
    label_est = clf_cls.predict(features_test)
    svc_test_time = time.time() - test_time

    # reshape
    label_test = label_test.reshape(label_test.shape[0])
    accuracy = np.sum(label_est == label_test) / label_test.shape[0]
    print("SVM Classification Result: accuracy %f, time %f/%f" % accuracy, svc_train_time, svc_test_time)

    return accuracy, svc_test_time


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ewine", help="dataset for usage, ewine or zenodo")
    parser.add_argument("--dataset_env", type=str, default="nlos", help="dataset (zenodo) of different environments")
    parser.add_argument("--mode", type=str, default="full", help="mode to assign train and test data")
    parser.add_argument("--split_factor", type=float, default=0.8, help="split factor for train and test data")
    opt = parser.parse_args()

    # assign different roots for each dataset
    if opt.dataset_name == 'zenodo':
        root = 'data/data_zenodo/dataset.pkl'
    elif opt.dataset_name == 'ewine':
        folderpaths = ['./data/data_ewine/dataset1/',  # tag_room0.csv, tag_room1.csv
                       './data/data_ewine/dataset2/',  # tag_room0.csv
                       './data/data_ewine/dataset2/tag_room1/']
        root = folderpaths

    # assign data for training and testing
    data_train, data_test, feature_train, feature_test = err_mitigation_dataset(
        root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env, split_factor=opt.split_factor,
        scaling=True, mode=opt.mode, feature_flag=False
    )

    # error regression
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test)
    print("error_abs: ", error_svm[0:10])
    print("error_gt: ", error_gt[0:10])
    save_path = "saved_results/data_%s_%s_mode_%s/SVR" % (opt.dataset_name, opt.dataset_env, opt.mode)
    os.makedirs(save_path, exist_ok=True)
    CDF_plot(err_gt, num=200, color='y')
    CDF_plot(res_svm, num=200, color='c')
    plt.legend(['Original error', 'SVM'], loc='lower right')
    plt.savefig(os.path.join(save_path, "CDF_svm.png"))
    plt.close()
    
    # env classification
    accuracy, svc_test_time = svm_classifier(data_train, data_test)

