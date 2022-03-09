import numpy as np
import time
import os
import scipy.io
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, SVC

from utils import *


# ------------------ SVM for comparison -------------------

def svm_regressor(data_train, data_test, dataset_name='ewine'):

    # assign data for svm baseline

    global train_cir, train_err, test_cir, test_label
    if dataset_name == 'zenodo':
        train_cir, train_err, train_olabel, train_rlabel = data_train
        test_cir, test_err, test_olabel, test_rlabel = data_test
    elif dataset_name == 'ewine':
        train_cir, train_err, train_label = data_train
        test_cir, test_err, test_label = data_test

    # extract feature
    train_time = time.time()
    train_features = feature_extraction(train_cir)

    # error regression
    clf_reg = make_pipeline(StandardScaler(), SVR(gamma='auto'))
    clf_reg.fit(train_features, train_err)
    svr_train_time = time.time() - train_time

    test_time = time.time()
    test_features = feature_extraction(test_cir)
    err_est = clf_reg.predict(test_features)
    svr_test_time = time.time() - test_time

    # reshape
    err_test = test_err.reshape(test_err.shape[0])
    rmse_error = (np.sum((err_est - err_test) ** 2) / err_test.shape[0]) ** 0.5
    abs_error = (np.sum(np.abs(err_est - err_test)) / err_test.shape[0])
    print("SVM Regression Results: rmse %.3f, abs %.3f, time %.3f/%.3f"
          % (rmse_error, abs_error, svr_train_time, svr_test_time))

    return np.abs(err_est - err_test), np.abs(err_test), svr_test_time


def svm_classifier(data_train, data_test, dataset_name='ewine', level='room'):

    # assign data for svm baseline
    global train_cir, train_label, test_cir, test_label
    if dataset_name == 'zenodo':
        train_cir, train_err, train_olabel, train_rlabel = data_train
        test_cir, test_err, test_olabel, test_rlabel = data_test
        train_label = train_rlabel if level == 'room' else train_olabel
        test_label = test_rlabel if level == 'room' else test_olabel
    elif dataset_name == 'ewine':
        train_cir, train_err, train_label = data_train
        test_cir, test_err, test_label = data_test

    # extract feature
    train_time = time.time()
    train_features = feature_extraction(train_cir)

    # label classification
    clf_cls = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf_cls.fit(train_features, train_label.squeeze())
    svc_train_time = time.time() - train_time

    test_time = time.time()
    test_features = feature_extraction(test_cir)
    label_est = clf_cls.predict(test_features)
    svc_test_time = (time.time() - test_time) / test_features.shape[0]

    # reshape
    label_test = test_label.reshape(test_label.shape[0])
    accuracy = np.sum(label_est == label_test) / label_test.shape[0]
    print("SVM Classification Result: accuracy %.3f, time %.3f/%.3f" % (accuracy, svc_train_time, svc_test_time))

    return accuracy, svc_test_time





if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ewine", help="dataset for usage, ewine or zenodo")
    parser.add_argument("--dataset_env", type=str, default="nlos", help="dataset of different envs")
    parser.add_argument("--split_factor", type=str, default=0.8, help="split factor for train and test data")
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
    data_train, data_test = err_mitigation_dataset(
        root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env,
        split_factor=opt.split_factor, scaling=True
    )

    # error regression
    res_svm, err_gt, svr_test_time = svm_regressor(data_train, data_test, dataset_name=opt.dataset_name)
    print("error_abs: ", res_svm[0:10])
    print("error_gt: ", err_gt[0: 10])
    save_path = "saved_results/data_%s-%s/SVR" % (opt.dataset_name, opt.dataset_env)
    os.makedirs(save_path, exist_ok=True)
    # # plot marker
    # CDF_plot(err_gt, num=10, color='y', marker='o')
    # CDF_plot(res_svm, num=10, color='c', marker='*')
    # plot lines
    CDF_plot2(err_gt, color='y', marker='o')
    CDF_plot2(res_svm, color='c', marker='*')
    plt.legend(['Original Range Error', 'SVM Residual Error'], loc='lower right')
    plt.savefig(os.path.join(save_path, "CDF_svm.png"))
    plt.close()

    # env classification
    if opt.dataset_name == 'ewine':
        accuracy, svc_test_time = svm_classifier(data_train, data_test, dataset_name="ewine", level='room')
    elif opt.dataset_name == 'zenodo':
        accuracy_r, svc_test_time_r = svm_classifier(data_train, data_test, dataset_name="zenodo", level='room')
        accuracy_o, svc_test_time_o = svm_classifier(data_train, data_test, dataset_name="zenodo", level='obstacle')
    else:
        raise ValueError("Unknown dataset name.")