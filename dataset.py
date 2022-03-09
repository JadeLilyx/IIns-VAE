import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from data_tools import *


def err_mitigation_dataset(root, dataset_name='zenodo', dataset_env=None,
                           split_factor=0.8, scaling=False):

    train = []
    test = []
    # get data from files
    if dataset_name == 'zenodo':
        if not dataset_env:
            dataset_env = 'nlos'
        cir, err, obs_label, room_label = load_pkl_data(filepath=root, option=dataset_env)
        # print("test before spliting: ", room_label)
        train_err, test_err, train_cir, test_cir, train_olabel, test_olabel, train_rlabel, test_rlabel = train_test_split(
            err, cir, obs_label, room_label, train_size=split_factor, random_state=42
        )
        # print("test after splitting: ", test_rlabel)

        # scale cir data to N(0, 1)
        if scaling:
            scaler = StandardScaler()
            train_cir = scaler.fit_transform(train_cir)
            test_cir = scaler.transform(test_cir)

        train = train_cir, train_err, train_olabel, train_rlabel
        test = test_cir, test_err, test_olabel, test_rlabel  # notice the different labels here

    elif dataset_name == 'ewine':
        cir, err, label = load_reg_data(folderpaths=root)
        train_err, test_err, train_cir, test_cir, train_label, test_label = train_test_split(
            err, cir, label, train_size=split_factor, random_state=42)

        # scale cir data to N(0, 1)
        if scaling:
            scaler = StandardScaler()
            train_cir = scaler.fit_transform(train_cir)
            test_cir = scaler.transform(test_cir)

        train = train_cir, train_err, train_label
        test = test_cir, test_err, test_label

    # split and reshape data
    # train_err, test_err = np.split(err, np.array([int(np.size(err, 0) * split_factor)]))
    # train_cir, test_cir = np.split(cir, np.array([int(np.size(cir, 0) * split_factor)]))
    # train_label, test_label = np.split(label, np.array([int(np.size(label, 0) * split_factor)]))

    # train_err = np.reshape(train_err, (len(train_err), 1))
    # test_err = np.reshape(test_err, (len(test_err), 1))
    # train_label = np.reshape(train_label, (len(train_label), 1))
    # test_label = np.reshape(test_label, (len(test_label), 1))

    # # scale cir data to N(0, 1)
    # if scaling:
    #     scaler = StandardScaler()
    #     train_cir = scaler.fit_transform(train_cir)
    #     test_cir = scaler.transform(test_cir)
    #
    # train = train_cir, train_err, train_label
    # test = test_cir, test_err, test_label

    return train, test


class UWBDatasetZenodo(Dataset):
    def __init__(self, data):
        # basic data
        self.data = data
        cir_arr, err_arr, olabel_arr, rlabel_arr = self.data
        self.cir = cir_arr
        self.err = err_arr
        self.obs_label = olabel_arr
        self.room_label = rlabel_arr

    def __getitem__(self, index):
        # get basic data
        cir_item = torch.Tensor(self.cir[index % len(self.cir)])
        err_item = torch.Tensor(self.err[index % len(self.err)])
        olabel_item = torch.Tensor(self.obs_label[index % len(self.obs_label)])
        rlabel_item = torch.Tensor(self.room_label[index % len(self.room_label)])

        return {"CIR": cir_item, "Err": err_item, "Label": olabel_item, "RoomLabel": rlabel_item}

    def __len__(self):
        return len(self.cir)


class UWBDatasetEwine(Dataset):
    def __init__(self, data):
        # basic data
        self.data = data
        cir_arr, err_arr, label_arr = self.data
        self.cir = cir_arr
        self.err = err_arr
        self.label = label_arr

    def __getitem__(self, index):
        # get basic data
        cir_item = torch.Tensor(self.cir[index % len(self.cir)])
        err_item = torch.Tensor(self.err[index % len(self.err)])
        label_item = torch.Tensor(self.label[index %len(self.label)])

        return {"CIR": cir_item, "Err": err_item, "Label": label_item}

    def __len__(self):
        return len(self.cir)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ewine", help="dataset name")
    parser.add_argument("--dataset_env", type=str, default="nlos", help="different dataset environment labels")
    parser.add_argument("--split_factor", type=float, default=0.8, help="split for training and testing data")
    opt = parser.parse_args()

    # assign different roots of each dataset
    if opt.dataset_name == 'zenodo':
        root = './data/data_zenodo/dataset.pkl'
    elif opt.dataset_name == 'ewine':
        folderpaths = [
            './data/data_ewine/dataset1/',
            './data/data_ewine/dataset2'
        ]
        root = folderpaths

    # assign data for training and testing
    data_train, data_test = err_mitigation_dataset(
        root=root, dataset_name=opt.dataset_name, dataset_env=opt.dataset_env,
        split_factor=opt.split_factor, scaling=True
    )
    print("check length: ", len(data_train[0]), len(data_test[0]))
    # zenodo: 9078/2270, ewine: 27990/6998

    # dataset
    if opt.dataset_name == "ewine":
        dataset_train = UWBDatasetEwine(data_train)
        dataset_test = UWBDatasetEwine(data_test)
    elif opt.dataset_name == "zenodo":
        dataset_train = UWBDatasetZenodo(data_train)
        dataset_test = UWBDatasetZenodo(data_test)

    # dataloaders
    dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=8,
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=1,
    )

    for i, batch in enumerate(dataloader_test):
        cir_gt = batch["CIR"]
        err_gt = batch["Err"]
        label_gt = batch["Label"]

        if torch.cuda.is_available():
            cir_gt = cir_gt.cuda()
            err_gt = err_gt.cuda()
            label_gt = label_gt.cuda()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        env = Variable(torch.randn(cir_gt.size(0), 8, 1, 1).type(Tensor))
        if i == 1:
            print("cir type: ", cir_gt.type)
            print("env type: ", env.type)
            print("cir shape: ", cir_gt.shape)
            print("cir waveform: {}, range error: {}, env label: {}".format(cir_gt, err_gt, label_gt))