import os
import pandas as pd
import numpy as np
import math
import csv
import collections
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt


# ------------=- tools for ewine dataset ---------------------
def load_data_from_file(filepath):
    """
    Read selected .csv file to numpy array

    for dataset file like './dataset1/tag_room0.csv':
        + dataset1
        |__ tag_room0.csv
    """
    # print("Loading" + filepath + "...")
    # read data from file
    df = pd.read_csv(filepath, sep=',', header=0)
    output_arr = df.values
    
    return output_arr


def load_data_from_folder(folderpath):
    """
    Read selected .csv file to numpy array (in an inner folder with several .csv files)

    for dataset folder like './dataset2/tag_room1/':
        + dataset2
        |__ tag_room1
            |__ tag_room_part0.csv
            |__ tag_room_part1.csv
            |__ tag_room_part2.csv
            |__ tag_room_part3.csv
    """
    rootdir = folderpath
    output_arr = []
    first = 1
    # folderpath = './dataset2/tag_room1/'
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file)
            print("Loading " + filename + "...")
            # read data from array
            df = pd.read_csv(filename, sep=',', header=0)
            input_data = df.values
            # append to array
            if first > 0:
                first = 0
                output_arr = input_data
            else:
                output_arr = np.vstack((output_arr, input_data))

    return output_arr


def load_reg_data(filepaths):
    """
    Calculate ranging error and import cir data

    Parameters
    -----------
    filepaths: str, absolute path to input .csv file

    Returns
    ---------
    error_arr: numpy.array
        array of ranging errors from input data
    cir_arr: numpy.array
        array of cir vectors from .csv file (length=152)
    """
    # read data from files
    input_arr = load_data_from_file(filepaths[0])
    print("Loading " + filepaths[0] + "...")
    if len(filepaths) > 1:
        for item in filepaths[1:]:
            print("Loading " + item + "...")
            temp = load_data_from_file(item)
            input_arr = np.vstack((input_arr, temp))

    # randomize input array
    np.random.shuffle(input_arr)

    # create blank output_Arrays for error and cir
    data_len = 152
    error_arr = np.zeros((len(input_arr), 1))
    label_arr = np.zeros((len(input_arr), 1))
    cir_arr = np.zeros((len(input_arr), data_len))

    for i in range(len(input_arr)):
        fp_idx = int(input_arr[i][8])
        # calculate ranging error
        error_arr[i] = math.fabs(
            math.sqrt(
                math.pow(input_arr[i][0] - input_arr[i][2], 2) +
                math.pow(input_arr[i][1] - input_arr[i][3], 2)
            ) - input_arr[i][4]
        )  # d_{GT} - d_{M}

        # pack nlos label, 1 if nlos and 0 if los
        label_arr[i] = input_arr[i][5]

        # pack cir to output cir array (cir/max_amplitude)
        cir_arr[i] = input_arr[i][fp_idx + 15: fp_idx + 15 + data_len] / float(input_arr[i][17])

    # print(cir_arr.shape)
    return cir_arr, error_arr, label_arr


#--------------- tools for zenodo dataset --------------------
def load_pkl_data(filepath, option=None):
    print("Loading " + filepath + "...")
    # read data from file
    data = pd.read_pickle(filepath)

    cir_arr = []
    err_arr = []
    label_arr = []
    if option == 'room_full' or option is None:  # full, 55158
        # select samples with room label 0~4
        ds = np.asarray(data[['CIR', 'Error', 'Room']])
        np.random.shuffle(ds)
        cir_arr = np.vstack(ds[:, 0])
        err_arr = np.vstack(ds[:, 1])
        label_arr = np.vstack(ds[:, 2])  # 55158
        lroom_arr = label_arr

    elif option == 'obstacle_full':
        # select samples with 10 obstacle labels 0~9
        ds_1 = np.asarray(data.loc[data['Obstacles']=='0000000001'][['CIR', 'Error', 'Room']])  # 'Obstacles'
        ds_1 = np.asarray([np.array(x) for x in ds_1])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        label_arr_1 = np.zeros((cir_arr_1.shape[0], 1))  # 3987
        lroom_arr_1 = np.vstack(ds_1[:, 2])  # additional room label for train/test assign

        ds_2 = np.asarray(data.loc[data['Obstacles']=='0000000010'][['CIR', 'Error', 'Room']])
        ds_2 = np.asarray([np.array(x) for x in ds_1])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1))  # 2253
        lroom_arr_2 = np.vstack(ds_2[:, 2])

        ds_3 = np.asarray(data.loc[data['Obstacles']=='0000000100'][['CIR', 'Error', 'Room']])
        ds_3 = np.asarray([np.array(x) for x in ds_3])
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        label_arr_3 = np.ones((cir_arr_3.shape[0], 1)) * 2  # 417
        lroom_arr_3 = np.vstack(ds_3[:, 2])

        ds_4 = np.asarray(data.loc[data['Obstacles']=='0000001000'][['CIR', 'Error', 'Room']])
        ds_4 = np.asarray([np.array(x) for x in ds_4])
        cir_arr_4 = np.vstack(ds_4[:, 0])
        err_arr_4 = np.vstack(ds_4[:, 1])
        label_arr_4 = np.ones((cir_arr_4.shape[0], 1)) * 3  # 3581
        lroom_arr_4 = np.vstack(ds_4[:, 2])

        ds_5 = np.asarray(data.loc[data['Obstacles']=='0000010000'][['CIR', 'Error', 'Room']])
        ds_5 = np.asarray([np.array(x) for x in ds_5])
        cir_arr_5 = np.vstack(ds_5[:, 0])
        err_arr_5 = np.vstack(ds_5[:, 1])
        label_arr_5 = np.ones((cir_arr_5.shape[0], 1)) * 4  # 4182
        lroom_arr_5 = np.vstack(ds_5[:, 2])

        ds_6 = np.asarray(data.loc[data['Obstacles']=='0000100000'][['CIR', 'Error', 'Room']])
        ds_6 = np.asarray([np.array(x) for x in ds_6])
        cir_arr_6 = np.vstack(ds_6[:, 0])
        err_arr_6 = np.vstack(ds_6[:, 1])
        label_arr_6 = np.ones((cir_arr_6.shape[0], 1)) * 5  # 2888
        lroom_arr_6 = np.vstack(ds_6[:, 2])

        ds_7 = np.asarray(data.loc[data['Obstacles']=='0001000000'][['CIR', 'Error', 'Room']])
        ds_7 = np.asarray([np.array(x) for x in ds_7])
        cir_arr_7 = np.vstack(ds_7[:, 0])
        err_arr_7 = np.vstack(ds_7[:, 1])
        label_arr_7 = np.ones((cir_arr_7.shape[0], 1)) * 6  # 2966
        lroom_arr_7 = np.vstack(ds_7[:, 2])

        ds_8 = np.asarray(data.loc[data['Obstacles']=='0010000000'][['CIR', 'Error', 'Room']])
        ds_8 = np.asarray([np.array(x) for x in ds_8])
        cir_arr_8 = np.vstack(ds_8[:, 0])
        err_arr_8 = np.vstack(ds_8[:, 1])
        label_arr_8 = np.ones((cir_arr_8.shape[0], 1)) * 7  # 3354
        lroom_arr_8 = np.vstack(ds_8[:, 2])

        ds_9 = np.asarray(data.loc[data['Obstacles']=='0100000000'][['CIR', 'Error', 'Room']])
        ds_9 = np.asarray([np.array(x) for x in ds_9])
        cir_arr_9 = np.vstack(ds_9[:, 0])
        err_arr_9 = np.vstack(ds_9[:, 1])
        label_arr_9 = np.ones((cir_arr_9.shape[0], 1)) * 8  # 1971
        lroom_arr_9 = np.vstack(ds_9[:, 2])

        ds_10 = np.asarray(data.loc[data['Obstacles']=='1000000000'][['CIR', 'Error', 'Room']])
        ds_10 = np.array([np.array(x) for x in ds_10])
        cir_arr_10 = np.vstack(ds_10[:, 0])
        err_arr_10 = np.vstack(ds_10[:, 1])
        label_arr_10 = np.ones((cir_arr_10.shape[0], 1)) * 9  # 954
        lroom_arr_10 = np.vstack(ds_10[:, 2])

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3, cir_arr_4, cir_arr_5, cir_arr_6, cir_arr_7, cir_arr_8, cir_arr_9, cir_arr_10))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3, err_arr_4, err_arr_5, err_arr_6, err_arr_7, err_arr_8, err_arr_9, err_arr_10))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3, label_arr_4, label_arr_5, label_arr_6, label_arr_7, label_arr_8, label_arr_9, label_arr_10))
        lroom_arr = np.vstack((lroom_arr_1, lroom_arr_2, lroom_arr_3, lroom_arr_4, lroom_arr_5, lroom_arr_6, lroom_arr_7, lroom_arr_8, lroom_arr_9, lroom_arr_10))
        data_module = np.hstack((cir_arr, err_arr, label_arr, lroom_arr))
        np.random.shuffle(data_module)
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:len(cir_arr[0])+2]  # 26553
        lroom_arr = data_module[:, len(cir_arr[0])+2:]
    
    elif option == 'nlos':  # 0~1
        ds_los = np.asarray(data.loc[data['Obstacles']=='0000000000'][['CIR', 'Error', 'Room']])
        cir_arr_los = np.vstack(ds_los[:, 0])
        err_arr_los = np.vstack(ds_los[:, 1])
        label_arr_los = np.zeros((cir_arr_los.shape[0], 1))
        lroom_arr_los = np.vstack(ds_los[:, 2])
        # print("los samples: ", cir_arr_los.shape[0])  # 4691

        number = 1
        for i in range(1, 4):  # 11
            target_str = '0' * (10 - i) + str(number)
            # print(target_str)  # one-hot version
            ds_nlos_i = np.asarray(data.loc[data['Obstacles']==target_str][['CIR', 'Error', 'Room']])
            cir_arr_nlos_i = np.vstack(ds_nlos_i[:, 0])
            err_arr_nlos_i = np.vstack(ds_nlos_i[:, 1])
            lroom_arr_nlos_i = np.vstack(ds_nlos_i[:, 2])
            if i == 1:
                cir_arr_nlos = cir_arr_nlos_i
                err_arr_nlos = err_arr_nlos_i
                lroom_arr_nlos = lroom_arr_nlos_i
            else:
                cir_arr_nlos = np.vstack((cir_arr_nlos, cir_arr_nlos_i))
                err_arr_nlos = np.vstack((err_arr_nlos, err_arr_nlos_i))
                lroom_arr_nlos = np.vstack((lroom_arr_nlos, lroom_arr_nlos_i))
            number *= 10
        label_arr_nlos = np.ones((cir_arr_nlos.shape[0], 1))
        # print("nlos samples: ", cir_arr_nlos.shape[0])  # 6657 for 3 and 26553 for 10 (range(1, 11))

        cir_arr = np.vstack((cir_arr_los, cir_arr_nlos))
        err_arr = np.vstack((err_arr_los, err_arr_nlos))
        label_arr = np.vstack((label_arr_los, label_arr_nlos))
        lroom_arr = np.vstack((lroom_arr_los, lroom_arr_nlos))
        data_module = np.hstack((cir_arr, err_arr, label_arr, lroom_arr))
        np.random.shuffle(data_module)
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:len(cir_arr[0])+2]
        lroom_arr = data_module[:, len(cir_arr[0])+2:]

    elif option == 'room_part':
        # big room
        ds_1 = np.asarray(data.loc[data['Room']==1][['CIR', 'Error', 'Room']])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        label_arr_1 = np.zeros((cir_arr_1.shape[0], 1))
        lroom_arr_1 = np.vstack(ds_1[:, 2])  # 18422

        # medium room
        ds_2 = np.asarray(data.loc[data['Room']==2][['CIR', 'Error', 'Room']])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1))
        lroom_arr_2 = np.vstack(ds_2[:, 2])  # 13210

        # small room
        ds_3 = np.asarray(data.loc[data['Room']==1][['CIR', 'Error', 'Room']])
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        label_arr_3 = np.ones((cir_arr_3.shape[0], 1)) * 2
        lroom_arr_3 = np.vstack(ds_3[:, 2])  # 18422

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3))
        lroom_arr = np.vstack((lroom_arr_1, lroom_arr_2, lroom_arr_3))
        data_module = np.hstack((cir_arr, err_arr, label_arr, lroom_arr))
        np.random.shuffle(data_module)
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:len(cir_arr[0])+2]
        lroom_arr = data_module[:, len(cir_arr[0])+2:]

    elif option == 'obstacle_part':
        # select samples with 4 obstacle labels
        # metal
        ds_mw = np.asarray(data.loc[data['Obstacles']=='0000000001'][['CIR', 'Error', 'Room']])  # 'Obstacles'
        ds_mp = np.asarray(data.loc[data['Obstacles']=='0000001000'][['CIR', 'Error', 'Room']])
        ds_1 = np.vstack((ds_mw, ds_mp))
        # ds_1 = np.asarray([np.array(x) for x in ds_1])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        label_arr_1 = np.zeros((cir_arr_1.shape[0], 1))
        lroom_arr_1 = np.vstack(ds_1[:, 2])
        # print('metal:', ds_mw.shape, ds_1.shape)  # 3987, 7568

        # wood
        ds_2 = np.asarray(data.loc[data['Obstacles']=='0000000100'][['CIR', 'Error', 'Room']])
        # ds_2 = np.asarray([np.array(x) for x in ds_2])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1))  # 2253
        lroom_arr_2 = np.vstack(ds_2[:, 2])
        # print('wood:', ds_2.shape)  # 417

        # plastic
        ds_3 = np.asarray(data.loc[data['Obstacles']=='0010000000'][['CIR', 'Error', 'Room']])
        # ds_3 = np.asarray([np.array(x) for x in ds_3])
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        label_arr_3 = np.ones((cir_arr_3.shape[0], 1)) * 2  # 417
        lroom_arr_3 = np.vstack(ds_3[:, 2])
        # print('plastic:', ds_3.shape)  # 3354

        # glass
        ds_4 = np.asarray(data.loc[data['Obstacles']=='0000000010'][['CIR', 'Error', 'Room']])
        # ds_4 = np.asarray([np.array(x) for x in ds_4])
        cir_arr_4 = np.vstack(ds_4[:, 0])
        err_arr_4 = np.vstack(ds_4[:, 1])
        label_arr_4 = np.ones((cir_arr_4.shape[0], 1)) * 3  # 3581
        lroom_arr_4 = np.vstack(ds_4[:, 2])
        # print('glass:', ds_4.shape)  # 2253

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3, cir_arr_4))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3, err_arr_4))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3, label_arr_4))
        lroom_arr = np.vstack((lroom_arr_1, lroom_arr_2, lroom_arr_3, lroom_arr_4))
        data_module = np.hstack((cir_arr, err_arr, label_arr, lroom_arr))
        np.random.shuffle(data_module)  # only this way can shuffle the labels
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:len(cir_arr[0]+2)]  # 13592
        lroom_arr = data_module[:, len(cir_arr[0])+2:]

    return cir_arr, err_arr, label_arr, lroom_arr  # (n, 1), (n, 157), (n, 1)


def feature_extraction(cir_data):

    # extract max amplitude and position from each sample
    cir_data_list = cir_data.tolist()
    M_AMP = []
    max_pos = []
    for list_item in cir_data_list:
        max_item = max(list_item)
        max_index = list_item.index(max_item)
        M_AMP.append(max_item)
        max_pos.append(max_index)

    # rise time
    R_T = []
    for index in range(len(cir_data_list)):
        mean_n = np.nanmean(np.asarray(cir_data_list[index][:]))
        sigma_n = np.nanstd(np.asarray(cir_data_list[index][:]))

        rise_t1 = np.argwhere(np.asarray(cir_data_list[index]) > (6 * (sigma_n + mean_n)))
        rise_t2 = np.argwhere(np.asarray(cir_data_list[index]) > (0.6 * M_AMP[index]))

        if rise_t1.size == 0:
            rise_t1 = [[0]]
        if rise_t2.size == 0:
            rise_t2 = [[0]]
        rise_time = max(0, rise_t2[0][0] - rise_t1[0][0])
        R_T.append(rise_time)

    # window
    data_w = []
    for index in range(len(cir_data_list)):
        if max_pos[index] - 20 < 0:
            data_w.append(cir_data_list[index][0 : 35])
        elif max_pos[index] + 15 > len(cir_data_list[index]):
            length = len(cir_data_list[index])
            data_w.append(cir_data_list[index][length - 35 : length])
        else:
            data_w.append(cir_data_list[index][max_pos[index] - 20 : max_pos[index] + 15])

    # energy
    data_w_np = np.asarray(data_w)
    data_w_np_power_2 = data_w_np ** 2
    Er = np.nansum(data_w_np, axis=1)

    # mean excess delay
    fhi = []
    T_EMD = []
    T_RMS = []
    for index1 in range(len(data_w)):
        fhi.append(data_w_np_power_2[index1] / Er[index1])
        T_EMD.append(0)
        T_RMS.append(0)
        for index2 in range(len(data_w[index1])):
            T_EMD[index1] += (index2 + 1) * fhi[index1][index2]
            T_RMS[index1] += ((index2 + 1 - (index2 + 2) * fhi[index1][index2]) ** 2) * fhi[index1][index2]

    # kurtosis
    mu = []
    sigma = []
    Kur = []
    for index1 in range(len(data_w)):
        mu.append(np.nansum(data_w[index1]) / len(data_w[index1]))
    mu_np = np.asarray(mu)
    for index1 in range(len(data_w)):
        square_temp = (np.asarray(data_w[index1]) - mu_np[index1]) ** 2
        sigma.append(np.nansum(square_temp) / len(data_w[index1]))
    for index1 in range(len(data_w)):
        power_4_temp = (np.asarray(data_w[index1]) - mu_np[index1])
        Kur.append(np.nansum(power_4_temp) / (len(data_w[index1]) * (sigma[index1]) ** 2))

    feature = list()
    for index1 in range(len(Er)):
        feature.append([Er[index1], T_EMD[index1], T_RMS[index1], Kur[index1], R_T[index1], M_AMP[index1]])

    return np.asarray(feature)
            

def label_dictionary(dataset_env):

    # match digit labels to string labels (0~n-1)
    if dataset_env == 'nlos':
        label_str_dict = {
            0: 'los', 1: 'nlos'
        }
    elif dataset_env == 'room_full':
        label_str_dict = {
            0: 'cross-room', 1: 'big room', 2: 'medium room', 3: 'small room', 4: 'outdoor'
        }
    elif dataset_env == 'obstacle_full':
        label_str_dict = {
            0: 'metal window', 1: 'glass plate', 2: 'wood door', 3: 'metal plate', 4: 'LCD TV',
            5: 'cardboard box', 6: 'plywood plate', 7: 'plastic', 8: 'polystyrene plate', 9: 'wall'
        }
    elif dataset_env == 'room_part':
        label_str_dict = {
            0: 'big room', 1: 'medium room', 2: 'small room'
        }
    elif dataset_env == 'obstacle_part':
        label_str_dict = {
            0: 'metal', 1: 'wood', 2: 'plastic', 3: 'glass'
        }

    return label_str_dict


def label_int2str(dataset_env, label_int):

    label_str_dict = label_dictionary(dataset_env)

    label_str = label_str_dict[label_int]
    return label_str


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ewine", help="dataset for usage, ewine or zenodo")
    parser.add_argument("--dataset_env", type=str, default="nlos", help="dataset of different environments")
    opt = parser.parse_args()

    # import extracted err and cir
    os.makedirs("saved_results", exist_ok=True)
    if opt.dataset_name == 'zenodo':
        cir_reg, err_reg, label_reg, lroom_reg = load_pkl_data('data/data_zenodo/dataset.pkl', option=opt.dataset_env)
        plt.plot(cir_reg[0], color='blue')
        label_int = label_reg[0][0]
        label_str = label_int2str(opt.dataset_env, label_int)
        plt.savefig("saved_results/%s_sample_%s.png" % (opt.dataset_name, label_str))
        plt.close()
    elif opt.dataset_name == 'ewine':
        filepaths = ['./data/data_ewine/dataset1/tag_room0.csv',
                     './data/data_ewine/dataset1/tag_room1.csv',
                     './data/data_ewine/dataset2/tag_room0.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part0.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part1.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part2.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part3.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part4.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part5.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part6.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part7.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part8.csv',
                     './data/data_ewine/dataset2/tag_room1/tag_room1_part9.csv']
        cir_reg, err_reg, label_reg = load_reg_data(filepaths)
        plt.plot(cir_reg[0], color='green')
        plt.savefig("saved_results/%s_sample.png" % opt.dataset_name)
        plt.close()
        
    # import extracted features
    features = feature_extraction(cir_reg)
    print(features[0])
