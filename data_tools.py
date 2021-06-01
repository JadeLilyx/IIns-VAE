import os
import pandas as pd
import numpy as np
import math
import csv
import collections
from sklearn.preprocessing import StandardScaler
# from sklearn.externals import joblib
import joblib
import matplotlib.pyplot as plt


# -------------------------------------------------
#               csv data of ewine
# -------------------------------------------------

def load_data_from_file(filepath):
    """
    Read selected .csv file to numpy array

    for dataset file like './dataset_reg1/tag_room0.csv':
    + dataset1
    |__ tag_room0.csv
    """
    print("Loading" + filepath + "...")
    # read data from file
    df = pd.read_csv(filepath, sep=',', header=0)
    output_arr = df.values  #as_matrix()

    return output_arr


def load_data_from_folder(folderpath):
    """
    Read selected .csv file to numpy array (in an inner folder with several .csv files)

    for dataset folder like './dataset_reg2/tag_room1/':
    + dataset2
    |__ tag_room1
      |__ tag_room1_part0.csv
      |__ tag_room1_part1.csv
      |__ tag_room1_part2.csv
      |__ tag_room1_part3.csv
    """
    rootdir = folderpath
    output_arr = []
    first = 1
    # rootdir = './dataset_reg2/tag_room1/'
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file)
            print("Loading " + filename + "...")
            # read data from array
            df = pd.read_csv(filename, sep=',', header=0)
            input_data = df.values  # as_matrix()
            # append to array
            if first > 0:
                first = 0
                output_arr = input_data
            else:
                output_arr = np.vstack((output_arr, input_data))

    return output_arr


def dist_gt(data, tag_h, anch_h):
    dist = math.sqrt(
        math.pow(data[1] - data[3], 2) + math.pow(data[2] - data[4], 2) + math.pow(tag_h - anch_h, 2)
    )
    return dist


def load_cls_data(folderpath):
    """
    Read selected .csv file to numpy array (in an inner folder with several .csv files)

    format: |LOS/NLOS label|data...| -> |NLOS|LOS|data...|
    for classification dataset './dataset_cls/':
    + dataset_cls
    |__ uwb_dataset_part1.csv
    |__ uwb_dataset_part2.csv
    |__ uwb_dataset_part3.csv
    |__ uwb_dataset_part4.csv
    |__ uwb_dataset_part5.csv
    |__ uwb_dataset_part6.csv
    |__ uwb_dataset_part7.csv
    """
    # search dir for files
    rootdir = folderpath
    output_arr = []
    first = 1
    # './dataset_cls/'
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file)
            print("Loading " + filename + "...")
            # read data from array
            df = pd.read_csv(filename, sep=',', header=1)
            input_data = df.values  # as_matrix()

            # expand 2 elements for label; omit last element because it is empty (nan)
            output_data = np.zeros((len(input_data), 1 + input_data.shape[1] - 1))
            # set NLOS status from filename
            for i in range(input_data.shape[0]):
                if input_data[i, 0] == 0:  # LOS=0
                    output_data[i, 0] = 1  # NLOS=1
                else:
                    output_data[i, 0] = 0

            # put data into output array, omit last, LOS/NLOS to LOS|NLOS 2 elements
            output_data[:, 1:] = input_data[:, :-1]  # expand a NLOS label on top

            # append files to array
            if first > 0:
                first = 0
                output_arr = output_data  # input_data
            else:
                output_arr = np.vstack((output_arr, output_data))

    # prepare nlos label and cir array
    data_len = 152
    label_arr = np.zeros((len(output_arr), 2))
    cir_arr = np.zeros((len(output_arr), data_len))

    # read data_len samples of CIR from the first path index
    for i in range(len(output_arr)):
        fp_idx = int(output_arr[i][3])
        label_arr[i][0: 2] = output_arr[i][0:2]
        cir_arr[i] = output_arr[i][fp_idx + 2 + 10: fp_idx + 2 + 10 + data_len] / float(output_arr[i, 10])

    # print(label_arr.shape)  # (41993, 2)
    # print(cir_arr.shape)  # (41993, 152)
    return label_arr, cir_arr  # (n, 2), (n, 152)


def load_reg_data(filepaths):
    """
    Calculate ranging error and import cir data

    Parameters
    -----------
    filepaths: str, absolute path to input .csv file

    Returns
    ----------
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

    # create blank output_arrays for error and cir
    data_len = 152  # cir length
    error_arr = np.zeros((len(input_arr), 1))
    cir_arr = np.zeros((len(input_arr), data_len))

    for i in range(len(input_arr)):
        fp_idx = int(input_arr[i][8])
        # calculate ranging error
        error_arr[i] = math.fabs(
            math.sqrt(
                math.pow(input_arr[i][0] - input_arr[i][2], 2) +
                math.pow(input_arr[i][1] - input_arr[i][3], 2)
            ) - input_arr[i][4]
        )  # d_{GT} - d_M

        # pack cir to output cir array
        cir_arr[i] = input_arr[i][fp_idx + 15: fp_idx + 15 + data_len] / float(input_arr[i][17])

    # print(error_arr.shape)  # (31489, 1)
    # print(cir_arr.shape)  # (31489, 152)
    return error_arr, cir_arr  # (n, 1), (n, 152)

# -------------------------------------------
#             pkl data of zenodo
# -------------------------------------------


def load_pkl_data_option(filepath, option=None):
    """
    Read selected .pkl file to numpy array

    for dataset file like './dataset.pkl'
    sample structure:
        |CIR (157, float)|error|room(int)|obstacle(10, bool)|
    """
    print("Loading" + filepath + "...")
    # read data from file
    data = pd.read_pickle(filepath)

    cir_arr = []
    err_arr = []
    if option is None:
        # select all samples
        ds = np.asarray(data[['CIR', 'Error']])
        cir_arr = np.vstack(ds[:, 0])  # (55158, 157)
        err_arr = np.vstack(ds[:, 1])  # (55158, 1)
    elif option == 'Room_big':
        # select specific rooms (room encoding: 0 for cross-room, 1 for big, 2 for medium, 3 for small, 4 for outdoor)
        ds = np.asarray(data.loc[data['Room'] == 1][['CIR', 'Error']])
        cir_arr = np.vstack(ds[:, 0])
        err_arr = np.vstack(ds[:, 1])  # 18422
    elif option == 'Obstacles':
        # select specific obstacle configurations (1-hot encoding)
        ds = np.asarray(data.loc[data['Obstacles'] == '0000000001'][['CIR', 'Error']])
        cir_arr = np.vstack(ds[:, 0])
        err_arr = np.vstack(ds[: 1])  # (3987)

    return err_arr, cir_arr  # (n, 1), (n, 157)


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

    elif option == 'obstacle_full':
        # select samples with 10 obstacle labels
        ds_1 = np.asarray(data.loc[data['Obstacles']=='0000000001'][['CIR', 'Error', 'Obstacles']])
        ds_1 = np.asarray([np.array(x) for x in ds_1])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        label_arr_1 = np.ones((cir_arr_1.shape[0], 1))  # 3987

        ds_2 = np.asarray(data.loc[data['Obstacles']=='0000000010'][['CIR', 'Error', 'Obstacles']])
        ds_2 = np.asarray([np.array(x) for x in ds_2])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1)) * 2  # 2253

        ds_3 = np.asarray(data.loc[data['Obstacles']=='0000000100'][['CIR', 'Error', 'Obstacles']])
        ds_3 = np.asarray([np.array(x) for x in ds_3])
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        label_arr_3 = np.ones((cir_arr_3.shape[0], 1)) * 3  # 417

        ds_4 = np.asarray(data.loc[data['Obstacles']=='0000001000'][['CIR', 'Error', 'Obstacles']])
        ds_4 = np.asarray([np.array(x) for x in ds_4])
        cir_arr_4 = np.vstack(ds_4[:, 0])
        err_arr_4 = np.vstack(ds_4[:, 1])
        label_arr_4 = np.ones((cir_arr_4.shape[0], 1)) * 4  # 3581

        ds_5 = np.asarray(data.loc[data['Obstacles']=='0000010000'][['CIR', 'Error', 'Obstacles']])
        ds_5 = np.asarray([np.array(x) for x in ds_5])
        cir_arr_5 = np.vstack(ds_5[:, 0])
        err_arr_5 = np.vstack(ds_5[:, 1])
        label_arr_5 = np.ones((cir_arr_5.shape[0], 1)) * 5  # 4182

        ds_6 = np.asarray(data.loc[data['Obstacles']=='0000100000'][['CIR', 'Error', 'Obstacles']])
        ds_6 = np.asarray([np.array(x) for x in ds_6])
        cir_arr_6 = np.vstack(ds_6[:, 0])
        err_arr_6 = np.vstack(ds_6[:, 1])
        label_arr_6 = np.ones((cir_arr_6.shape[0], 1)) * 6  # 2888

        ds_7 = np.asarray(data.loc[data['Obstacles']=='0001000000'][['CIR', 'Error', 'Obstacles']])
        ds_7 = np.asarray([np.array(x) for x in ds_7])
        cir_arr_7 = np.vstack(ds_7[:, 0])
        err_arr_7 = np.vstack(ds_7[:, 1])
        label_arr_7 = np.ones((cir_arr_7.shape[0], 1)) * 7  # 2966

        ds_8 = np.asarray(data.loc[data['Obstacles']=='0010000000'][['CIR', 'Error', 'Obstacles']])
        ds_8 = np.asarray([np.array(x) for x in ds_8])
        cir_arr_8 = np.vstack(ds_8[:, 0])
        err_arr_8 = np.vstack(ds_8[:, 1])
        label_arr_8 = np.ones((cir_arr_8.shape[0], 1)) * 8  # 3354

        ds_9 = np.asarray(data.loc[data['Obstacles']=='0100000000'][['CIR', 'Error', 'Obstacles']])
        ds_9 = np.asarray([np.array(x) for x in ds_9])
        cir_arr_9 = np.vstack(ds_9[:, 0])
        err_arr_9 = np.vstack(ds_9[:, 1])
        label_arr_9 = np.ones((cir_arr_9.shape[0], 1)) * 9  # 1971

        ds_10 = np.asarray(data.loc[data['Obstacles']=='1000000000'][['CIR', 'Error', 'Obstacles']])
        ds_10 = np.array([np.array(x) for x in ds_10])
        cir_arr_10 = np.vstack(ds_10[:, 0])
        err_arr_10 = np.vstack(ds_10[:, 1])
        label_arr_10 = np.ones((cir_arr_10.shape[0], 1)) * 10  # 954

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3, cir_arr_4, cir_arr_5, cir_arr_6, cir_arr_7, cir_arr_8, cir_arr_9, cir_arr_10))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3, err_arr_4, err_arr_5, err_arr_6, err_arr_7, err_arr_8, err_arr_9, err_arr_10))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3, label_arr_4, label_arr_5, label_arr_6, label_arr_7, label_arr_8, label_arr_9, label_arr_10))
        data_module = np.hstack((cir_arr, err_arr, label_arr))
        np.random.shuffle(data_module)
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:]  # 26553

    elif option == 'room_part':
        # big room
        ds_1 = np.asarray(data.loc[data['Room']==1][['CIR', 'Error', 'Room']])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        label_arr_1 = np.vstack(ds_1[:, 2])
        # print("big:", ds_1.shape)  # 18422

        # medium room
        ds_2 = np.asarray(data.loc[data['Room']==2][['CIR', 'Error', 'Room']])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        label_arr_2 = np.vstack(ds_2[:, 2])
        # print("med:", ds_2.shape)  # 13210

        # small room
        ds_3 = np.asarray(data.loc[data['Room']==1][['CIR', 'Error', 'Room']])
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        label_arr_3 = np.vstack(ds_3[:, 2])
        # print("small:", ds_3.shape)  # 18422

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3))
        data_module = np.hstack((cir_arr, err_arr, label_arr))
        np.random.shuffle(data_module)  # only together can shuffle the labels
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:]  # 50054

    elif option == 'room_full_rough':
        # trhough wall
        ds_1 = np.asarray(data.loc[data['Room']==0][['CIR', 'Error']])  # 'Room'
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        # label_arr_1 = np.vstack(ds_1[:, 2])
        label_arr_1 = np.ones((cir_arr_1.shape[0], 1))
        # print("through:", ds_1.shape)  # 954

        # outdoor
        ds_2 = np.asarray(data.loc[data['Room']==4][['CIR', 'Error']])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        # label_arr_2 = np.vstack(ds_2[:, 2])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1)) * 2
        # print("out:", ds_2.shape)  # 4971

        # rooms (big medium small)
        ds_rb = np.asarray(data.loc[data['Room']==1][['CIR', 'Error']])
        ds_rm = np.asarray(data.loc[data['Room']==2][['CIR', 'Error']])
        ds_rs = np.asarray(data.loc[data['Room']==3][['CIR', 'Error']])
        ds_3 = np.vstack((ds_rb, ds_rm, ds_rs))
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        # label_arr_3 = np.vstack(ds_3[:, 2])
        label_arr_3 = np.ones((cir_arr_3.shape[0], 1)) * 3
        # print("ds_3:", ds_rb.shape, ds_3.shape)  # 18422, 49233

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3))
        data_module = np.hstack((cir_arr, err_arr, label_arr))
        np.random.shuffle(data_module)
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:]  # 55158

    elif option == 'obstacle_part':
        # select samples with 4 obstacle labels
        # metal
        ds_mw = np.asarray(data.loc[data['Obstacles']=='0000000001'][['CIR', 'Error']])  # 'Obstacles'
        ds_mp = np.asarray(data.loc[data['Obstacles']=='0000001000'][['CIR', 'Error']])
        ds_1 = np.vstack((ds_mw, ds_mp))
        # ds_1 = np.asarray([np.array(x) for x in ds_1])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        label_arr_1 = np.ones((cir_arr_1.shape[0], 1))
        # print('metal:', ds_mw.shape, ds_1.shape)  # 3987, 7568

        # wood
        ds_2 = np.asarray(data.loc[data['Obstacles']=='0000000100'][['CIR', 'Error']])
        # ds_2 = np.asarray([np.array(x) for x in ds_2])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1)) * 2  # 2253
        # print('wood:', ds_2.shape)  # 417

        # plastic
        ds_3 = np.asarray(data.loc[data['Obstacles']=='0010000000'][['CIR', 'Error']])
        # ds_3 = np.asarray([np.array(x) for x in ds_3])
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        label_arr_3 = np.ones((cir_arr_3.shape[0], 1)) * 3  # 417
        # print('plastic:', ds_3.shape)  # 3354

        # glass
        ds_4 = np.asarray(data.loc[data['Obstacles']=='0000000010'][['CIR', 'Error']])
        # ds_4 = np.asarray([np.array(x) for x in ds_4])
        cir_arr_4 = np.vstack(ds_4[:, 0])
        err_arr_4 = np.vstack(ds_4[:, 1])
        label_arr_4 = np.ones((cir_arr_4.shape[0], 1)) * 4  # 3581
        # print('glass:', ds_4.shape)  # 2253

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3, cir_arr_4))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3, err_arr_4))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3, label_arr_4))
        data_module = np.hstack((cir_arr, err_arr, label_arr))
        np.random.shuffle(data_module)  # only this way can shuffle the labels
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:]  # 13592

    elif option == 'obstacle_part2':
        # select samples with 4 obstacle labels
        # metal
        ds_mw = np.asarray(data.loc[data['Obstacles']=='0000000001'][['CIR', 'Error']])  # 'Obstacles'
        ds_mp = np.asarray(data.loc[data['Obstacles']=='0000001000'][['CIR', 'Error']])
        ds_1 = np.vstack((ds_mw, ds_mp))
        # ds_1 = np.asarray([np.array(x) for x in ds_1])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        label_arr_1 = np.ones((cir_arr_1.shape[0], 1))
        # print('heavy:', ds_1.shape)  # 7568

        # wood, plastic, glass
        ds_w = np.asarray(data.loc[data['Obstacles']=='0000000100'][['CIR', 'Error']])
        ds_p = np.asarray(data.loc[data['Obstacles']=='0010000000'][['CIR', 'Error']])
        ds_g = np.asarray(data.loc[data['Obstacles']=='0000000010'][['CIR', 'Error']])
        ds_2 = np.vstack((ds_w, ds_p, ds_g))
        # ds_2 = np.asarray([np.array(x) for x in ds_2])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1)) * 2  
        # print('light:', ds_2.shape)  # 2253

        cir_arr = np.vstack((cir_arr_1, cir_arr_2))
        err_arr = np.vstack((err_arr_1, err_arr_2))
        label_arr = np.vstack((label_arr_1, label_arr_2))
        data_module = np.hstack((cir_arr, err_arr, label_arr))
        np.random.shuffle(data_module)  # only this way can shuffle the labels
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:]  # 6024
        # print(label_arr)
    
    elif option == 'room_full_rough2':
        # outdoor
        ds_1 = np.asarray(data.loc[data['Room']==4][['CIR', 'Error']])
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        # label_arr_2 = np.vstack(ds_2[:, 2])
        label_arr_1 = np.ones((cir_arr_1.shape[0], 1))
        # print("ds_1:", ds_1.shape)  # (4971, 2)

        # rooms (big medium small across)
        ds_rb = np.asarray(data.loc[data['Room']==1][['CIR', 'Error']])
        ds_rm = np.asarray(data.loc[data['Room']==2][['CIR', 'Error']])
        ds_rs = np.asarray(data.loc[data['Room']==3][['CIR', 'Error']])
        ds_aw = np.asarray(data.loc[data['Room']==0][['CIR', 'Error']])
        ds_2 = np.vstack((ds_rb, ds_rm, ds_rs, ds_aw))
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        # label_arr_3 = np.vstack(ds_3[:, 2])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1)) * 2
        # print("ds_2:", ds_2.shape)  # (50187, 2)

        cir_arr = np.vstack((cir_arr_1, cir_arr_2))
        err_arr = np.vstack((err_arr_1, err_arr_2))
        label_arr = np.vstack((label_arr_1, label_arr_2))
        data_module = np.hstack((cir_arr, err_arr, label_arr))
        np.random.shuffle(data_module)
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:]  # 55158

    elif option == 'paper':  # 1: 'cross-room', 2: 'big room', 3: 'medium room', 4: 'small room'
        # trhough wall
        ds_1 = np.asarray(data.loc[data['Room']==0][['CIR', 'Error']])  # 'Room'
        cir_arr_1 = np.vstack(ds_1[:, 0])
        err_arr_1 = np.vstack(ds_1[:, 1])
        # label_arr_1 = np.vstack(ds_1[:, 2])
        label_arr_1 = np.ones((cir_arr_1.shape[0], 1))
        # print("through:", ds_1.shape)  # 954

        # big room
        ds_2 = np.asarray(data.loc[data['Room']==1][['CIR', 'Error']])
        cir_arr_2 = np.vstack(ds_2[:, 0])
        err_arr_2 = np.vstack(ds_2[:, 1])
        # label_arr_2 = np.vstack(ds_2[:, 2])
        label_arr_2 = np.ones((cir_arr_2.shape[0], 1)) * 2
        # print("out:", ds_2.shape)  # 4971

        # medium room
        ds_3 = np.asarray(data.loc[data['Room']==2][['CIR', 'Error']])
        cir_arr_3 = np.vstack(ds_3[:, 0])
        err_arr_3 = np.vstack(ds_3[:, 1])
        # label_arr_2 = np.vstack(ds_2[:, 2])
        label_arr_3 = np.ones((cir_arr_3.shape[0], 1)) * 3
        # print("out:", ds_2.shape)  # 4971

        # small room
        ds_4 = np.asarray(data.loc[data['Room']==3][['CIR', 'Error']])
        cir_arr_4 = np.vstack(ds_4[:, 0])
        err_arr_4 = np.vstack(ds_4[:, 1])
        # label_arr_3 = np.vstack(ds_3[:, 2])
        label_arr_4 = np.ones((cir_arr_4.shape[0], 1)) * 4
        # print("ds_3:", ds_rb.shape, ds_3.shape)  # 18422, 49233

        cir_arr = np.vstack((cir_arr_1, cir_arr_2, cir_arr_3, cir_arr_4))
        err_arr = np.vstack((err_arr_1, err_arr_2, err_arr_3, err_arr_4))
        label_arr = np.vstack((label_arr_1, label_arr_2, label_arr_3, label_arr_4))
        data_module = np.hstack((cir_arr, err_arr, label_arr))
        np.random.shuffle(data_module)
        cir_arr = data_module[:, 0:len(cir_arr[0])]
        err_arr = data_module[:, len(cir_arr[0]):len(cir_arr[0])+1]
        label_arr = data_module[:, len(cir_arr[0])+1:]  # 55158

    return err_arr, cir_arr, label_arr  # (n, 1), (n, 157), (n, 1)


# -----------------------------------------------
#     extract features from cir data
# -----------------------------------------------


def feature_extraction(cir_data):
    
    # extract max amplitude and position from each sample M_AMP
    cir_data_list = cir_data.tolist()
    M_AMP = []
    max_pos = []
    for list_item in cir_data_list:
        max_item = max(list_item)
        max_index = list_item.index(max_item)
        M_AMP.append(max_item)
        max_pos.append(max_index)

    # rise time R_T
    R_T = []
    for index in range(len(cir_data_list)):
        # mean_n = np.nanmean(cir_data_list[index][:])
        mean_n = np.nanmean(np.asarray(cir_data_list[index][:]))
        sigma_n = np.nanstd(np.asarray(cir_data_list[index][:]))

        rise_t1 = np.argwhere(np.asarray(cir_data_list[index]) > (6 * (sigma_n + mean_n)))
        rise_t2 = np.argwhere(np.asarray(cir_data_list[index]) > (0.6 * M_AMP[index]))

        if rise_t1.size == 0:
            rise_t1 = [[0]]
        if rise_t2.size == 0:
            rise_t2 = [[0]]
        # print(rise_t1[0][0], rise_t2[0][0])
        rise_time = max(0, rise_t2[0][0] - rise_t1[0][0])
        R_T.append(rise_time)

    # window T
    data_w = []
    for index in range(len(cir_data_list)):
        # data_w.append(cir_data_list[index][max_pos[index] - 20 : max_pos[index] + 15])
        if max_pos[index] - 20 < 0:
            data_w.append(cir_data_list[index][0 : 35])
        elif max_pos[index] + 15 > len(cir_data_list[index]):
            length = len(cir_data_list[index])
            data_w.append(cir_data_list[index][length - 35 : length])
        else:
            data_w.append(cir_data_list[index][max_pos[index] - 20 : max_pos[index] + 15])

    # energy Er
    data_w_np = np.asarray(data_w)
    data_w_np_power_2 = data_w_np ** 2
    Er = np.nansum(data_w_np, axis=1)

    # mean excess delay T_EMD, T_RMS
    fhi = []
    T_EMD = []
    T_RMS = []
    for index1 in range(len(data_w)):
        fhi.append(data_w_np_power_2[index1] / Er[index1])
        T_EMD.append(0)
        T_RMS.append(0)
        for index2 in range(len(data_w[index1])):
            # T_EMD.append(np.nansum((index2 + 1) * fhi[index1][index2]))
            # T_RMS.append(np.nansum(((index2 + 1 - T_EMD[index1]) ** 2) * fhi[index1][index2]))
            T_EMD[index1] += (index2 + 1) * fhi[index1][index2]
            T_RMS[index1] += ((index2 + 1 - (index2 + 1) * fhi[index1][index2]) ** 2) * fhi[index1][index2]

    # kurtosis Kur
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
        power_4_temp = (np.asarray(data_w[index1]) - mu_np[index1]) ** 4
        Kur.append(np.nansum(power_4_temp) / (len(data_w[index1]) * (sigma[index1] ** 2)))

    feature = list()
    for index1 in range(len(Er)):
        feature.append([Er[index1], T_EMD[index1], T_RMS[index1], Kur[index1], R_T[index1], M_AMP[index1]])

    return np.asarray(feature)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="zenodo",
                        help="dataset for usage, ewine or zenodo")
    parser.add_argument("--dataset_use", type=str, default='regression',
                        help="dataset (ewine) of different usage, including classification and regression")
    parser.add_argument("--dataset_env", type=str, default='room_full',
                        help="dataset (zenodo) of different environments, including rooms and obstacles")
    opt = parser.parse_args()

    if opt.dataset_name == 'ewine':
        if opt.dataset_use == "classification":
            # import raw data from file
            print("Importing dataset from classification dataset.")
            data_cls = load_data_from_folder('./data/data_ewine/dataset_cls/')  # the classification dataset path
            print("Number of samples in dataset: %d" % len(data_cls))  # 42000
            print("Length of one sample: %d" % len(data_cls[0]))  # 1031, matrix (42000, 1031)

            # divide CIR by RX preamble count (get CIR of single preamble pulse)
            # item[2] represents number of acquired preamble symbols
            for item in data_cls:
                item[15:] = item[15:] / float(item[2])

            print("Unlabeled dataset: ", data_cls)
            
            # import extracted label and cir
            print("Loading data for classification.")
            label_cls, cir_cls = load_cls_data('./data/data_ewine/dataset_cls/')
            print("Shape of label: ", label_cls.shape)  # (41993, 2)
            print("Shape of cir sample: ", cir_cls.shape)  # (41993, 152)
            plt.title("Illustration of CIR waveform of the dataset %s for %s." % (opt.dataset_name, opt.dataset_use))
            plt.xlabel("Time Interval")
            plt.ylabel("CIR")
            plt.plot(cir_cls[0], color='blue')
            plt.plot(cir_cls[1], color='red')
            plt.legend(["sample 0", "sample 1"])
            plt.show()

        elif opt.dataset_use == 'regression':
            # 1) import raw data from file (numpy array)
            print("Importing dataset from regression dataset.")
            data1 = load_data_from_file('./data/data_ewine/dataset_reg1/tag_room0.csv')  # set1-room0
            print("shape0: %d" % data1.shape[0])  # 4801, sample number
            print("shape1: %d" % data1.shape[1])  # 1040, sample length
            data2 = load_data_from_file('./data/data_ewine/dataset_reg1/tag_room1.csv')  # set1-room1
            print("shape0: %d" % data2.shape[0])  # 5099
            print("shape1: %d" % data2.shape[1])  # 1040
            data3 = load_data_from_file('./data/data_ewine/dataset_reg2/tag_room0.csv')  # set2-room0
            print("shape0: %d" % data3.shape[0])  # 3499
            print("shape1: %d" % data3.shape[1])  # 1040
            data4 = load_data_from_folder('./data/data_ewine/dataset_reg2/tag_room1/')  # set2-room1
            print("shape0: %d" % data4.shape[0])  # 21589
            print("shape1: %d" % data4.shape[1])  # 1040
            # note that data1-4 are different datasets

            # divide CIR by RX preamble count (get CIR of single preamble pulse)
            # item[17] represents number of acquired preamble symbols
            for item in data1:
                item[24:] = item[24:] / float(item[17])

            print("Labeled dataset: ", data1)

            # 2) import extracted err and cir
            print("Loading data for regression.")
            filepaths = ['./data/data_ewine/dataset_reg1/tag_room0.csv',
                         './data/data_ewine/dataset_reg1/tag_room1.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part0.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part1.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part2.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part3.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part4.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part5.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part6.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part7.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part8.csv',
                         './data/data_ewine/dataset_reg2/tag_room1/tag_room1_part9.csv']
            err_reg, cir_reg = load_reg_data(filepaths)
            print("Shape of error: ", err_reg.shape)  # (31489, 1)
            print("Shape of cir sample: ", cir_reg.shape)  # (31489, 152)
            plt.title("Illustration of CIR waveform of the dataset %s for %s." % (opt.dataset_name, opt.dataset_use))
            plt.xlabel("Time Interval")
            plt.ylabel("CIR")
            plt.plot(cir_reg[0], color='blue')
            plt.plot(cir_reg[1], color='red')
            plt.legend(["sample 0", "sample 1"])
            plt.show()

            # 3) import extracted features
            Er, T_EMD, T_RMS, Kur, R_T, M_AMP = feature_extraction(cir_reg)
            print("Shape of Energy: ", Er.shape)
            print("Energy value: ", Er[0])


    elif opt.dataset_name == 'zenodo':
        # 1) import extracted err and cir
        print("Loading data for regression.")
        err_reg, cir_reg, label_reg = load_pkl_data('data/data_zenodo/dataset.pkl', option=opt.dataset_env)  # option=opt.dataset_env
        print("Shape of error: ", err_reg.shape)  # (55158, 1)
        # print("Shape of cir sample: ", cir_reg.shape)  # (55158, 157)
        # print("Shape of label: ", label_reg.shape)  # (55158, 1)
        err_reg = err_reg.reshape(err_reg.shape[0])
        unmitigated_error = np.sum(err_reg) / err_reg.shape[0]
        print("Unmitigated Error: abs %f" % unmitigated_error)
        # plt.title("Illustration of CIR waveform of the dataset %s with env %s." % (opt.dataset_name, opt.dataset_env))
        # plt.xlabel("Time Interval")
        # plt.ylabel("CIR")
        # plt.plot(cir_reg[0], color='blue')
        # plt.plot(cir_reg[1], color='red')
        # plt.legend(["sample 0", "sample 1"])
        # plt.show()
        # plt.close()

        # plt.title("Illustration of CIR waveform of the dataset %s with env %s." % (opt.dataset_name, opt.dataset_env))
        # plt.xlabel("Time Interval")
        # plt.ylabel("CIR")
        plt.plot(cir_reg[0], color='blue')
        # d_GT = cir_reg[0].max()
        # d_M = cir_reg[0].max()
        # plt.legend(["d_GT %d, d_M %d, err: %d" % d_GT, d_M, err_reg[0]])
        # plt.show()

        # 2) import extracted features
        # features = feature_extraction(cir_reg)
        
        # match digit label to string label
        if opt.dataset_env == 'room_full':
            label_str_dict = {
                0: 'cross-room', 1: 'big room', 2: 'medium room', 3: 'small room', 4: 'outdoor'
            }
        elif opt.dataset_env == 'obstacle_full':
            label_str_dict = {
                1: 'metal window', 2: 'glass plate', 3: 'wood door', 4: 'metal plate', 5: 'LCD TV',
                6: 'cardboard box', 7: 'plywood plate', 8: 'plastic', 9: 'polystyrene plate', 10: 'wall'
            }
        elif opt.dataset_env == 'room_part':
            label_str_dict = {
                1: 'big room', 2: 'medium room', 3: 'small room'
            }
        elif opt.dataset_env == 'room_full_rough':
            label_str_dict = {
                1: 'cross-room', 2: 'outdoor', 3: 'rooms'
            }
        elif opt.dataset_env == 'obstacle_part':
            label_str_dict = {
                1: 'metal', 2: 'wood', 3: 'plastic', 4: 'glass'
            }
        elif opt.dataset_env == 'obstacle_part2':
            label_str_dict = {
                1: 'heavy', 2: 'light'
            }
        elif opt.dataset_env == 'room_full_rough2':
            label_str_dict = {
                1: 'indoor', 2: 'outdoor'
            }
        elif opt.dataset_env == 'paper':
            label_str_dict = {
                1: 'cross-room', 2: 'big room', 3: 'medium room', 4: 'small room'
            }
        label = label_reg[0][0]
        print(label)
        label = label_str_dict[label]
        print(label)
        os.makedirs("saved_results", exist_ok=True)
        plt.savefig("saved_results/sample_%s.png" % label)
        plt.close()
        
