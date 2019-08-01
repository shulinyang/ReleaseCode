from __future__ import print_function

import numpy as np
import os
import sys
import scipy.io as scio
from random import shuffle


#domain_type:   0:user;   1:gesture;    2:location;   3:orientation;    4:cross_domain
#target_type:   0:user;   1:gesture;    2:location;   3:orientation
def get_bvp_data(matlab_data, dataset_root, dataset_path, domain_type,
                 target_type, train_test_ratio, user_list, gesture_list,
                 env_list, position_list, orientation_list, instance_list):
    n_target_type = [len(user_list), len(gesture_list), 5, 5]
    n_train_test_ratio = train_test_ratio  # Test set size

    # -------------------------------------------------------
    # Parameters Fixed
    n_timesteps = 0  # Time series

    #############################################Load all VS
    print('Loading Data...')
    train_data_ori = []
    train_label_ori = []
    test_data_ori = []
    test_label_ori = []

    domain_list = [env_list, [], position_list, orientation_list]
    # Load Train Set
    root_path = dataset_root
    train_dir = dataset_path
    data_path = os.path.join(root_path, train_dir)
    all_data_ori = []
    all_label_ori = []

    for root, dirs, files in os.walk(data_path):
        print("domain_type:", domain_type)
        train_file_length = int(len(files) * n_train_test_ratio)
        if (domain_type == 4):
            shuffle(files)
        else:
            if (domain_type == 0):
                files.sort()
            else:
                files.sort(key=lambda x: domain_list[domain_type].index(
                    int(x.split('-')[domain_type])))
            shuffle_files = np.array(files)
            shuffle(shuffle_files[0:train_file_length])
            files = shuffle_files.tolist()
        for file in files:
            file_path = os.path.join(root, file)
            try:
                train_data_single = scio.loadmat(
                    file_path)['velocity_spectrum_ro'].tolist()
                n_target_type_label = [
                    int(file.split('-')[0]),
                    int(file.split('-')[1]),
                    int(file.split('-')[2]),
                    int(file.split('-')[3]),
                    int(file.split('-')[4])
                ]
                if ((n_target_type_label[0] not in user_list)
                        or (n_target_type_label[1] not in gesture_list)
                        or (n_target_type_label[4] not in instance_list)):
                    continue

                if (target_type == 0):
                    train_label_single = user_list.index(
                        n_target_type_label[0]) + 1
                else:
                    train_label_single = gesture_list.index(
                        n_target_type_label[1]) + 1
                n_classes = n_target_type[target_type]  # Gesture Classes
                # Normalization
                train_data_single_arr = np.array(train_data_single)
                train_data_single_max = np.concatenate((train_data_single_arr.max(axis=0),\
                    train_data_single_arr.max(axis=1)),axis=0).max(axis=0)
                train_data_single_min = np.concatenate((train_data_single_arr.min(axis=0),\
                    train_data_single_arr.min(axis=1)),axis=0).min(axis=0)
                if (len(
                        np.where((train_data_single_max -
                                  train_data_single_min) == 0)[0]) > 0):
                    continue
                train_data_single_max_rep = np.tile(train_data_single_max,(train_data_single_arr.shape[0],\
                    train_data_single_arr.shape[1],1))
                train_data_single_min_rep = np.tile(train_data_single_min,(train_data_single_arr.shape[0],\
                    train_data_single_arr.shape[1],1))
                train_data_single_arr_nor = (train_data_single_arr - \
                    train_data_single_min_rep)/(train_data_single_max_rep - train_data_single_min_rep)
                # Save List
                all_data_ori.append(train_data_single_arr_nor.tolist())
                all_label_ori.append(train_label_single)
                if n_timesteps < np.array(train_data_single).shape[2]:
                    n_timesteps = np.array(train_data_single).shape[2]
            except IndexError as identifier:
                print("error", file, identifier)
    train_data_ori.extend(
        all_data_ori[0:int(len(all_data_ori) * n_train_test_ratio)])
    train_label_ori.extend(
        all_label_ori[0:int(len(all_label_ori) * n_train_test_ratio)])

    test_data_ori.extend(
        all_data_ori[int(len(all_data_ori) *
                         n_train_test_ratio):len(all_data_ori)])
    test_label_ori.extend(
        all_label_ori[int(len(all_label_ori) *
                          n_train_test_ratio):len(all_label_ori)])

    # Zero-padding
    train_data_pad = []
    train_label_pad = train_label_ori
    for i in range(len(train_data_ori)):
        t = np.array(train_data_ori[i]).shape[2]
        train_data_pad.append(np.pad(train_data_ori[i] \
            , ((0,0),(0,0),(n_timesteps - t,0)), 'constant', constant_values = 0).tolist())
    test_data_pad = []
    test_label_pad = test_label_ori
    for i in range(len(test_data_ori)):
        t = np.array(test_data_ori[i]).shape[2]
        test_data_pad.append(np.pad(test_data_ori[i] \
            , ((0,0),(0,0),(n_timesteps - t,0)), 'constant', constant_values = 0).tolist())

    # Convert to Ndarray
    train_data_all = np.array(train_data_pad)
    train_label_all = np.array(train_label_pad)
    test_data_all = np.array(test_data_pad)
    test_label_all = np.array(test_label_pad)

    # One-hot Encoding
    train_label_all = np.eye(n_classes)[train_label_all - 1]
    test_label_all = np.eye(n_classes)[test_label_all - 1]
    # print("label:",train_label_all.shape,test_label_all.shape)

    # Specify Train/Test Set
    train_data = train_data_all
    train_label = train_label_all
    test_data = test_data_all
    test_label = test_label_all

    # Swap Axes
    train_data = np.swapaxes(np.swapaxes(train_data, 1, 3), 2, 3)
    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.swapaxes(np.swapaxes(test_data, 1, 3), 2, 3)
    test_data = np.expand_dims(test_data, axis=-1)

    return [[train_data], [train_label], [test_data], [test_label],
            [n_classes], n_timesteps]
