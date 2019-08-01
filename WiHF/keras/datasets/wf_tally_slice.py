from __future__ import print_function

import numpy as np
import os
import sys
import scipy.io as scio
from random import shuffle


def get_wf_tally_slice_data(matlab_data, dataset_root, dataset_path,
                            domain_type, target_type, train_test_ratio,
                            receiver_list, pca_list, user_list, gesture_list,
                            position_list, orientation_list, instance_list,
                            time_limit):
    n_classes_gesture = len(gesture_list)
    n_classes_user = len(user_list)
    n_classes_position = len(position_list)
    n_classes_orientation = len(orientation_list)
    n_train_test_ratio = train_test_ratio  # Test set size

    # -------------------------------------------------------
    # Parameters Fixed

    #############################################Load all VS
    print('Loading Data...')
    all_data_ori = []
    all_data_ori_aux = []
    all_label_ori_user = []
    all_label_ori_gesture = []
    all_label_ori_orientation = []
    all_label_ori_position = []
    domain_list = [user_list, [], position_list, orientation_list]
    # Load Train Set
    root_path = dataset_root
    train_dir = dataset_path

    data_path = os.path.join(root_path, train_dir)
    for root, dirs, files in os.walk(data_path):
        print("domain_type:", domain_type)
        train_file_length = int(len(files) * n_train_test_ratio)
        if (domain_type == 4):
            shuffle(files)
        else:
            files.sort(key=lambda x: domain_list[domain_type].index(
                int(x.split('-')[domain_type])))
            shuffle_files = np.array(files)
            shuffle(shuffle_files[0:train_file_length])
            files = shuffle_files.tolist()
        for file in files:
            file_path = os.path.join(root, file)
            try:
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
                train_label_single_user = user_list.index(
                    n_target_type_label[0]) + 1
                train_label_single_gesture = gesture_list.index(
                    n_target_type_label[1]) + 1
                train_label_single_position = position_list.index(
                    n_target_type_label[2]) + 1
                train_label_single_orientation = orientation_list.index(
                    n_target_type_label[3]) + 1

                # Restriction for number of receivers
                train_data_single = scio.loadmat(file_path)[
                    matlab_data[0]].tolist()

                # train_data_single_aux = scio.loadmat(file_path)[
                #     matlab_data[1]].tolist()

                train_data_single_arr = np.array(train_data_single)
                if (train_data_single_arr.shape[-1] < time_limit):
                    print("timestamp error:", train_data_single_arr.shape[-1])
                    continue

                train_data_single_aux = (
                    train_data_single_arr[receiver_list][:, pca_list, :, (
                        time_limit // 2):time_limit]).tolist()

                train_data_single = (
                    train_data_single_arr[receiver_list][:, pca_list, :, 0:(
                        time_limit // 2)]).tolist()
                # Normalization only for each frame with power

                # train_data_single_max = np.array([
                #     train_data_single_arr[:, :, :, key].max()
                #     for key in range(train_data_single_arr.shape[-1])
                # ])
                # train_data_single_min = np.array([
                #     train_data_single_arr[:, :, :, key].min()
                #     for key in range(train_data_single_arr.shape[-1])
                # ])

                # train_data_single_max_rep = np.tile(train_data_single_max,(train_data_single_arr.shape[0],\
                #             train_data_single_arr.shape[1],train_data_single_arr.shape[2],1))
                # train_data_single_min_rep = np.tile(train_data_single_min,(train_data_single_arr.shape[0],\
                #             train_data_single_arr.shape[1],train_data_single_arr.shape[2],1))
                # train_data_single_arr_nor = (train_data_single_arr - \
                #             train_data_single_min_rep)/(train_data_single_max_rep - train_data_single_min_rep)

                # Save List
                # all_data_ori.append(train_data_single_arr_nor.tolist())

                all_data_ori.append(train_data_single)
                all_data_ori_aux.append(train_data_single_aux)
                all_label_ori_user.append(train_label_single_user)
                all_label_ori_gesture.append(train_label_single_gesture)
                all_label_ori_position.append(train_label_single_position)
                all_label_ori_orientation.append(
                    train_label_single_orientation)

            except ValueError as identifier:
                print("ValueError: ", file, identifier)
            except IndexError as identifier:
                print("IndexError: ", file, identifier)
    try:
        train_sample_length = int(len(all_data_ori) * n_train_test_ratio)
        sample_length = len(all_data_ori)

        train_data = np.array(all_data_ori[0:train_sample_length])
        train_data_aux = np.array(all_data_ori_aux[0:train_sample_length])
        train_label_all_user = np.array(
            all_label_ori_user[0:train_sample_length])
        train_label_all_gesture = np.array(
            all_label_ori_gesture[0:train_sample_length])
        train_label_all_position = np.array(
            all_label_ori_position[0:train_sample_length])
        train_label_all_orientation = np.array(
            all_label_ori_orientation[0:train_sample_length])

        test_data = np.array(all_data_ori[train_sample_length:sample_length])
        test_data_aux = np.array(
            all_data_ori_aux[train_sample_length:sample_length])
        test_label_all_user = np.array(
            all_label_ori_user[train_sample_length:sample_length])
        test_label_all_gesture = np.array(
            all_label_ori_gesture[train_sample_length:sample_length])
        test_label_all_position = np.array(
            all_label_ori_position[train_sample_length:sample_length])
        test_label_all_orientation = np.array(
            all_label_ori_orientation[train_sample_length:sample_length])
        # One-hot Encoding
        train_label_user = np.eye(n_classes_user)[train_label_all_user - 1]
        train_label_gesture = np.eye(n_classes_gesture)[train_label_all_gesture
                                                        - 1]
        train_label_position = np.eye(n_classes_position)[
            train_label_all_position - 1]
        train_label_orientation = np.eye(n_classes_orientation)[
            train_label_all_orientation - 1]

        test_label_user = np.eye(n_classes_user)[test_label_all_user - 1]
        test_label_gesture = np.eye(n_classes_gesture)[test_label_all_gesture -
                                                       1]
        test_label_position = np.eye(n_classes_position)[
            test_label_all_position - 1]
        test_label_orientation = np.eye(n_classes_orientation)[
            test_label_all_orientation - 1]

        # Swap Axes
        # channel=receiver
        # train_data = np.reshape(train_data,
        #                         (train_data.shape[0], -1, 20, n_timesteps))
        train_data = np.swapaxes(train_data, 1, -1)
        train_data = np.swapaxes(train_data, 2, -1)
        # train_data = np.expand_dims(train_data, axis=-1)

        # test_data = np.reshape(test_data,
        #                        (test_data.shape[0], -1, 20, n_timesteps))
        test_data = np.swapaxes(test_data, 1, -1)
        test_data = np.swapaxes(test_data, 2, -1)
        # test_data = np.expand_dims(test_data, axis=-1)

        # train_data_aux= np.reshape(
        #     train_data_aux, (train_data_aux.shape[0], -1, 20, n_timesteps))
        train_data_aux = np.swapaxes(train_data_aux, 1, -1)
        train_data_aux = np.swapaxes(train_data_aux, 2, -1)
        # train_data_aux= np.expand_dims(train_data_aux, axis=-1)

        # test_data_aux= np.reshape(
        #     test_data_aux, (test_data_aux.shape[0], -1, 20, n_timesteps))
        test_data_aux = np.swapaxes(test_data_aux, 1, -1)
        test_data_aux = np.swapaxes(test_data_aux, 2, -1)
        # test_data_aux= np.expand_dims(test_data_aux, axis=-1)
    except ValueError as identifier:
        print("error", identifier)
    return [[train_data, train_data_aux],
            [
                train_label_user, train_label_gesture, train_label_position,
                train_label_orientation
            ], [test_data, test_data_aux],
            [
                test_label_user, test_label_gesture, test_label_position,
                test_label_orientation
            ],
            [
                n_classes_user, n_classes_gesture, n_classes_position,
                n_classes_orientation
            ],
            int(time_limit // 2)]
