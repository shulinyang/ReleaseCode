"""Helpful functions for project."""
import os
import random
import matplotlib.pyplot as plt

from datasets.bvp import get_bvp_data

from datasets.wf_tally import get_wf_tally_data
from datasets.wf_tally_slice import get_wf_tally_slice_data



def get_data_loader(name, matlab_data_list, dataset_root, dataset_path,
                    domain_type, target_type, train_test_ratio, receiver_list,
                    pca_list, user_list, gesture_list, env_list, position_list,
                    orientation_list, instance_list, time_limit):
    """Get data loader by name."""
    if name == "bvp":
        return get_bvp_data(matlab_data_list[0], dataset_root, dataset_path,
                            domain_type, target_type, train_test_ratio,
                            user_list, gesture_list, env_list, position_list,
                            orientation_list, instance_list)
    
    elif name == "wf_tally":
        return get_wf_tally_data(matlab_data_list, dataset_root, dataset_path,
                                 domain_type, target_type, train_test_ratio,
                                 receiver_list, pca_list, user_list,
                                 gesture_list, env_list, position_list,
                                 orientation_list, instance_list, time_limit)
    
    elif name == "wf_tally_slice":
        return get_wf_tally_slice_data(matlab_data_list, dataset_root,
                                       dataset_path, domain_type, target_type,
                                       train_test_ratio, receiver_list,
                                       pca_list, user_list, gesture_list,
                                       position_list, orientation_list,
                                       instance_list, time_limit)

def plot_train_figures(history):
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()