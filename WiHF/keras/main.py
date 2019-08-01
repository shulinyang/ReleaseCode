"""Main script for project."""
from __future__ import print_function

from models.WF_TALLY_cross import WF_TALLY_cross
from models.CNN_GRU_bvp import CNN_GRU_bvp
from models.CNN_GRU_wf_tally import CNN_GRU_wf_tally


from utils import get_data_loader
from utils import plot_train_figures
import numpy as np
import os
import sys
import argparse

from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import datetime
n_batch_size = 32
n_epochs = 1000
V_bins = 20
n_iteration = 1

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description=
        "WF_Tally for gesture recognition and human identification.",
        epilog="Contact me if you have any question!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-user",
                        '--userlist',
                        nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8],
                        type=int)
    parser.add_argument("-gesture",
                        '--gesturelist',
                        nargs='+',
                        default=[1, 2, 3, 4, 5, 6],
                        type=int)
    parser.add_argument("-instance",
                        '--instancelist',
                        nargs='+',
                        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        type=int)
    parser.add_argument("-receiver",
                        '--receiverlist',
                        nargs='+',
                        default=[0, 1, 2, 3, 4, 5],
                        type=int)
    parser.add_argument("-pca",
                        '--pcalist',
                        nargs='+',
                        default=[0, 1, 2],
                        type=int)

    parser.add_argument("-m",
                        "--mode",
                        default="train",
                        const="train",
                        nargs="?",
                        choices=["train", "evaluate", "demo"],
                        help="process mode (default: %(default)s)")
    parser.add_argument(
        "-ds",
        "--datasource",
        default="wf_tally",
        const="wf_tally",
        nargs="?",
        choices=["bvp", "wf_tally", "wf_tally_slice"],
        help="data_source (default: %(default)s)")
    parser.add_argument(
        "-nt",
        "--nettype",
        default="CNN_GRU_wf_tally",
        const="CNN_GRU_wf_tally",
        nargs="?",
        choices=["CNN_GRU_bvp", "CNN_GRU_wf_tally", "WF_TALLY_cross"],
        help="network_type (default: %(default)s)")
    parser.add_argument(
        "-dt",
        "--domaintype",
        default=4,
        nargs="?",
        type=int,
        choices=[0, 2, 3, 4],
        help=
        "domain_type (default: %(default)s),  0:env; 2:location; 3:orientation; 4:in_domain"
    )
    parser.add_argument(
        "-tt",
        "--targettype",
        default=1,
        nargs="?",
        type=int,
        choices=[0, 1],
        help="target_type (default: %(default)s), 0:user; 1:gesture")
    parser.add_argument("-pt",
                        "--pathtype",
                        required=True,
                        nargs="?",
                        choices=[
                            "HuFuM", "HuFuE",
                            "HuFu", "WIDAR3"
                        ],
                        help="path_type (default: %(default)s)")
    parser.add_argument("-ttr",
                        "--traintestratio",
                        default=0.8,
                        type=float,
                        help="train_test_ratio (default: %(default)s)")
    parser.add_argument("-ss",
                        "--serverset",
                        required=True,
                        choices=["local", "remote"],
                        help="server_set (default: %(default)s)")
    parser.add_argument("-npath",
                        "--numberofpath",
                        default=5,
                        type=int,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30],
                        help="number_of_path (default: %(default)s)")
    parser.add_argument("-gpu",
                        "--gpuloadratio",
                        default=0.2,
                        type=float,
                        help="gpu_load_ratio (default: %(default)s)")
    parser.add_argument("-grr",
                        "--gradientreversalrate",
                        default=0.0,
                        type=float,
                        help="gradient_reversal_rate (default: %(default)s)")
    parser.add_argument("-sf",
                        "--sizeoffeature",
                        default=64,
                        type=int,
                        choices=[16, 32, 64, 128, 256],
                        help="size_of_feature (default: %(default)s)")
    parser.add_argument("-tl",
                        "--timelimit",
                        default=58,
                        type=int,
                        help="time_limit (default: %(default)s)")
    parser.add_argument("-yal",
                        "--yaxislength",
                        default=20,
                        type=int,
                        help="y_axis_length (default: %(default)s)")
    parser.add_argument("-pa",
                        "--patience",
                        default=10,
                        type=int,
                        help="patience (default: %(default)s)")
    parser.add_argument("-mn",
                        "--matlabname",
                        nargs="+",
                        type=str,
                        default=["velocity_bins_freq"],
                        help="matlab_name (default: %(default)s)")
    parser.add_argument("-pd",
                        "--positiondomain",
                        default=4,
                        type=int,
                        help="position_domain (default: %(default)s)")
    parser.add_argument("-od",
                        "--orientationdomain",
                        default=4,
                        type=int,
                        help="orientation_domain (default: %(default)s)")
    parser.add_argument("-ed",
                        "--envdomain",
                        default=1,
                        type=int,
                        choices=[1, 2, 3],
                        help="env_domain (default: %(default)s)")
    return parser


if __name__ == "__main__":
    # parse arguments
    parser = parse_args()
    args = parser.parse_args()
    print('args：', args)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpuloadratio
    set_session(tf.Session(config=config))
    numberofreceiver = len(args.receiverlist)
    numberofpca = len(args.pcalist)
    n_timesteps = args.timelimit
    n_classes = [9, 9, 5, 5]

    evaluation_prefix = './evaluation/' + args.datasource + '_' + args.nettype + '_' + args.pathtype + '_nlink_' + str(
        numberofreceiver) + '_npca_' + str(numberofpca) + '_domain_' + str(
            args.domaintype) + '_target_' + str(
                args.targettype) + '_ttr_' + str(
                    args.traintestratio) + '_' + str(
                        datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + '_'

    n_patience = args.patience
    # select_data_source
    if args.datasource == "bvp":
        if args.serverset == "remote":
            root_path = '/srv/node/sdb1/widar_3/wf_tally/'
        else:
            root_path = 'F:\widar_3\BVP/20181130-VS/'
        train_dir = args.pathtype
        n_features_row = V_bins
        n_features_column = V_bins
        n_features_channel = 1
    elif args.datasource == "wf_tally" or args.datasource == "wf_tally_slice":
        if args.serverset == "remote":
            root_path = '/srv/node/sdb1/widar_3/wf_tally/'
        else:
            root_path = 'F:\widar_3/result/'
        n_features_column = args.yaxislength
        train_dir = args.pathtype
        # n_features_row = numberofpca
        # n_features_channel = numberofreceiver
        n_features_row = numberofreceiver
        n_features_channel = numberofpca
    else:
        parser.print_help()
        sys.exit(1)
    positionlist = [1, 2, 3, 4, 5]
    orientationlist = [1, 2, 3, 4, 5]
    positionlist[args.positiondomain], positionlist[-1] = positionlist[
        -1], positionlist[args.positiondomain]
    orientationlist[args.orientationdomain], orientationlist[
        -1] = orientationlist[-1], orientationlist[args.orientationdomain]
    if args.envdomain == 2:
        args.userlist.extend([9, 10])
    if args.envdomain == 3:
        args.userlist.extend([11, 12])
    envlist = args.userlist
    print('ed: ', envlist, 'pd: ', positionlist, 'od: ', orientationlist)
    for n_itr in range(n_iteration):
        save_model_path = evaluation_prefix + str(n_itr) + '_model.h5'
        accuracy_log_file_name = evaluation_prefix + str(n_itr) + '_acc.log'
        cm_log_file_name = evaluation_prefix + str(n_itr) + '_cm.log'
        with open(accuracy_log_file_name, 'w') as accuracy_log_file:
            if args.serverset == "remote":
                [
                    train_data, train_label, test_data, test_label, n_classes,
                    n_timesteps
                ] = get_data_loader(
                    name=args.datasource,
                    matlab_data_list=args.matlabname,
                    dataset_root=root_path,
                    dataset_path=train_dir,
                    domain_type=args.domaintype,
                    target_type=args.targettype,
                    train_test_ratio=args.traintestratio,
                    receiver_list=args.receiverlist,
                    pca_list=args.pcalist,
                    user_list=args.userlist,
                    gesture_list=args.gesturelist,
                    env_list=envlist,
                    position_list=positionlist,
                    orientation_list=orientationlist,
                    instance_list=args.instancelist,
                    time_limit=args.timelimit,
                )
                print('Loaded Data',
                      args.datasource, len(train_data), train_data[0].shape,
                      len(train_label), train_label[0].shape, len(test_data),
                      test_data[0].shape)
                print('n_timesteps: ' + str(n_timesteps))
            if args.nettype == "CNN_GRU_bvp":
                # Create architecture with a custom number of features and timesteps.
                cnn_gru_bvp = CNN_GRU_bvp(
                    n_timesteps=n_timesteps,
                    n_features_row=n_features_row,
                    n_features_column=n_features_column,
                    n_features_channel=n_features_channel,
                    n_batch_size=n_batch_size,
                    n_epochs=n_epochs,
                    n_patience=n_patience,
                    n_classes=n_classes[0],
                    save_model_path=save_model_path)
                cnn_gru_bvp.create_architecture()
                model = cnn_gru_bvp.compile_model()
                if args.serverset == "remote":
                    model_history = cnn_gru_bvp.fit_model(
                        train_data[0], train_label[0])
                    [cm, scores
                     ] = cnn_gru_bvp.predict_proba(test_data[0], test_label[0])
            elif args.nettype == "CNN_GRU_wf_tally":
                cnn_gru_wf_tally = CNN_GRU_wf_tally(
                    n_timesteps=n_timesteps,
                    n_features_row=n_features_row,
                    n_features_column=n_features_column,
                    n_features_channel=n_features_channel,
                    n_batch_size=n_batch_size,
                    n_epochs=n_epochs,
                    n_patience=n_patience,
                    n_classes=n_classes[args.targettype],
                    save_model_path=save_model_path)
                cnn_gru_wf_tally.create_architecture()
                model = cnn_gru_wf_tally.compile_model()
                if args.serverset == "remote":
                    model_history = cnn_gru_wf_tally.fit_model(
                        train_data[0], train_label[args.targettype])
                    plot_train_figures(model_history)
                    [cm, scores] = cnn_gru_wf_tally.predict_proba(
                        test_data[0], test_label[args.targettype])
                    # intermediate_layer_output = cnn_gru_wf_tally.get_intermediate_layer_output(
                    #     test_data[0])
                    # print("intermediate_layer_output:",
                    #     np.array(intermediate_layer_output))
            elif args.nettype == "WF_TALLY_cross":
                wf_tally_cross = WF_TALLY_cross(
                    n_timesteps=n_timesteps,
                    n_features_row=n_features_row,
                    n_features_column=n_features_column,
                    n_features_channel=n_features_channel,
                    n_classes_aux=n_classes[1 - args.targettype],
                    n_classes_main=n_classes[args.targettype],
                    n_features=args.sizeoffeature,
                    n_batch_size=n_batch_size,
                    n_epochs=n_epochs,
                    n_patience=n_patience,
                    save_model_path=save_model_path,
                    gradient_reversal_rate=args.gradientreversalrate)
                model = wf_tally_cross.compile_model()
                if args.serverset == "remote":
                    if (os.path.exists(save_model_path)
                            and args.mode == 'test'):
                        model = wf_tally_cross.retrieve_model()
                    else:
                        # Train the model
                        model_history = wf_tally_cross.fit_model(
                            train_data[args.targettype],
                            train_data[1 - args.targettype],
                            train_label[1 - args.targettype],
                            train_label[args.targettype])

                    [cm, scores] = wf_tally_cross.predict_proba(
                        test_data[args.targettype],
                        test_data[1 - args.targettype],
                        test_label[1 - args.targettype],
                        test_label[args.targettype])
            else:
                parser.print_help()
                sys.exit(1)

            print(model.summary())
            if (args.serverset == 'local'):
                plot_model(model,
                           to_file='figures/' + args.nettype + '.png',
                           show_shapes=True,
                           show_layer_names=True)
            else:
                # Save Results to File
                np.savetxt(cm_log_file_name, cm)
                accuracy_log_file.write('Test loss:' + str(scores[0]))
                accuracy_log_file.write('Test accuracy:' + str(scores[1]))
                accuracy_log_file.close()
