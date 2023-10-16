import os
import glob
import argparse

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat, savemat


def parse_arguments():
    """Read commandline arguments

    Returns:
        args (Namespace): input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--norm", type=str, required=True)
    parser.add_argument("--saving-dir", type=str, required=True)

    args = parser.parse_args()

    args.max_neural_len = 919  # length of max neural signal

    args.features = ["spikePow", "tx1", "tx2", "tx3", "tx4"]
    args.data_types = ["train", "test", "competitionHoldOut"]

    args.mean_dict = {}
    args.std_dict = {}

    return args


def make_dir(path):
    """Make directory if not exists

    Arguments:
        path (string): directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return


def norm_data_elec_block(args, filename):
    """z-scoring all neural features for each electrode in the block

    Arguments:
        args (Namespace): commandline arguments
        filename (string): matfile full path
    """
    mat_signal = loadmat(filename)

    for feature in args.features:
        neurals = mat_signal[feature]

        neural_stack = np.vstack((neurals[0, :]))
        means = neural_stack.mean(axis=0)
        stds = neural_stack.std(axis=0)

        if "/train/" in filename:  # save mean and std dict
            print(f"\tTrain {feature}")
            args.mean_dict[feature] = means
            args.std_dict[feature] = stds
            zscored = stats.zscore(neural_stack, axis=0)
        else:  # zscore based on saved mean and std dict
            print(f"\tTest/Comp {feature}")

            np.seterr(divide="ignore", invalid="ignore")
            zscored = (neural_stack - args.mean_dict[feature]) / args.std_dict[feature]

            if np.isnan(zscored).sum() > 0:
                print(f"\t\tFixing nans in true divide")
                zscored = np.nan_to_num(zscored, nan=0)
                assert np.isnan(zscored).sum() == 0

        for neural_idx, neural in enumerate(neurals[0, :]):  # reassign zscored values
            neurals[0, neural_idx] = zscored[0 : neural.shape[0], :]
            zscored = zscored[neural.shape[0] :, :]

        # reassign zscored values to mat_signal
        mat_signal[feature] = neurals

    newfilename = filename.replace("/competitionData/", f"/{args.saving_dir}/")
    savemat(newfilename, mat_signal)

    return


def norm_data_block(args, filename):
    """z-scoring all neural features for all electrode in the block

    Arguments:
        args (Namespace): commandline arguments
        filename (string): matfile full path
    """
    mat_signal = loadmat(filename)

    for feature in args.features:
        neurals = mat_signal[feature]

        neural_stack = np.vstack((neurals[0, :]))
        means = neural_stack.mean()
        stds = neural_stack.std()

        if "/train/" in filename:  # save mean and std dict
            print(f"\tTrain {feature}")
            args.mean_dict[feature] = means
            args.std_dict[feature] = stds
            zscored = stats.zscore(neural_stack, axis=None)
        else:  # zscore based on saved mean and std dict
            print(f"\tTest/Comp {feature}")

            np.seterr(divide="ignore", invalid="ignore")
            zscored = (neural_stack - args.mean_dict[feature]) / args.std_dict[feature]

            if np.isnan(zscored).sum() > 0:
                print(f"\t\tFixing nans in true divide")
                zscored = np.nan_to_num(zscored, nan=0)
                assert np.isnan(zscored).sum() == 0

        for neural_idx, neural in enumerate(neurals[0, :]):  # reassign zscored values
            neurals[0, neural_idx] = zscored[0 : neural.shape[0], :]
            zscored = zscored[neural.shape[0] :, :]

        # reassign zscored values to mat_signal
        mat_signal[feature] = neurals

    newfilename = filename.replace("/competitionData/", f"/{args.saving_dir}/")
    savemat(newfilename, mat_signal)

    return


def norm_data(args):
    """Main function to normalize data

    Arguments:
        args (Namespace): commandline arguments
    """
    # making directories
    original_data_dir = os.path.join("data", "competitionData")
    saving_data_dir = os.path.join("data", args.saving_dir)
    make_dir(saving_data_dir)

    for data_type in args.data_types:
        make_dir(os.path.join(saving_data_dir, data_type))

    # normalizing data
    norm_func = norm_data_block

    files = glob.glob(os.path.join(original_data_dir, args.data_types[0], "t12*.mat"))
    for file in files:  # loop through blocks
        print(os.path.basename(file))
        norm_func(args, file)

        test_file = file.replace(f"/{args.data_types[0]}/", f"/{args.data_types[1]}/")
        norm_func(args, test_file)

        comp_file = file.replace(f"/{args.data_types[0]}/", f"/{args.data_types[2]}/")
        if os.path.exists(comp_file):  # only have for some blocks
            norm_func(args, comp_file)

    return


def main():
    # Read command line arguments
    args = parse_arguments()

    # Normalize data
    norm_data(args)
    return


if __name__ == "__main__":
    main()
