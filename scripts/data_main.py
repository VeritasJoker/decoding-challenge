import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat


def measure_data(groups):
    """Measure and print the longest ecog data length (max_neural_len)

    Args:
        groups ([str]): list of strings containing types of data
    """
    sentence_len_max = 0
    for group in groups:
        files = glob.glob(os.path.join("data", "competitionData", group, "t12*.mat"))
        for file in files:
            print(f"{file}")
            mat_signal = loadmat(file)
            neurals = mat_signal["spikePow"]  # x, spike power (binned 20ms)
            tx1 = mat_signal["tx1"]  # x, threshold crossing counts (-3.5 * RMS)
            for sentence in np.arange(0, neurals.shape[1]):
                neural = neurals[0, sentence]
                assert neural.shape[0] == tx1[0, sentence].shape[0]
                sentence_len = neural.shape[0]
                if sentence_len > sentence_len_max:
                    print(sentence_len)
                    sentence_len_max = sentence_len
    return sentence_len_max


def pad_neural_data(args, neural):
    """Transpose and pad neural signal to max_neural_len

    Arguments:
        args (Namespace): commandline arguments
        neural (np.array): np array of neural signal

    Returns:
        neural (np.array): padded neural signal correctly sorted by electrode grid
    """
    neural = neural.T
    neural = np.pad(neural, [(0, 0), (0, args.max_neural_len - neural.shape[1])])
    return neural[args.eleclist]  # correct grid


def load_data_block(args, filename):
    """Load a block of neural data

    Arguments:
        args (Namespace): commandline arguments
        filename (str): matfile full path

    Returns:
        signals_padded ([np.array]): list of np arrays of neural signal
        sentences (np.array): np array of sentences
    """
    mat_signal = loadmat(filename)
    sentences = mat_signal["sentenceText"]  # y, sentence
    neurals = mat_signal["spikePow"]  # x, spike power (binned 20ms)
    # tx1 = mat_signal["tx1"]  # x, threshold crossing counts (-3.5 * RMS)
    # tx2 = mat_signal["tx2"]  # x, threshold crossing counts (-4.5 * RMS)
    # tx3 = mat_signal["tx3"]  # x, threshold crossing counts (-5.5 * RMS)
    # tx4 = mat_signal["tx4"]  # x, threshold crossing counts (-6.5 * RMS)

    signals_padded = []

    for neural_idx in np.arange(0, neurals.shape[1]):  # for each sentence
        signal = []
        for feature in args.features:  # loop through features
            neural = pad_neural_data(args, mat_signal[feature][0, neural_idx])
            signal.append(neural)
        signal = np.vstack(signal)  # stack features

        signals_padded.append(signal)  # add signal

    return signals_padded, sentences


def load_data(args, group):
    """Load neural data

    Arguments:
        args (Namespace): commandline arguments
        group (str): type of neural data (train/test/competitionHoldOut)

    Returns:
        signals ([np.array]): list of np arrays of neural signal
        sentences ([np.array]): list of np array of sentences
    """
    print(f"\tData {group}")
    files = glob.glob(os.path.join(args.data_dir, group, "t12*.mat"))
    signals = []
    sentences = []
    for file in files:  # loop through blocks
        signals_block, sentences_block = load_data_block(args, file)
        signals.extend(signals_block)
        sentences.extend(sentences_block)

    return signals, sentences


def main():
    measure_data(["train", "test", "competitionHoldOut"])
    # NEURAL_MAX = 919
    # data = load_data("train", NEURAL_MAX)
    return


if __name__ == "__main__":
    main()
