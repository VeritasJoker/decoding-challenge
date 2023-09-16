import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat


def measure_data(groups):
    sentence_len_max = 0
    for group in groups:
        files = glob.glob(os.path.join("data", "competitionData", group, "t12*.mat"))
        for file in files:
            print(f"{file}")
            mat_signal = loadmat(file)
            neural = mat_signal["spikePow"]  # x, spike power (binned 20ms)
            tx1 = mat_signal["tx1"]  # x, threshold crossing counts (-3.5 * RMS)
            for sentence in np.arange(0, neural.shape[1]):
                assert neural[0, sentence].shape[0] == tx1[0, sentence].shape[0]
                sentence_len = neural[0, sentence].shape[0]
                if sentence_len > sentence_len_max:
                    print(sentence_len_max)
                    sentence_len_max = sentence_len
    return sentence_len_max


def load_data(group, neural_max):
    files = glob.glob(os.path.join("data", "competitionData", group, "t12*.mat"))
    for file in files:
        mat_signal = loadmat(file)
        sentences = mat_signal["sentenceText"]  # y, sentence
        neurals = mat_signal["spikePow"]  # x, spike power (binned 20ms)
        tx1 = mat_signal["tx1"]  # x, threshold crossing counts (-3.5 * RMS)
        tx2 = mat_signal["tx2"]  # x, threshold crossing counts (-4.5 * RMS)
        tx3 = mat_signal["tx3"]  # x, threshold crossing counts (-5.5 * RMS)
        tx4 = mat_signal["tx4"]  # x, threshold crossing counts (-6.5 * RMS)

        for neural_idx in np.arange(0, neurals.shape[1]):
            neural = neurals[0, neural_idx].T
            neural = np.pad(neural, [(0, 0), (0, neural_max - neural.shape[1])])
            breakpoint()

    return


def main():
    NEURAL_MAX = 906
    # measure_data(["train", "test", "competitionHoldOut"])
    data = load_data("train", NEURAL_MAX)
    return


if __name__ == "__main__":
    main()
