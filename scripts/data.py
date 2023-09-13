import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


def main():
    file = os.path.join("data", "competitionData", "test", "t12.2022.05.05.mat")
    mat_signal = loadmat(file)

    sentence = mat_signal["sentenceText"]  # y, sentence
    neural = mat_signal["spikePow"]  # x, spike power (binned 20ms)
    tx1 = mat_signal["tx1"]  # x, threshold crossing counts (-3.5 * RMS)
    tx2 = mat_signal["tx1"]  # x, threshold crossing counts (-4.5 * RMS)
    tx3 = mat_signal["tx1"]  # x, threshold crossing counts (-5.5 * RMS)
    tx4 = mat_signal["tx1"]  # x, threshold crossing counts (-6.5 * RMS)
    breakpoint()
    return


if __name__ == "__main__":
    main()
