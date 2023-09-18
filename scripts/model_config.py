import argparse
import os
import json
import glob
import pandas as pd
import torch


def parse_arguments():
    """Read commandline arguments
    Returns:
        Namespace: input as well as default arguments
    """

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument("--grid", type=str, required=True)
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--model-size", type=str, required=True)
    parser.add_argument("--saving-dir", type=str, required=True)
    parser.add_argument("--freeze-decoder", action="store_true")

    args = parser.parse_args()

    # Set parameters
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.max_neural_len = 919  # length of max neural signal
    # args.max_source_positions = 919  # length of encoder hidden states
    args.max_source_positions = 460  # length of encoder hidden states
    args.grid_elec_num = 64  # num of elec per grid

    # load electrode grid
    if args.grid == "all":
        args.grids = [1, 2, 3, 4]
    elif args.grid == "6v":
        args.grids = [1, 2]
    elif args.grid == "BA44":
        args.grids = [3, 4]
    else:
        assert args.grid.isdigit()
        args.grids = int(args.grid)

    # for the grids, load electrode idx
    eleclist = []
    for grid in args.grids:
        gridfile = glob.glob(os.path.join("data", "grids", f"grid-{grid}*.txt"))
        with open(gridfile[0]) as f:
            while line := f.readline():
                elec = line.rstrip()
                assert elec.isdigit()
                eleclist.append(int(elec))
    args.eleclist = eleclist

    args.features = args.feature.split("-")
    args.feature_dim = len(args.grids) * len(args.features)
    args.num_mel_bins = args.grid_elec_num * args.feature_dim

    args.saving_dir = os.path.join("models", args.saving_dir)

    return args


def write_model_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)
    config_file = f"{dictionary['saving_dir']}_config.json"
    with open(config_file, "w") as outfile:
        outfile.write(json_object)
