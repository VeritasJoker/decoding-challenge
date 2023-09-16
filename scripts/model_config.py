import argparse
import os
import json
import glob
import pandas as pd


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

    args = parser.parse_args()

    # Set parameters
    args.device = "cpu"
    args.neural_max_len = 919  # length of max neural signal

    # load electrode grid
    if args.grid == "all":
        args.grids = [1, 2, 3, 4]
    elif args.grid == "6v":
        args.grids = [1, 2]
    elif args.grid == "44":
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
