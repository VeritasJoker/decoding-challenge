import argparse
import os
import json
import glob


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

    if args.grid == "6v":
        args.grids = [1, 2]
    elif args.grid == "44":
        args.grids = [3, 4]
    else:
        assert args.grid.isdigit()
        args.grids = int(args.grid)

    args.features = args.feature.split("-")

    return args


def write_model_config(dictionary):
    """Write configuration to a file
    Args:
        CONFIG (dict): configuration
    """
    json_object = json.dumps(dictionary, sort_keys=True, indent=4)

    config_file = os.path.join("models", f"{dictionary['saving_dir']}_config.json")
    with open(config_file, "w") as outfile:
        outfile.write(json_object)
