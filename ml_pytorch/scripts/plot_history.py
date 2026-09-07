import argparse
from scipy.ndimage import uniform_filter1d
import os
import numpy as np
from glob import glob

import mplhep as hep
from utils_configs.plot.HEPPlotter import HEPPlotter

LUMITEXT = "2022 (13.6 TeV)"

# CMS colour palette of mplhep, used consistently in all the plotting scripts
CMS_COLORS = [cycle["color"] for cycle in hep.style.CMS["axes.prop_cycle"]]


def read_from_txt(file):
    # get accuracy and loss and separate them between training and validation
    with open(file, "r") as f:
        lines = f.readlines()
        train_accuracy = []
        train_loss = []
        val_accuracy = []
        val_loss = []
        lr = []
        i = 0
        j = 0
        for line in lines:
            if "Training batch" in line and not "batch 0." in line:
                train_accuracy.append(float(line.split("accuracy: ")[1].split(" ")[0]))
                train_loss.append(float(line.split("loss: ")[1].split("\n")[0]))
                i += 1
            elif "Validation batch" in line and not "batch 0." in line:
                val_accuracy.append(float(line.split("accuracy: ")[1].split(" ")[0]))
                val_loss.append(float(line.split("loss: ")[1].split("\n")[0]))
                j += 1
                if j > 10000:
                    print(line)
            elif "learning rate" in line:
                lr.append(float(line.split("rate: ")[1].split("\n")[0]))

    print("len train accuracy: ", len(train_accuracy))
    print("len train loss: ", len(train_loss))
    print("len val accuracy: ", len(val_accuracy))
    print("len val loss: ", len(val_loss))
    print("\n")
    print("Training loss:")
    print(train_loss)
    print("Validation loss:")
    print(val_loss)
    return train_accuracy, train_loss, val_accuracy, val_loss, lr


def plot_history(
    train_accuracy,
    train_loss,
    val_accuracy,
    val_loss,
    dir,
    show,
    uniform_filter=10,
    lenght=-1,
    comet_logger=None,
):
    infos_dict = {
        "accuracy": {"train": train_accuracy[:lenght], "val": val_accuracy[:lenght]},
        "loss": {"train": train_loss[:lenght], "val": val_loss[:lenght]},
    }
    line_style = {
        "accuracy": "--",
        "loss": "-",
    }

    # one graph per curve: the smoothed values as a function of the epoch
    series_dict = {}
    for type, info in infos_dict.items():
        print("len info train: ", len(info["train"]))
        print("len info val: ", len(info["val"]))

        if len(info["train"]) == 0:
            continue

        series_dict[f"Training {type}"] = {
            "data": {
                "x": [
                    np.linspace(0, len(info["train"]) / 10, len(info["train"])),
                    None,
                ],
                "y": [uniform_filter1d(info["train"], size=uniform_filter), None],
            },
            "style": {
                "color": CMS_COLORS[0],
                "linestyle": line_style[type],
                "markersize": 0,
            },
        }

        if len(info["val"]) == 0:
            continue

        series_dict[f"Validation {type}"] = {
            "data": {
                "x": [
                    np.linspace(
                        0,
                        (
                            len(info["val"]) / 10
                            if len(info["val"]) / 10 == len(info["train"]) / 10
                            else len(info["train"]) / 10
                        ),
                        len(info["val"]),
                    ),
                    None,
                ],
                "y": [uniform_filter1d(info["val"], size=uniform_filter), None],
            },
            "style": {
                "color": CMS_COLORS[1],
                "linestyle": line_style[type],
                "markersize": 0,
            },
        }

    if not series_dict:
        print("WARNING: no history to plot")
        return

    plotter = (
        HEPPlotter("CMS")
        .set_plot_config(lumitext=LUMITEXT)
        .set_output(f"{dir}/history")
        .set_labels(xlabel="Epoch", ylabel="")
        .set_data(series_dict, plot_type="graph")
        .set_options(
            legend_loc="center right",
            split_legend=False,
            grid=True,
            set_ylim=False,
        )
    )
    if show:
        plotter.show()
    plotter.run()

    if comet_logger:
        comet_logger.log_image(f"{dir}/history.png", name="history")


def plot_lr(lr, main_dir, show, comet_logger=None):
    if len(lr) == 0:
        print("WARNING: no learning rate to plot")
        return

    series_dict = {
        "Learning rate": {
            "data": {
                "x": [np.arange(len(lr)), None],
                "y": [np.asarray(lr, dtype=float), None],
            },
            "style": {"color": CMS_COLORS[0], "linestyle": "-", "markersize": 0},
        }
    }

    plotter = (
        HEPPlotter("CMS")
        .set_plot_config(lumitext=LUMITEXT)
        .set_output(f"{main_dir}/lr")
        .set_labels(xlabel="Epoch", ylabel="Learning rate")
        .set_data(series_dict, plot_type="graph")
        .set_options(y_log=True, legend=False, grid=True, set_ylim=False)
    )
    if show:
        plotter.show()
    plotter.run()

    if comet_logger:
        comet_logger.log_image(f"{main_dir}/lr.png", name="learning_rate")


def main():
    # plot the history for the training and validation losses and accuracies
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-path", type=str, help="path to log file")
    parser.add_argument(
        "-u",
        "--uniform-filter",
        default=10,
        type=int,
        help="size of the uniform filter",
    )
    parser.add_argument(
        "-s",
        "--show",
        default=False,
        action="store_true",
        help="show plots",
    )
    parser.add_argument(
        "-l",
        "--lenght",
        default=-1,
        help="max lenght of the plot",
        type=int,
    )
    args = parser.parse_args()

    # find the file starting with logger in args.input_path using os.listdir
    log_file = args.input_path

    print(log_file)

    train_accuracy, train_loss, val_accuracy, val_loss, lr = read_from_txt(log_file)

    plot_history(
        train_accuracy,
        train_loss,
        val_accuracy,
        val_loss,
        os.path.dirname(args.input_path),
        args.show,
        args.uniform_filter,
        args.lenght,
    )

    plot_lr(lr, os.path.dirname(args.input_path), args.show)


if __name__ == "__main__":
    main()
