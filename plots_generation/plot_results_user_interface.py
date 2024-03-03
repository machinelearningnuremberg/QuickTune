import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
plt.rc("font", family="serif")


def map_to_common_index(common_index, other_index, values):
    new_values = np.zeros_like(common_index).astype(float)
    for i in range(1, len(other_index)):
        new_values[int(other_index[i - 1]) : int(other_index[i])] = values[i - 1]
    new_values[int(other_index[-1]) : len(common_index)] = values[-1]
    return new_values


def aggregate_results(common_indexes, results_dirs, dataset):
    reshaped_list = []
    for meta_test_id_dir in results_dirs:
        temp_path = meta_test_id_dir / dataset / "results.json"
        with open(temp_path) as f:
            results_json = json.load(f)
        reshaped = map_to_common_index(common_indexes, results_json[1], results_json[2])
        reshaped_list.append(reshaped)

    reshaped_results = np.array(reshaped_list).mean(0)
    return reshaped_results


def plot_results(results_folder, linewidth=3, fontsize=17, font=None):

    if font is None:
        font = {"family": "serif", "size": 17}

    experiment_path = Path("output") / results_folder
    with open(experiment_path / "perf_curves.json") as f:
        results = json.load(f)

    regret = 1 - np.maximum.accumulate(np.array(results["all_perf"]))
    cost = results["all_cost"]

    plt.plot(cost, regret, linewidth=linewidth)
    plt.title("Regret", fontdict=font)
    plt.xlabel("Wallclock Time (s)", fontdict=font)
    plt.yscale("log")
    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)
    plt.tight_layout()

    figures_path = Path("plots") / results_folder
    figures_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_path / "regret.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="qt_metalearned_mini_imagenette/imagenette2-320/",
    )
    args = parser.parse_args()
    plot_results(results_folder=args.results_folder)
