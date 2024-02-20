import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from hpo.optimizers.qt_metadataset import QuickTuneMetaDataset

plt.rc('font', family='serif')

def map_to_common_index (common_index, other_index, values):
    new_values = np.zeros_like(common_index).astype(float)
    for i in range(1,len(other_index)):
        new_values[int(other_index[i-1]):int(other_index[i])] = values[i-1]
    new_values[int(other_index[-1]):len(common_index)] = values[-1] 
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

def plot_results(experiment_id, 
                 results_folder = "output",
                 qt_version = "micro",
                 normalize = True,
                 fontsize = 17,
                 linewidth = 3,
                 time_budget = 3600,
                 num_test_runs = 1,
                 font = None):

    if font is None:
        font = {'family': 'serif', 'size': 17}


    results = []
    experiment_path = Path(results_folder) / experiment_id
    results_dirs = [experiment_path / x for x in os.listdir(experiment_path) if not x.endswith(".pt")]

    assert num_test_runs == len(results_dirs), "Not all runs are available"

    datasets = os.listdir(results_dirs[0])
    common_indexes = np.arange(time_budget)
    if normalize:
        metadataset = QuickTuneMetaDataset(aggregate_data=False, version=qt_version)

    for dataset in datasets:
        try:
            temp_results = aggregate_results(common_indexes, results_dirs, dataset)
            if normalize:
                dataset_parts = dataset.split("_")
                adapted_dataset_name = "/".join(dataset_parts[:3]) + "/" + "_".join(dataset_parts[3:])
                metadataset.set_dataset_name(adapted_dataset_name)
                best_performance = metadataset.get_best_performance() / 100
                worst_performance = metadataset.get_worst_performance() / 100
                temp_results = ((best_performance - np.array(temp_results)) / (
                            best_performance - worst_performance)).tolist()
            else:
                temp_results = 1 - temp_results
                temp_results = temp_results.tolist()
            results.append(temp_results)

        except Exception as e:
            print(dataset, e)

    # plot confidence interval
    results = np.array(results)
    results_avg = results.mean(0)
    results_std = results.std(0)
    scale = 1.96 / np.sqrt(len(datasets)*num_test_runs)
    results_std = results_std * scale

    plt.plot(common_indexes, results_avg, label="Quick-Tune", linewidth=linewidth)
    plt.fill_between(common_indexes, results_avg - results_std, results_avg + results_std,
                            alpha=0.2)
 
    anchor = (0.50, -0.22)
    plt.title(qt_version, fontdict=font)
    plt.xlabel("Wallclock Time (s)", fontdict=font)
    #plt.yscale("log")

    plt.tick_params(axis="x", labelsize=fontsize)
    plt.tick_params(axis="y", labelsize=fontsize)

    plt.tight_layout()
    plt.legend(loc='lower center', bbox_to_anchor=anchor, ncol=6, fontsize=fontsize)
    
    figures_path = Path("plots") / experiment_id 
    figures_path.mkdir(exist_ok=True, parents=True)
    plt.savefig(figures_path / "regret.png", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--experiment_id", type=str, default="qt_micro")
    parser.add_argument("--results_folder", type=str, default="output")
    parser.add_argument("--qt_version", type=str, default="micro")
    parser.add_argument("--time_budget", type=int, default=3600)
    parser.add_argument("--num_test_runs", type=int, default=1)

    args = parser.parse_args()


    plot_results(experiment_id = args.experiment_id, 
                 results_folder = args.results_folder,
                 qt_version = args.qt_version,
                 time_budget = args.time_budget,
                 num_test_runs = args.num_test_runs)