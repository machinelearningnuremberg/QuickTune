import argparse
import pandas as pd

from hpo.optimizers.qt_metadataset import QuickTuneMetaDataset

# from hpo.optimizers.asha.asha import AHBOptimizer
from sklearn.neighbors import KNeighborsRegressor
import os
import time
import json

import numpy as np
from syne_tune.blackbox_repository import (
    load_blackbox,
    add_surrogate,
    BlackboxRepositoryBackend,
    UserBlackboxBackend,
)
from syne_tune.experiments import load_experiment
from syne_tune.blackbox_repository.blackbox_tabular import BlackboxTabular

# from benchmarking.commons.benchmark_definitions.lcbench import lcbench_benchmark
# from syne_tune.blackbox_repository import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.baselines import (
    BayesianOptimization,
    RandomSearch,
    ASHA,
    BOHB,
    DEHB,
    SyncBOHB,
)
from syne_tune import Tuner, StoppingCriterion
import syne_tune.config_space as sp


def log_info(
    self,
    hp_index: int,
    performance: float,
    budget: int,
    best_value_observed: float,
    time_duration: float,
):
    """Log information after every HPO iteration.

    Args:
        hp_index: int
            The index of the suggested hyperparameter candidate.
        performance: float
            The performance of the hyperparameter candidate.
        budget: int
            The budget at which the hyperpararameter candidate has been evaluated so far.
        best_value_observed: float
            The incumbent value observed so far during the optimization.
        time_duration: float
            The time taken for the HPO iteration.
    """
    if "hp" in self.info_dict:
        self.info_dict["hp"].append(hp_index)
    else:
        self.info_dict["hp"] = [hp_index]

    accuracy_performance = performance

    if "scores" in self.info_dict:
        self.info_dict["scores"].append(accuracy_performance)
    else:
        self.info_dict["scores"] = [accuracy_performance]

    incumbent_acc_performance = best_value_observed

    if "curve" in self.info_dict:
        self.info_dict["curve"].append(incumbent_acc_performance)
    else:
        self.info_dict["curve"] = [incumbent_acc_performance]

    if "epochs" in self.info_dict:
        self.info_dict["epochs"].append(budget)
    else:
        self.info_dict["epochs"] = [budget]

    if "overhead" in self.info_dict:
        self.info_dict["overhead"].append(time_duration)
    else:
        self.info_dict["overhead"] = [time_duration]

    with open(self.result_file, "w") as fp:
        json.dump(self.info_dict, fp)


if __name__ == "__main__":

    aggregate_data = False

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Reproducibility seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mtlbm/mini/set1/PNU",
        help="Dataset name.",
    )
    parser.add_argument(
        "--set_name",
        type=str,
        default="mini",
        help="The set to use.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default="gp",
        help="The algorithm name.",
    )

    args = parser.parse_args()

    def run(args):

        time_extensions = {
            "micro": 1.5,
            "mini": 6,
            "extended": 24,
        }

        method_algorithms = {
            "asha": ASHA,
            "dehb": DEHB,
            "bohb": BOHB,
            "sync-bohb": SyncBOHB,
            "gp": BayesianOptimization,
        }

        seed = args.seed
        set_name = args.set_name
        benchmark = QuickTuneMetaDataset(aggregate_data=aggregate_data, set=set_name)
        dataset_names = benchmark.get_datasets()
        print(dataset_names)
        # print(" ".join(dataset_names))
        # dataset_name = 'mtlbm/micro/set1/PNU'
        # dataset_name = 'mtlbm/micro/set2/ACT_410'
        benchmark.set_dataset_name(args.dataset_name)
        curve_lengths = []
        first_fidelity_values = []
        final_values = []
        for hp_index in range(benchmark.get_hyperparameters_candidates().shape[0]):
            curve = benchmark.get_curve(hp_index)
            curve_lengths.append(len(curve))
            first_fidelity_values.append(curve[0])
            final_values.append(curve[-1])

        max_budget = max(curve_lengths)
        mean_first_fidelity = np.mean(first_fidelity_values)

        hp_candidates = benchmark.get_hyperparameters_candidates()
        hp_names = hp_candidates.columns
        # pandas to numpy
        hp_candidates = hp_candidates.to_numpy()
        first_hp_curve = benchmark.get_curve(0, 52)
        full_hp_candidates = []
        benchmark_hp_index = 0
        hp_indice_map = dict()
        for hp_index in range(hp_candidates.shape[0]):
            if curve_lengths[hp_index] < max_budget:
                if final_values[hp_index] > mean_first_fidelity:
                    full_hp_candidates.append(hp_candidates[hp_index])
                    hp_indice_map[benchmark_hp_index] = hp_index
                    benchmark_hp_index += 1
            else:
                full_hp_candidates.append(hp_candidates[hp_index])
                hp_indice_map[benchmark_hp_index] = hp_index
                benchmark_hp_index += 1

        full_hp_candidates = np.array(full_hp_candidates)

        hp_ranges = []
        for hp_index in range(0, full_hp_candidates.shape[1]):
            min_value = np.min(full_hp_candidates[:, hp_index])
            max_value = np.max(full_hp_candidates[:, hp_index])
            if min_value == max_value:
                if min_value == 0:
                    max_value = 1
                else:
                    min_value = 0
            hp_ranges.append((min_value, max_value))
        config_space = {
            hp_names[hp_index]: sp.uniform(
                hp_ranges[hp_index][0], hp_ranges[hp_index][1]
            )
            for hp_index in range(0, full_hp_candidates.shape[1])
        }
        cs_fidelity = {
            "hp_epochs": sp.randint(0, max_budget),
        }
        print(config_space)
        objectives_evaluations = np.zeros(
            (full_hp_candidates.shape[0], 1, max_budget, 2)
        )

        for hp_index in range(0, full_hp_candidates.shape[0]):
            mapped_hp_index = hp_indice_map[hp_index]
            curve = benchmark.get_curve(mapped_hp_index, max_budget)
            original_curve_budget = len(curve)
            last_step_cost = benchmark.get_step_cost(
                mapped_hp_index, original_curve_budget
            )
            if original_curve_budget < max_budget:
                curve = np.arange(
                    mean_first_fidelity,
                    curve[-1] + (curve[-1] - mean_first_fidelity) / max_budget,
                    (curve[-1] - mean_first_fidelity) / (max_budget - 1),
                )
                curve_cost = np.arange(
                    last_step_cost / max_budget,
                    last_step_cost + last_step_cost / max_budget,
                    (last_step_cost - (last_step_cost / max_budget)) / (max_budget - 1),
                )
            for budget in range(1, len(curve) + 1):
                for objective_index in range(0, 2):
                    if original_curve_budget == max_budget:
                        if objective_index == 0:
                            objective_value = curve[budget - 1]
                        else:
                            objective_value = benchmark.get_step_cost(
                                mapped_hp_index, budget
                            )

                    else:
                        if objective_index == 0:
                            objective_value = curve[budget - 1]
                        else:
                            objective_value = curve_cost[budget - 1]
                    objectives_evaluations[hp_index, 0, budget - 1, objective_index] = (
                        objective_value
                    )

        full_hp_candidates = pd.DataFrame(full_hp_candidates, columns=hp_names)
        benchmark = BlackboxTabular(
            hyperparameters=full_hp_candidates,
            configuration_space=config_space,
            fidelity_space=cs_fidelity,
            objectives_evaluations=objectives_evaluations,
            objectives_names=["accuracy", "runtime"],
        )

        max_resource_attr = "hp_epochs"
        if (
            args.method_name == "dehb"
            or args.method_name == "bohb"
            or args.method_name == "sync-bohb"
        ):
            backend_blackbox = add_surrogate(
                benchmark, surrogate=KNeighborsRegressor(n_neighbors=1)
            )
        else:
            backend_blackbox = benchmark

        trial_backend = UserBlackboxBackend(
            blackbox=backend_blackbox,
            elapsed_time_attr="runtime",
        )
        blackbox = trial_backend.blackbox
        restrict_configurations = benchmark.all_configurations()

        algorithm = method_algorithms[args.method_name]
        scheduler = algorithm(
            config_space=blackbox.configuration_space_with_max_resource_attr(
                max_resource_attr
            ),
            resource_attr="hp_epochs",
            max_resource_attr=max_resource_attr,
            mode="max",
            metric="accuracy",
            random_seed=seed,
            search_options=dict(restrict_configurations=restrict_configurations),
        )

        stop_criterion = StoppingCriterion(
            max_wallclock_time=3600 * time_extensions[set_name]
        )
        # Printing the status during tuning takes a lot of time, and so does
        # storing results.
        print_update_interval = 700
        results_update_interval = 300
        # It is important to set ``sleep_time`` to 0 here (mandatory for simulator
        # backend)
        dataset_name = args.dataset_name.split("/")[-1]
        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=1,
            sleep_time=0,
            results_update_interval=results_update_interval,
            print_update_interval=print_update_interval,
            # This callback is required in order to make things work with the
            # simulator callback. It makes sure that results are stored with
            # simulated time (rather than real time), and that the time_keeper
            # is advanced properly whenever the tuner loop sleeps
            callbacks=[SimulatorCallback()],
            tuner_name=f"{set_name}-{seed}-{dataset_name.replace('_', '-')}",
            metadata={"description": "Running a baseline for AutoFineTune"},
        )
        try:
            tuner.run()
        except ValueError as e:
            print(e)
            pass
        # print(tuner.get_best_configuration())
        tuning_experiment = load_experiment(tuner.name)
        # print(tuning_experiment)
        result_df = tuning_experiment.results
        epochs = result_df["hp_epochs"].values
        accuracies = result_df["accuracy"].values
        runtimes = result_df["runtime"].values
        info_dict = dict()
        epochs = epochs.tolist()
        spent_epochs = [i for i in range(0, len(epochs))]
        info_dict["accuracy"] = accuracies.tolist()
        max_value = 0
        incumbent_trajectory = []
        for accuracy in accuracies:
            if accuracy > max_value:
                max_value = accuracy
            incumbent_trajectory.append(max_value)

        info_list = []
        info_list.append(spent_epochs)
        info_list.append(runtimes.tolist())
        info_list.append(incumbent_trajectory)
        info_dict["incumbent_trajectory"] = incumbent_trajectory

        output_dir = os.path.join(
            args.output_dir,
            args.method_name,
            set_name,
        )

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{dataset_name}_{seed}.json"), "w") as fp:
            json.dump(info_list, fp)

    run(args)
    print("Done")
