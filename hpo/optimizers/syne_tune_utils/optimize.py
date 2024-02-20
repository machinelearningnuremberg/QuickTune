import json
import argparse
import os

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend import LocalBackend
from syne_tune.config_space import randint, loguniform, choice
from syne_tune.optimizer.baselines import ASHA
from syne_tune.experiments import load_experiment
from syne_tune.constants import ST_TUNER_TIME
from syne_tune.optimizer.baselines import (
    ASHA,
    MOBSTER,
    HyperTune,
    SyncHyperband,
    SyncBOHB,
    SyncMOBSTER,
    DEHB,
)


from hpo.optimizers.qt_metadataset import QuickTuneMetaDataset
from hpo.search_space import SearchSpace
from meta_album.dataset import AVAILABLE_MTLBM_DATASETS, DEFAULT_FOLDER, AVAILABLE_SETS
from hpo.optimizers.quick_tune.cost_metalearner import CostMetaLearner

SPLITS = {
    0: [(0, 1, 2), (3,), (4,)],
    1: [(1, 2, 3), (4,), (0,)],
    2: [(2, 3, 4), (0,), (1,)],
    3: [(3, 4, 0), (1,), (2,)],
    4: [(4, 0, 1), (2,), (3,)],
}


def get_task_info(version, set, dataset):

    path = os.path.dirname(__file__)
    with open(
        os.path.join(
            path, "..", "..", "..", DEFAULT_FOLDER, version, set, dataset, "info.json"
        )
    ) as f:
        info_json = json.load(f)
    train_split = "train"
    val_split = "val"
    dataset = "mtlbm/{}/{}/{}".format(version, set, dataset)
    num_classes = info_json["total_categories"]

    task_info = {
        "train_split": train_split,
        "val_split": val_split,
        "dataset": dataset,
        "num_classes": num_classes,
    }

    dataset = dataset.replace("/", "_")
    return task_info, dataset


# hyperparameter search space to consider
current_dir = os.path.dirname(os.path.realpath(__file__))
# dataset_name = "mtlbm/micro/set2/TEX_ALOT"


def adapt_search_space(search_space):

    original_config_space = {
        #'lr': loguniform(1e-5, 1e-1),
        # "batch_size": randint(1, 8),
        # "model": choice(["edgenext_x_small", "volo_d5_512"]),
        "epochs": 50,
        "report_synetune": 1,
        "val-split": "val",
        "train-split": "train",
    }

    omit_args = [
        "epochs",
        "data_augmentation",
        "amp",
        "stoch_norm",
        "linear_probing",
        "trivial_augment",
    ]

    for key, values in search_space.data.items():
        if key not in omit_args:
            if isinstance(values, dict):
                values = values["options"]
            values = [x for x in values if x is not None and x != "None"]
            if "True" in values:
                values = [1 if x == "True" else 0 for x in values]
            if key == "opt":
                values = ["adam", "adamw", "adamp"]
            original_config_space[key] = choice(values)

    return original_config_space


def get_scheduler(method, config_space, method_kwargs):

    if method == "BOHB":
        scheduler = SyncBOHB(config_space, **method_kwargs)
    elif method == "DEHB":
        scheduler = DEHB(config_space, **method_kwargs)
    elif method == "MOBSTER":
        scheduler = SyncMOBSTER(config_space, **method_kwargs)
    elif method == "ASHA":
        scheduler = ASHA(config_space, **method_kwargs)
    else:
        raise NotImplementedError

    return scheduler


def eval_meta_album(ss, metadataset, args, experiment_id, budget_limit, n_workers):

    datasets = metadataset.get_datasets()
    train_splits, test_splits, val_splits = SPLITS[args.split_id]
    datasets_for_split = CostMetaLearner.get_splits(
        None,
        datasets,
        train_splits=train_splits,
        test_splits=test_splits,
        val_splits=val_splits,
    )
    for dataset_name in datasets_for_split["test"]:
        # dataset_name = datasets[0]

        aft_set, set, short_dataset_name = dataset_name.split("/")[-3:]
        task_info, new_dataset_name = get_task_info(
            version=aft_set, set=set, dataset=short_dataset_name
        )
        original_config_space = adapt_search_space(ss)
        config_space = original_config_space.copy()
        config_space["dataset"] = dataset_name
        config_space["num_classes"] = task_info["num_classes"]

        method_kwargs = {
            "metric": "eval_accuracy",
            "resource_attr": "epoch",
            "max_resource_attr": "epochs",
            "search_options": {"debug_log": False},
            "mode": "max",
        }
        scheduler = get_scheduler(args.method, config_space, method_kwargs)

        tuner = Tuner(
            trial_backend=LocalBackend(
                entry_point=os.path.join(current_dir, "..", "..", "..", "finetune.py")
            ),
            # scheduler=ASHA(
            #    config_space,
            #    metric='eval_accuracy',
            #    resource_attr='epoch',
            #    max_resource_attr="epochs",
            #    search_options={'debug_log': False},
            #    mode="max"
            # ),
            scheduler=scheduler,
            stop_criterion=StoppingCriterion(max_wallclock_time=budget_limit),
            n_workers=n_workers,  # how many trials are evaluated in parallel
            tuner_name=experiment_id + str(args.split_id),
        )
        tuner.run()
        tuning_experiment = load_experiment(tuner.name)
        # tuning_experiment.plot(figure_path = os.path.join(current_dir, "..", "..", "..", "plots" , "figures", "synetune.png"))

        output_dir = os.path.join(
            current_dir,
            "..",
            "..",
            "..",
            "experiments",
            "output",
            "hpo",
            experiment_id,
            new_dataset_name,
        )

        df = tuning_experiment.results
        if df is not None and len(df) > 0:
            df = df.sort_values(ST_TUNER_TIME)
            x = df.loc[:, ST_TUNER_TIME].values.reshape(-1)
            y = df.loc[:, "eval_accuracy"].cummax().values.reshape(-1) / 100
            results = {"all_perf": y.tolist(), "all_cost": x.tolist()}

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, "perf_curves.json"), "w") as f:
                json.dump(results, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--aft_set", type=str, default="micro")
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--experiment_id", type=str, default="Synetune")
    parser.add_argument("--budget_limit", type=int, default=3600)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--method", type=str, default="BOHB")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")

    args = parser.parse_args()

    aft_set = args.aft_set
    experiment_id = args.experiment_id
    budget_limit = args.budget_limit
    n_workers = args.n_workers
    num_classes = args.num_classes
    dataset = args.dataset
    train_split = args.train_split
    val_split = args.val_split

    metadataset = QuickTuneMetaDataset(
        aggregate_data=False, set=aft_set, load_only_dataset_descriptors=True
    )
    ss = SearchSpace(version="v7")

    if dataset is None:
        eval_meta_album(ss, metadataset, args, experiment_id, budget_limit, n_workers)

    else:
        original_config_space = adapt_search_space(ss)
        config_space = original_config_space.copy()
        config_space["dataset"] = dataset
        config_space["num_classes"] = num_classes
        config_space["train-split"] = train_split
        config_space["val-split"] = val_split
        config_space["workers"] = 1
        config_space["checkpoint_hist"] = 1

        method_kwargs = {
            "metric": "eval_accuracy",
            "resource_attr": "epoch",
            "max_resource_attr": "epochs",
            "search_options": {"debug_log": False},
            "mode": "max",
        }
        scheduler = get_scheduler(args.method, config_space, method_kwargs)

        tuner = Tuner(
            trial_backend=LocalBackend(
                entry_point=os.path.join(
                    current_dir, "..", "..", "..", "finetune_root_folder.py"
                )
            ),
            # scheduler=ASHA(
            #    config_space,
            #    metric='eval_accuracy',
            #    resource_attr='epoch',
            #    max_resource_attr="epochs",
            #    search_options={'debug_log': False},
            #    mode="max"
            # ),
            scheduler=scheduler,
            stop_criterion=StoppingCriterion(max_wallclock_time=budget_limit),
            n_workers=n_workers,  # how many trials are evaluated in parallel
            tuner_name=experiment_id + str(args.split_id),
        )
        tuner.run()
        tuning_experiment = load_experiment(tuner.name)
        # tuning_experiment.plot(figure_path = os.path.join(current_dir, "..", "..", "..", "plots" , "figures", "synetune.png"))

        output_dir = os.path.join(
            current_dir,
            "..",
            "..",
            "..",
            "experiments",
            "output",
            "hpo",
            experiment_id,
            dataset,
        )

        df = tuning_experiment.results
        if df is not None and len(df) > 0:
            df = df.sort_values(ST_TUNER_TIME)
            x = df.loc[:, ST_TUNER_TIME].values.reshape(-1)
            y = df.loc[:, "eval_accuracy"].cummax().values.reshape(-1) / 100
            results = {"all_perf": y.tolist(), "all_cost": x.tolist()}

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(os.path.join(output_dir, "perf_curves.json"), "w") as f:
                json.dump(results, f)
