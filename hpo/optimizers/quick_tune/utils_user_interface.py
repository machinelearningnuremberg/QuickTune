import os
import json
import datetime
import shutil
import torch
import time
import pickle

import numpy as np
import pandas as pd

from meta_album.dataset import AVAILABLE_MTLBM_DATASETS, DEFAULT_FOLDER, AVAILABLE_SETS
from hpo.optimizers.quick_tune.factory import create_qt_optimizer, SPLITS
from hpo.optimizers.bo import BO
from finetune_utils.eval_autofinetune import eval_finetune_conf


def get_optimizer(
    metadataset,
    budget_limit,
    new_dataset_name,
    dataset_name,
    temp_output_dir,
    output_dir,
    experiment_id,
    log_indicator,
    split_id=0,
    learning_rate=0.0001,
    meta_learning_rate=0.0001,
    hidden_dim=32,
    output_dim=32,
    with_scheduler=True,
    cost_aware=True,
    train_iter=100,
    use_encoders_for_model=True,
    include_metafeatures=True,
    output_dim_metafeatures=4,
    freeze_feature_extractor=False,
    explore_factor=0.0,
    meta_train=True,
    load_meta_trained=False,
):

    optimizer = create_qt_optimizer(
        metadataset,
        experiment_id=experiment_id,
        output_dim_metafeatures=output_dim_metafeatures,
        freeze_feature_extractor=freeze_feature_extractor,
        explore_factor=explore_factor,
        load_meta_trained=load_meta_trained,
        meta_output_dir=output_dir,
        output_dir=temp_output_dir,
        dataset_name=dataset_name,
        new_dataset_name=new_dataset_name,
        log_indicator=log_indicator,
        budget_limit=budget_limit,
        include_metafeatures=include_metafeatures,
        meta_train=meta_train,
        learning_rate=learning_rate,
        meta_learning_rate=meta_learning_rate,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        with_scheduler=with_scheduler,
        cost_aware=cost_aware,
        use_encoders_for_model=use_encoders_for_model,
        train_iter=train_iter,
        split_id=split_id,
    )

    return optimizer


def preprocess_configurations(configurations, hp_names, ss, metadataset):

    conf_df = pd.DataFrame(configurations)

    conf_df["batch_size"] = conf_df["batch_size"].apply(lambda x: min(x - x % 2, 256))

    # convert to the hp names type
    data_augmentations = ["trivial_augment", "random_augment"]
    categorical_groups, mapping_to_group = get_categorical_groups(ss)
    hp_values = []
    default_values = {
        "patience_epochs": 10.0,
        "decay_epochs": 20.0,
        "decay_rate": 0.1,
        "momentum": 0.9,
        "ra_magnitude": 8.0,
        "ra_num_ops": 2.0,
    }

    try:
        for hp in hp_names:
            if hp.startswith("cat__"):
                if hp in mapping_to_group.keys():
                    hp_name, hp_option = mapping_to_group[hp]
                    if hp_name in conf_df.columns:
                        hp_values.append(
                            conf_df[hp_name]
                            .apply(lambda x: 1 if x == hp_option else 0)
                            .values
                        )
                    else:
                        hp_values.append(np.zeros(conf_df.shape[0]))
                else:
                    hp_values.append(np.zeros(conf_df.shape[0]))
            elif hp in data_augmentations:
                if "data_augmentation" in conf_df.columns:
                    hp_values.append(
                        conf_df["data_augmentation"]
                        .apply(lambda x: 1 if x == hp else 0)
                        .values
                    )
                # else:
                #    hp_values.append(np.zeros(conf_df.shape[0]))
            else:
                if hp in conf_df.columns:
                    x = conf_df[hp].values
                    x[x == "None"] = -1.0
                    if hp in default_values.keys():
                        x = [default_values[hp] if np.isnan(v) else v for v in x]
                elif hp in default_values.keys():
                    x = [default_values[hp] for v in x]
                else:
                    x = np.zeros(conf_df.shape[0])
                x = np.array(x, dtype=np.float32)
                if np.isnan(x).any():
                    print("hp {} is nan".format(hp))
                hp_values.append(x)
        # convert to the hp names type
        # hp_values = np.concatenate(hp_values, axis=1)
    except Exception as e:
        print(e)
        raise ValueError("The configuration is not valid")
    hp_values = [np.array(x, dtype=np.float32).reshape(1, -1) for x in hp_values]
    hp_values = np.vstack(hp_values).T.round(6)
    # standardize the hp for input to optimizer

    mean_values = metadataset.args_mean[hp_names].values
    std_values = metadataset.args_std[hp_names].values

    for i in range(hp_values.shape[1]):
        if not hp_names[i].startswith("cat") and std_values[i] != 0:
            hp_values[:, i] = (hp_values[:, i] - mean_values[i]) / std_values[i]

    return hp_values


def postprocess_configurations(x, metadataset, hp_names):
    # only one configuration is valid
    mean_values = metadataset.args_mean[hp_names].values.round(6)
    std_values = metadataset.args_std[hp_names].values.round(6)
    x = x.round(6)
    for i in range(x.shape[0]):
        if not hp_names[i].startswith("cat") and std_values[i] != 0:
            x[i] = x[i] * std_values[i] + mean_values[i]

    return x.round(6)


def validate_configuration(configuration, search_space):

    configuration["batch_size"] = min(
        configuration["batch_size"] + configuration["batch_size"] % 2, 256
    )

    for hp in configuration.keys():

        data = search_space.__dict__["data"]
        if hp in data.keys():
            values = data[hp].copy()
            current_value = configuration[hp]

            if hp == "clip_grad":
                values.remove("None")

            if isinstance(values, dict):
                values = values["options"]

            if isinstance(current_value, (int, float, np.float32)):

                if "None" in values:
                    values.remove("None")
                min_hp = min(values)
                max_hp = max(values)
                configuration[hp] = min(max(current_value, min_hp), max_hp)

            elif current_value == "None":
                pass
            elif isinstance(current_value, str):
                if hp != "opt_betas":
                    assert (
                        configuration[hp] in values
                    ), "The value {} is not in the search space".format(
                        configuration[hp]
                    )

    return configuration


def check_values(hp_values, metadataset, hp_names):
    # check if the hp values are within the search space

    current_min_values = hp_values.min(axis=1)
    metadataset_min_values = metadataset.args_min[hp_names]
    check_min = [
        x1 >= x2 - 0.0000001
        for x1, x2 in zip(current_min_values, metadataset_min_values)
    ]

    current_max_values = hp_values.max(axis=1)
    metadataset_max_values = metadataset.args_max[hp_names]
    check_max = [
        x1 <= x2 + 0.0000001
        for x1, x2 in zip(current_max_values, metadataset_max_values)
    ]
    print(hp_values.shape)


def check_aft_config(aft_config, original_config):

    try:
        assert aft_config["model"] == original_config["model"]
        assert aft_config["sched"] == original_config["sched"]
        assert aft_config["opt"] == original_config["opt"]
        if aft_config["opt_betas"] != "None":
            assert (
                aft_config["opt_betas"]
                .replace("[", "")
                .replace("]", "")
                .replace(",", "")
                .replace("0.0", "0")
                == original_config["opt_betas"]
            )
        assert np.isclose(
            aft_config["weight_decay"], original_config["weight_decay"], atol=1e-5
        )
        assert np.isclose(aft_config["lr"], original_config["lr"], atol=1e-5)
        assert np.isclose(
            aft_config["delta_reg"], original_config["delta_reg"], atol=1e-5
        )
        assert np.isclose(aft_config["bss_reg"], original_config["bss_reg"], atol=1e-5)
        assert np.isclose(
            aft_config["cotuning_reg"], original_config["cotuning_reg"], atol=1e-5
        )
        assert np.isclose(aft_config["cutmix"], original_config["cutmix"], atol=1e-5)
        assert np.isclose(aft_config["drop"], original_config["drop"], atol=1e-5)
        assert np.isclose(aft_config["drop"], original_config["drop"], atol=1e-5)
        assert np.isclose(
            aft_config["warmup_lr"], original_config["warmup_lr"], atol=1e-5
        )
        assert np.isclose(aft_config["sp_reg"], original_config["sp_reg"], atol=1e-5)
        if "layer_decay" in aft_config.keys():
            assert np.isclose(
                aft_config["layer_decay"], original_config["layer_decay"], atol=1e-5
            )
    except:
        print(aft_config)
        print(original_config)
        raise ValueError("The configuration is not valid")


def get_categorical_groups(search_space):

    categorical_vars = ["model", "sched", "auto_augment", "opt", "opt_betas"]
    categorical_groups = {}
    mapping_to_group = {}
    for var in categorical_vars:

        if isinstance(search_space.data[var], list):

            names = [f"cat__{var}_{i}" for i in search_space.data[var]]
            categorical_groups[var] = names
            for i, k in enumerate(names):
                mapping_to_group[k] = (var, search_space.data[var][i])
        else:
            if var == "opt_betas":
                names = [f"cat__{var}_[{i}]" for i in search_space.data[var]["options"]]
                names = [name.replace(" ", ", ") for name in names]
                names = [name.replace("0,", "0.0,") for name in names]
            else:
                names = [f"cat__{var}_{i}" for i in search_space.data[var]["options"]]
            categorical_groups[var] = names
            for i, k in enumerate(names):
                mapping_to_group[k] = (var, search_space.data[var]["options"][i])

    return categorical_groups, mapping_to_group


def get_task_info(version, set, dataset, rootdir):

    with open(
        os.path.join(
            rootdir,
            "..",
            "AutoFinetune",
            DEFAULT_FOLDER,
            version,
            set,
            dataset,
            "info.json",
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


def run_test(optimizer, hp_names, ss, metadataset):

    original_configurations = [
        {
            "amp": False,
            "batch_size": 8,
            "bss_reg": 0,
            "clip_grad": 10,
            "cotuning_reg": 1,
            "cutmix": 0.1,
            "data_augmentation": "trivial_augment",
            "delta_reg": 0.1,
            "drop": 0.3,
            "epochs": 50,
            "layer_decay": "None",
            "linear_probing": False,
            "lr": 0.001,
            "mixup": 1,
            "mixup_prob": 0,
            "model": "tf_efficientnet_b6_ns",
            "opt": "sgd",
            "pct_to_freeze": 0.8,
            "sched": "None",
            "smoothing": 0,
            "sp_reg": 0,
            "stoch_norm": True,
            "warmup_epochs": 5,
            "warmup_lr": 1e-05,
            "weight_decay": 0.001,
        }
    ]

    configurations = preprocess_configurations(
        original_configurations, hp_names, ss, metadataset
    )
    x = dict(zip(hp_names, configurations[0].tolist()))
    x2 = pd.DataFrame([x])
    posprocessed_configuration = postprocess_configurations(
        configurations[0], metadataset, hp_names
    )
    aft_config = optimizer.to_qt_config(posprocessed_configuration)


def prepare_metafeatures(dataset_size, num_classes, image_size, num_channels):

    # example descriptor: [[22391, 50, 128, 3]]
    return torch.FloatTensor(
        [dataset_size, num_classes, image_size, num_channels]
    ).reshape(1, -1)


def run_quicktune(
    dataset_name,
    ss,
    metadataset,
    budget_limit,
    experiment_id,
    create_optimizer=False,
    meta_train=False,
    cost_aware=False,
    split_id=0,
    verbose=False,
    task_info=None,
    data_path=None,
    metafeatures=None,
    optimizer_name=None,
    rootdir=None,
    train_iter=10000,
    num_configurations=400,
    dataset_download=False,
):
    """
    By default (when task_info and data_path are None), this function will run the experiment on meta-album.
    """
    num_hps = 69
    log_indicator = [False for _ in range(num_hps)]

    run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    if create_optimizer and task_info is not None:
        raise ValueError("If you want to create an optimizer, task_info must be None")

    if task_info is None:
        aft_set, set, short_dataset_name = dataset_name.split("/")[-3:]
        new_dataset_name = dataset_name.replace("/", "_")
        metadataset.set_dataset_name(dataset_name)
        hp_names = metadataset.get_hyperparameters_names()
        task_info, dataset = get_task_info(
            version=aft_set, set=set, dataset=short_dataset_name, rootdir=rootdir
        )

    else:
        new_dataset_name = task_info["dataset"]
        dataset = new_dataset_name
        metadataset.set_dataset_name(
            metadataset.get_datasets()[0]
        )  # inneficient, to improve
        hp_names = metadataset.get_hyperparameters_names()

    # load pickled optimizer
    experiment_dir = os.path.join(rootdir, "output", experiment_id)

    if create_optimizer:
        temp_output_dir = os.path.join(experiment_dir, run_id, new_dataset_name)

        if not os.path.exists(temp_output_dir):
            os.makedirs(temp_output_dir)

        optimizer = get_optimizer(
            metadataset,
            budget_limit,
            new_dataset_name,
            dataset_name,
            temp_output_dir,
            experiment_dir,
            experiment_id,
            log_indicator,
            train_iter=train_iter,
            meta_train=meta_train,
            cost_aware=cost_aware,
            split_id=split_id,
        )

        # pickle the optimizer
        with open(os.path.join(experiment_dir, "optimizer.pkl"), "wb") as f:
            pickle.dump(optimizer, f)

    else:
        # temp_output_dir = os.path.join(output_dir, run_id, new_dataset_name)
        if optimizer_name is None:
            optimizer_path = os.path.join(experiment_dir, "optimizer.pkl")
        else:
            optimizer_path = os.path.join(
                rootdir, "output", optimizer_name, "optimizer.pkl"
            )
        with open(optimizer_path, "rb") as f:
            optimizer = pickle.load(f)

    if metafeatures is None:
        optimizer.model.metafeatures = metadataset.get_metafeatures() / 10000
    else:
        optimizer.model.metafeatures = metafeatures / 10000

    original_configurations = ss.sample_configuration(num_configurations)
    configurations = preprocess_configurations(
        original_configurations, hp_names, ss, metadataset
    )

    # prepare the hp for input to the response function
    optimizer.set_hyperparameter_candidates(configurations)

    if data_path is None:
        data_path = os.path.join(rootdir, "datasets", "meta-album")

    perf_curves = {"all_perf": [], "all_cost": []}
    temp_output = os.path.join(
        rootdir, "output", experiment_id, "temp", run_id, dataset
    )

    # remove folder
    if os.path.exists(os.path.join(temp_output)):
        shutil.rmtree(os.path.join(temp_output))

    info_dict = {
        "all_configs": [dict(config) for config in original_configurations],
        "observed_ids": [],
        "query_config": [],
        "status": [],
    }

    highest_perf = 0
    start_time = time.time()
    done = False

    while not done:
        hp_index, budget = optimizer.suggest()

        if not str(hp_index) in perf_curves.keys():
            perf_curves[str(hp_index)] = []
        torch.cuda.empty_cache()
        selected_hp = configurations[hp_index]
        selected_hp = postprocess_configurations(selected_hp, metadataset, hp_names)
        aft_config = optimizer.to_qt_config(selected_hp)

        temp_config_id = f"{experiment_id}.{run_id}.{hp_index}"
        aft_config = validate_configuration(aft_config, ss)
        print("Dataset download")
        perf, cost, status = eval_finetune_conf(
            aft_config,
            task_info,
            budget=budget,
            experiment=temp_config_id,
            data_path=data_path,
            output=temp_output,
            verbose=verbose,
            dataset_download=dataset_download,
        )

        if perf == 0:
            optimizer.converged_configs.append(hp_index)

        perf_curves[str(hp_index)].append(perf / 100)
        overhead_time = optimizer.observe(hp_index, budget, perf_curves[str(hp_index)])

        perf_curves["all_perf"].append(perf / 100)
        perf_curves["all_cost"].append(time.time() - start_time)
        info_dict["observed_ids"].append(int(hp_index))
        info_dict["query_config"].append(str(aft_config))
        info_dict["status"].append(status)

        os.makedirs(os.path.join(experiment_dir, new_dataset_name), exist_ok=True)
        if perf > highest_perf:
            highest_perf = perf
            # save best config
            # move file to output folder
            shutil.copy(
                os.path.join(temp_output, temp_config_id, "last.pth.tar"),
                os.path.join(experiment_dir, new_dataset_name, "best_model.pt"),
            )

        # save info dict
        with open(
            os.path.join(experiment_dir, new_dataset_name, "info_dict.json"), "w"
        ) as f:
            # dump json
            json.dump(info_dict, f)

        # save curves in output folder
        with open(
            os.path.join(experiment_dir, new_dataset_name, "perf_curves.json"), "w"
        ) as f:
            # dump json
            json.dump(perf_curves, f)

        if perf_curves["all_cost"][-1] > budget_limit:
            done = True
