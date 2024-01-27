import copy
import json
import logging
import math
import os
import time
from typing import Dict, List, Optional, Tuple
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, t
import torch

from hpo.optimizers.qt_metadataset import QTMetaDataset
from hpo.optimizers.quick_tune.cost_predictor import CostPredictorTrainer, CostPredictor
from hpo.optimizers.quick_tune.qt_factory import create_qt_optimizer, SPLITS
from hpo.optimizers.bo import BO
from hpo.optimizers.random_search import RandomSearchOptimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--budget_limit", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="output/")
    parser.add_argument("--aggregate_data", type=int, default=0)
    parser.add_argument("--qt_set", type=str, default="micro")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--output_dim", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--meta_learning_rate", type=float, default=0.01)
    parser.add_argument("--train_iter", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--with_scheduler", type=int, default=0)
    parser.add_argument("--include_metafeatures", type=int, default=1)
    parser.add_argument("--acqf_fc", type=str, default="ei")
    parser.add_argument("--explore_factor", type=float, default=0.1)
    parser.add_argument("--experiment_id", type=str, default="")
    parser.add_argument("--meta_test_id", type=str, default="mt0")
    parser.add_argument("--load_meta_trained", type=int, default=0)
    parser.add_argument("--load_cost_predictor", type=int, default=0)
    parser.add_argument("--output_dim_metafeatures", type=int, default=2)
    parser.add_argument("--freeze_feature_extractor", type=int, default=0)
    parser.add_argument("--run_random", type=int, default=0)
    parser.add_argument("--meta_train", type=int, default=1)
    parser.add_argument("--load_only_dataset_descriptors", type=int, default=1)
    parser.add_argument("--cost_aware", type=int, default=0)
    parser.add_argument("--use_encoders_for_model", type=int, default=0)
    parser.add_argument("--observe_cost", type=int, default=0)
    parser.add_argument("--target_model", type=str, default=None)
    parser.add_argument("--test_generalization_to_model", type=int, default=0)
    parser.add_argument("--use_only_target_model", type=int, default=0)
    parser.add_argument("--split_id", type=int, default=None)
    parser.add_argument("--dataset_id_in_split", type=str, default=None)
    parser.add_argument("--conditioned_time_limit", type=int, default=0)
    parser.add_argument("--subsample_models_in_hub", type=int, default=None)
    parser.add_argument("--measure_for_target_model", type=str, default=None)
    parser.add_argument("--file_with_init_indices", type=str, default=None)

    args = parser.parse_args()
    print(args)
    load_meta_trained = args.load_meta_trained
    load_cost_predictor = args.load_cost_predictor
    results = {}

    rootdir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(rootdir, args.output_dir, args.experiment_id)

    if args.file_with_init_indices is not None:
        args.file_with_init_indices = os.path.join(rootdir, "hpo", "meta_data", args.file_with_init_indices)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadataset = QTMetaDataset(aggregate_data=args.aggregate_data,
                                path = "data",
                                 set=args.qt_set,
                                 load_only_dataset_descriptors= args.load_only_dataset_descriptors)
    if not args.meta_train:
        args.output_dim_metafeatures = 0

    #not used for now
    augmentation_ids = [None]

    if args.split_id is not None:
        split_ids = [args.split_id]
    else:
        split_ids = np.arange(0, 5)

    for split_id in split_ids:
        load_meta_trained = args.load_meta_trained
        load_cost_predictor = args.load_cost_predictor
        datasets = metadataset.get_datasets()
        train_splits, test_splits, val_splits = SPLITS[split_id]
        datasets_for_split = CostPredictorTrainer.get_splits(None, datasets, train_splits=train_splits,
                                                             test_splits=test_splits,
                                                             val_splits=val_splits)
        if args.dataset_id_in_split is not None:
            datasets = [datasets_for_split["test"][int(args.dataset_id_in_split)]]
        else:
            datasets = datasets_for_split["test"]

        for dataset_name in datasets:
            for augmentation_id in augmentation_ids:
                try:
                    metadataset.set_dataset_name(dataset_name, augmentation_id=augmentation_id)
                    hyperparameter_candidates = metadataset.get_hyperparameters_candidates().values.tolist()
                    log_indicator = [False for _ in range(len(hyperparameter_candidates[0]))] #Dont apply log 
                    new_dataset_name = dataset_name.replace("/", "_")
                    if augmentation_id is not None:
                        new_dataset_name = new_dataset_name + "_aug" + str(augmentation_id)

                    #loads the best model per dataset
                    if args.measure_for_target_model is not None:
                        with open(os.path.join(rootdir, "meta_data", "best_models_per_dataset.json"), "r") as f:
                            best_models_per_datasets = json.load(f)
                        target_model = best_models_per_datasets[dataset_name][args.measure_for_target_model]
                    else:
                        target_model = args.target_model

                    #TODO: check the functionality
                    if args.conditioned_time_limit:
                        with open(os.path.join(rootdir, "meta_data", "time_counts.json"), "r") as f:
                            time_limits = json.load(f)
                        budget_limit = time_limits[args.qt_set][new_dataset_name] + 100
                    else:
                        budget_limit = args.budget_limit

                    temp_output_dir = os.path.join(output_dir, args.meta_test_id,  new_dataset_name)
                    if not os.path.exists(temp_output_dir):
                        os.makedirs(temp_output_dir)

                    if args.run_random:
                        random_search = RandomSearchOptimizer(metadataset, seed=args.seed)
                        optimizer_budget, optimizer_cost, optimizer_performance = BO(random_search, metadataset, args.budget_limit)
                        results = [optimizer_budget, optimizer_cost, optimizer_performance]
                    else:
                        optimizer = create_qt_optimizer(metadataset, experiment_id = args.experiment_id,
                                                                    output_dim_metafeatures = args.output_dim_metafeatures,
                                                                    freeze_feature_extractor = args.freeze_feature_extractor,
                                                                    explore_factor = args.explore_factor,
                                                                    load_meta_trained = load_meta_trained,
                                                                    meta_output_dir = output_dir,
                                                                    output_dir = temp_output_dir,
                                                                    dataset_name = dataset_name,
                                                                    new_dataset_name = new_dataset_name,
                                                                    log_indicator = log_indicator,
                                                                    budget_limit = budget_limit,
                                                                    include_metafeatures = args.include_metafeatures,
                                                                    meta_train = args.meta_train,
                                                                    acqf_fc = args.acqf_fc,
                                                                    learning_rate = args.learning_rate,
                                                                    meta_learning_rate = args.meta_learning_rate,
                                                                    train_iter = args.train_iter,
                                                                    hidden_dim = args.hidden_dim,
                                                                    output_dim = args.output_dim,
                                                                    with_scheduler = args.with_scheduler,
                                                                    cost_aware=args.cost_aware,
                                                                    use_encoders_for_model=args.use_encoders_for_model,
                                                                    load_cost_predictor=load_cost_predictor,
                                                                    split_id=split_id,
                                                                    augmentation_id=augmentation_id,
                                                                    observe_cost=args.observe_cost,
                                                                    target_model = target_model,
                                                                    seed = args.seed,
                                                                    test_generalization_to_model = args.test_generalization_to_model,
                                                                    use_only_target_model = args.use_only_target_model,
                                                                    subsample_models_in_hub = args.subsample_models_in_hub,
                                                                    file_with_init_indices = args.file_with_init_indices)

                        optimizer_budget, optimizer_cost, optimizer_performance = BO(optimizer, metadataset, budget_limit,
                                                                                     observe_cost= args.observe_cost)
                        results = [optimizer_budget, optimizer_cost, optimizer_performance]
                        load_meta_trained = True
                        load_cost_predictor = True

                    with open(os.path.join(temp_output_dir, f"results.json"), "w") as f:
                        json.dump(results, f)

                except Exception as e:
                    print(e)
                    print("Error in dataset: ", dataset_name, augmentation_id)
