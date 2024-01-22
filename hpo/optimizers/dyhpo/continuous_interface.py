import copy

from hpo.optimizers.dyhpo.dyhpo import DyHPO, MetaTrainer, FeatureExtractor
from hpo.optimizers.aft_metadataset import AFTMetaDataset
from hpo.optimizers.dyhpo.discrete_interface import prepare_dyhpo_optimizer
import os
import torch
from discrete_interface import BO
from scipy.optimize import rosen, differential_evolution
import numpy as np
from finetune_utils.eval_autofinetune import eval_finetune_conf
from meta_album.dataset import AVAILABLE_MTLBM_DATASETS, DEFAULT_FOLDER, AVAILABLE_SETS
import json
import pickle

def get_task_info(version, set, dataset):

    path = os.path.dirname(__file__)
    with open(os.path.join(path, "..", "..", "..", DEFAULT_FOLDER, version, set, dataset, "info.json")) as f:
        info_json = json.load(f)
    train_split = "train"
    val_split = "val"
    dataset = "mtlbm/{}/{}/{}".format(version, set, dataset)
    num_classes = info_json["total_categories"]

    task_info = {"train_split": train_split,
                 "val_split": val_split,
                 "dataset": dataset,
                 "num_classes": num_classes}

    dataset = dataset.replace("/", "_")
    return task_info, dataset


def get_continuous_config_code(config):
    config_code = ""
    for k, v in config.items():
        if isinstance(v, float):
            if v!=0:
                num_significant_digits = int(np.ceil(-np.log10(np.abs(v))+3))
            v = np.round(v, num_significant_digits)
        config_code += "{}_{}_".format(k, v)


    return config_code

if __name__ == "__main__":

    #WARNING THIS WILL BREAK BECAUSE THE NUMBER OF CLASSES CHANGE! TO DO: FIX IT
    max_budget = 50
    seed = 100
    fantasize_step = 1
    minimization = False
    budget_limit = 1000
    output_dir = "experiments/output/hpo/dyhpo20/"
    #output_dir = "../../../experiments/output/hpo/dyhpo13/"
    #output_dir = "experiments/output/hpo/dyhpo03/"
    #Suggested conf: 315, budget: 1

    aggregate_data = False
    aft_set = "micro"
    num_hps = 74
    hidden_dim = 64
    output_dim = 32
    learning_rate = 0.001
    meta_learning_rate = 0.01
    train_iter = 1000
    device = "cuda"
    with_scheduler = True
    include_metafeatures = True
    acqf_fc = "ei"
    explore_factor = 0.1
    run_id = "continuous_dyhpo00"
    meta_trained = False
    results = {}
    # check if output dir exists

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    metadataset = AFTMetaDataset(aggregate_data=aggregate_data, set=aft_set)
    dataset_name = metadataset.get_datasets()[0]


    metadataset.set_dataset_name(dataset_name)
    hyperparameter_candidates = metadataset.get_hyperparameters_candidates().values.tolist()
    log_indicator = [False for _ in range(len(hyperparameter_candidates[0]))]
    new_dataset_name = dataset_name.replace("/", "_")


    optimizer = prepare_dyhpo_optimizer(metadataset, name = "dyhpo_non_metatrained1",
                                                    hidden_dim = 32,
                                                    output_dim = 32,
                                                    learning_rate = 0.0001,
                                                    output_dim_metafeatures = 0,
                                                    freeze_feature_extractor = False,
                                                    explore_factor = 0.0,
                                                    load_meta_trained=False,
                                                    output_dir = output_dir,
                                                    dataset_name=dataset_name,
                                                    new_dataset_name=new_dataset_name,
                                                    log_indicator=log_indicator)

    with open(os.path.join(output_dir, "optimizer.pkl"), "wb") as f:
        pickle.dump(optimizer, f)

    max_normalized = (metadataset.args_max-metadataset.args_mean)/metadataset.args_std
    min_normalized = (metadataset.args_min-metadataset.args_mean)/metadataset.args_std
    max_normalized = max_normalized[metadataset.hyperparameter_names]
    surrogate = optimizer.model
    optimizer_budget, optimizer_cost, optimizer_performance = BO(optimizer, metadataset, budget_limit=2)
    train_data = optimizer._prepare_dataset_and_budgets()



    extended_test_data = copy.deepcopy(train_data)

    for x, y, budget, curve in zip(train_data["X_train"], train_data["y_train"], \
                                   train_data["train_budgets"], train_data["train_curves"]):
        curve[budget.int()] = y
        budget += 1
        print(curve, budget)
    extended_test_data.pop("y_train")
    extended_test_data["X_test"] = train_data["X_train"]
    extended_test_data["test_budgets"] = train_data["train_budgets"]
    extended_test_data["test_curves"] = train_data["train_curves"]

    extended_mean_predictions, extended_std_predictions, extedend_cost_predictions = optimizer.model.predict_pipeline(train_data, extended_test_data)

    #treating special hps
    idx_num_classes = np.where(np.array(metadataset.hyperparameter_names) == "num_classes")[0]
    idx_amp = np.where(np.array(metadataset.hyperparameter_names) == "amp")[0]

    def objective(x):
        num_samples, dim = train_data["X_train"].shape
        device = train_data["X_train"].device

        x = torch.FloatTensor(x).reshape(-1, dim).to(device)
        budgets = torch.FloatTensor([0]*len(x)).to(device)
        curves = torch.zeros(len(x), max_budget).to(device)

        #setting common features
        x[:,idx_num_classes] = train_data["X_train"][:,idx_num_classes][0]
        x[:,idx_amp] = train_data["X_train"][:,idx_amp][0]

        test_data = {
            'X_test': x,
            'test_budgets': budgets,
            'test_curves': curves
        }

        mean_predictions, std_predictions, costs = optimizer.model.predict_pipeline(train_data, test_data)
        return -mean_predictions

    print("Done")
    bounds = list(zip(min_normalized, max_normalized))
    result = differential_evolution(objective, bounds)
    suggested_x = result.x
    suggested_x = suggested_x * metadataset.args_std[metadataset.hyperparameter_names] + metadataset.args_mean[metadataset.hyperparameter_names]
    aft_config = optimizer.to_aft_config(suggested_x)
    aft_config["opt_betas"] = aft_config["opt_betas"].replace("[", "").replace("]", "").replace(", "," ")
    aft_config["batch_size"] = aft_config["batch_size"] - aft_config["batch_size"] % 2
    rootdir = os.path.dirname(os.path.abspath(__file__))

    version = "micro"
    set = "set0"
    dataset = AVAILABLE_MTLBM_DATASETS[set][0]
    data_path = os.path.join(rootdir, "..", "..", "..", "datasets", "meta-album")
    output = os.path.join(rootdir, "..", "..", "..", "experiments", "output", "temp", run_id, dataset)

    task_info, dataset = get_task_info(version=version, set=set, dataset=dataset)
    experiment = get_continuous_config_code(aft_config)
    perf, cost = eval_finetune_conf(aft_config, task_info, experiment = experiment,
                                              data_path = data_path,
                                              output = output)
    #check search space contiions> betas not allowed in SGD
    #log scale transformation
    #return the best curve
    #execute the returned learning curve




