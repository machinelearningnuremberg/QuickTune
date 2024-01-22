from hpo.optimizers.dyhpo.dyhpo import DyHPO, MetaTrainer
from hpo.optimizers.performance_predictor import FeatureExtractor
from hpo.optimizers.aft_metadataset import AFTMetaDataset
from hpo.optimizers.dyhpo.discrete_interface import prepare_dyhpo_optimizer
import os
import torch

if __name__ == "__main__":


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
    num_hps = 75
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
                                                    output_dim_metafeatures = 0,
                                                    freeze_feature_extractor = False,
                                                    explore_factor = 0.0,
                                                    meta_train=False,
                                                    output_dir = output_dir,
                                                    dataset_name=dataset_name,
                                                    new_dataset_name=new_dataset_name,
                                                    log_indicator=log_indicator)

    max_normalized = (metadataset.args_max-metadataset.args_mean)/metadataset.args_std
    min_normalized = (metadataset.args_min-metadataset.args_mean)/metadataset.args_std
    max_normalized = max_normalized[optimizer.args_names]
    surrogate = optimizer.model
    optimizer_budget, optimizer_cost, optimizer_performance = BO(optimizer, metadataset, budget_limit)


    def objective(x):

        x = torch.FloatTensor(x)
        budgets = torch.FloatTensor([0]*len(x))
        curves = torch.zeros(len(x), max_budget)
        train_data = optimizer._prepare_dataset_and_budgets()

        test_data = {
            'X_test': x,
            'test_budgets': budgets,
            'test_curves': curves
        }

        mean_predictions, std_predictions = optimizer.model.predict_pipeline(train_data, test_data)
        return mean_predictions






