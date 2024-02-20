import torch
import json
import os

from hpo.optimizers.dyhpo.dyhpo import DyHPO, FeatureExtractor
from hpo.optimizers.quick_tune.quick_tune import QuickTuneOptimizer
from hpo.optimizers.quick_tune.cost_metalearner import CostMetaLearner, CostPredictor
from hpo.optimizers.quick_tune.peformance_metalearner import PerformanceMetaLearner

SPLITS = {
    0: [(0, 1, 2), (3,), (4,)],
    1: [(1, 2, 3), (4,), (0,)],
    2: [(2, 3, 4), (0,), (1,)],
    3: [(3, 4, 0), (1,), (2,)],
    4: [(4, 0, 1), (2,), (3,)],
}


def create_qt_optimizer(
    metadataset,
    experiment_id="qt_meta_trained",
    output_dir="output",
    num_hps=70,
    output_dim=32,
    hidden_dim=64,
    output_dim_metafeatures=16,
    freeze_feature_extractor=False,
    new_dataset_name=None,
    dataset_name=None,
    fantasize_step=1,
    minimization=False,
    explore_factor=0.1,
    acqf_fc="ei",
    seed=100,
    budget_limit=500,
    learning_rate=0.001,
    meta_learning_rate=0.01,
    max_budget=50,  # per configuration model
    include_metafeatures=True,
    with_scheduler=True,
    load_meta_trained=False,
    device="cuda",
    train_iter=1000,
    log_indicator=None,
    meta_train=False,
    cost_aware=False,
    load_cost_predictor=False,
    use_encoders_for_model=False,
    meta_output_dir="output",
    split_id=0,
    augmentation_id=None,
    observe_cost=False,
    target_model=None,
    test_generalization_to_model=False,
    use_only_target_model=False,
    subsample_models_in_hub=None,
    file_with_init_indices=None,
    cost_trainer_iter=50,
    input_dim_curves=1,
    **kwargs,
):
    """
    Prepares the QuickTune optimizer according to the specified setup. In performs the following steps:
        - Filters the meta-train data and the meta-dataset (used during BO)
        - Builds the cost and performance predictors
        - Metatrains the cost and performance predictor
    """

    train_splits, test_splits, val_splits = SPLITS[split_id]
    hp_names = metadataset.get_hyperparameters_names()
    hyperparameter_candidates = (
        metadataset.get_hyperparameters_candidates().values.tolist()
    )

    if metadataset.load_only_dataset_descriptors:
        input_dim_metafeatures = 4
    else:
        input_dim_metafeatures = 7684

    if use_encoders_for_model:
        models_input_id = [
            i for i, x in enumerate(hp_names) if x.startswith("cat__model_")
        ]
        encoder_dim_ranges = [
            (models_input_id[0], models_input_id[-1]),
            (models_input_id[-1], len(hp_names)),
        ]
    else:
        encoder_dim_ranges = None

    surrogate_output_dim_metafeatures = (
        output_dim_metafeatures if include_metafeatures else 0
    )
    feature_extractor = FeatureExtractor(
        input_dim_hps=num_hps,
        output_dim=output_dim,
        input_dim_curves=input_dim_curves,
        hidden_dim=hidden_dim,
        output_dim_metafeatures=surrogate_output_dim_metafeatures,
        input_dim_metafeatures=input_dim_metafeatures,
        encoder_dim_ranges=encoder_dim_ranges,
    )

    model = DyHPO(
        device=torch.device(device),
        dataset_name=new_dataset_name,
        output_path=output_dir,
        seed=seed,
        feature_extractor=feature_extractor,
        output_dim=output_dim,
        include_metafeatures=include_metafeatures,
    )
    metadataset.set_dataset_name(dataset_name, augmentation_id=augmentation_id)
    metafeatures = torch.FloatTensor(metadataset.get_metafeatures()).to(device) / 10000
    model.metafeatures = metafeatures

    if file_with_init_indices is not None:
        with open(file_with_init_indices, "r") as f:
            init_conf_indices = json.load(f)[dataset_name]
    else:
        init_conf_indices = None

    optimizer = QuickTuneOptimizer(
        hyperparameter_candidates,
        log_indicator,
        hp_names=hp_names,
        model=model,
        seed=seed,
        max_benchmark_epochs=max_budget,
        fantasize_step=fantasize_step,
        minimization=minimization,
        dataset_name=new_dataset_name,
        output_path=output_dir,
        explore_factor=explore_factor,
        acqf_fc=acqf_fc,
        learning_rate=learning_rate,
        device=device,
        init_conf_indices=init_conf_indices,
    )

    if (target_model is not None) and test_generalization_to_model:
        metadataset.set_action_on_model(target_model, "omit_it")
        metadataset.set_dataset_name(dataset_name, augmentation_id=augmentation_id)

    if (
        (target_model is not None)
        and (not test_generalization_to_model)
        and use_only_target_model
    ):
        metadataset.set_action_on_model(
            target_model,
            "omit_the_rest",
        )
        metadataset.set_dataset_name(dataset_name, augmentation_id=augmentation_id)

    if subsample_models_in_hub is not None:
        metadataset.set_subsample_models(subsample_models_in_hub)
        metadataset.set_dataset_name(dataset_name, augmentation_id=augmentation_id)

    if observe_cost or cost_aware:
        cost_predictor = CostPredictor(
            input_dim_hps=num_hps,
            output_dim_feature_extractor=output_dim,
            input_dim_curves=input_dim_curves,
            hidden_dim=hidden_dim,
            output_dim_metafeatures=output_dim_metafeatures,
            input_dim_metafeatures=input_dim_metafeatures,
        )

        cost_metalearner = CostMetaLearner(
            cost_predictor,
            metadataset,
            checkpoint_path=os.path.join(
                meta_output_dir, f"{experiment_id}_cost_{split_id}.pt"
            ),
            train_splits=train_splits,
            val_splits=val_splits,
            test_splits=test_splits,
            train_iter=train_iter,
        )

        if cost_aware:  # train cost predictor
            if load_cost_predictor:
                cost_metalearner.load_checkpoint()

            else:
                cost_metalearner.meta_train()
                cost_metalearner.save_checkpoint()

        cost_metalearner.cost_predictor.to(device)
        cost_metalearner.train_iter = cost_trainer_iter
        optimizer.model.set_cost_predictor(cost_predictor)

        if observe_cost:
            cost_metalearner.finetuning = True
            optimizer.cost_trainer = cost_metalearner

    if meta_train:
        meta_checkpoint = os.path.join(
            meta_output_dir, f"{experiment_id}_{split_id}.pt"
        )

        if load_meta_trained:
            model.meta_checkpoint = meta_checkpoint
            model.load_checkpoint(meta_checkpoint)
        else:
            perf_metaleaner = PerformanceMetaLearner(
                model,
                metadataset,
                train_iter=train_iter,
                device=optimizer.device,
                learning_rate=meta_learning_rate,
                with_scheduler=with_scheduler,
                include_metafeatures=include_metafeatures,
                train_splits=train_splits,
                val_splits=val_splits,
                test_splits=test_splits,
            )
            model.meta_checkpoint = meta_checkpoint
            val_error = perf_metaleaner.meta_train()
            model.save_checkpoint(checkpoint_file=meta_checkpoint)

        if freeze_feature_extractor:
            optimizer.model.feature_extractor.freeze()
            optimizer.model.original_feature_extractor.freeze()

        # model.checkpoint_file = dataset_checkpoint
        # metadataset.set_dataset_name(dataset_name, augmentation_id=augmentation_id) #metraining changes the dataset name

    if (
        (target_model is not None)
        and test_generalization_to_model
        or use_only_target_model
    ):
        metadataset.set_action_on_model(target_model, "omit_the_rest")

    metadataset.set_dataset_name(dataset_name, augmentation_id=augmentation_id)

    return optimizer
