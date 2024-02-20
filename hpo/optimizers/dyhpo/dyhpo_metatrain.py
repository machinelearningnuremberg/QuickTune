from hpo.optimizers.aft_metadataset import AFTMetaDataset
from hpo.optimizers.performance_predictor import FeatureExtractor
from hpo.optimizers.dyhpo.dyhpo import DyHPO, MetaTrainer
import os

if __name__ == "__main__":

    aggregate_data = False
    aft_set = "micro"
    num_hps = 75
    output_dim = 32
    hidden_dim = 32
    device = "cuda"
    dataset_name = "test"
    seed = 100
    output_path = "experiments/output/hpo/metatrained_dyhpo2/"
    train_iter = 1000
    device = "cuda"
    learning_rate = 0.01
    with_scheduler = False
    include_metafeatures = False
    test_iter = 50
    surrogate_config = {}

    def run():
        if include_metafeatures:
            output_dim_metafeatures = 16
        else:
            output_dim_metafeatures = 0

        metadataset = AFTMetaDataset(aggregate_data=aggregate_data, set=aft_set)
        feature_extractor = FeatureExtractor(
            input_dim_hps=num_hps,
            output_dim=output_dim,
            input_dim_curves=1,
            hidden_dim=hidden_dim,
            output_dim_metafeatures=output_dim_metafeatures,
        )

        model = DyHPO(
            device=device,
            dataset_name=None,
            output_path=output_path,
            seed=seed,
            feature_extractor=feature_extractor,
            output_dim=output_dim,
        )

        model.checkpoint_file = os.path.join(output_path, f"dyhpo_meta_trained.pt")

        meta_trainer = MetaTrainer(
            model,
            metadataset,
            train_iter=train_iter,
            test_iter=test_iter,
            device=device,
            learning_rate=learning_rate,
            with_scheduler=with_scheduler,
            include_metafeatures=include_metafeatures,
        )

        val_error = meta_trainer.meta_train()

        return val_error

    for include_metafeatures in [False, True]:
        val_error = run()
        print(
            f"+++Val error: {val_error}, include_metafeatures: {include_metafeatures}"
        )

    for with_scheduler in [True, False]:
        val_error = run()
        print(f"+++Val error: {val_error}, with_scheduler: {with_scheduler}")

    for learning_rate in [0.001, 0.0001, 0.01]:
        val_error = run()
        print(f"+++Val error: {val_error}, learning_rate: {learning_rate}")

    for hidden_dim in [32, 64, 128]:
        val_error = run()
        print(f"+++Val error: {val_error}, hidden_dim: {hidden_dim}")

    print("Done")
