import os
import argparse

from hpo.optimizers.quick_tune.factory import  SPLITS
from hpo.optimizers.quick_tune.cost_metalearner import CostMetaLearner
from hpo.optimizers.quick_tune.utils_user_interface import run_quicktune, prepare_metafeatures
from hpo.optimizers.qt_metadataset import QuickTuneMetaDataset
from hpo.search_space import SearchSpace


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--max_budget", type=int, default=50)
    parser.add_argument("--train_iter", type=int, default=10000)
    parser.add_argument("--budget_limit", type=int, default=3600)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--experiment_id", type=str, default="qt-mini")
    parser.add_argument("--metadataset_version", type=str, default="micro")
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--cost_aware", action="store_true", default=False)
    parser.add_argument("--meta_train", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_channels", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--dataset_size", type=int, default=None)
    parser.add_argument("--optimizer_name", type=str, default=None)
    parser.add_argument("--search_space_version", type=str, default="v6")
    parser.add_argument("--dataset_download", action="store_true", default=False)

    args = parser.parse_args()

    device = args.device
    meta_train = args.meta_train
    verbose = args.verbose
    experiment_id = args.experiment_id
    metadataset_version = args.metadataset_version
    split_id = args.split_id
    cost_aware = args.cost_aware
    train_iter = args.train_iter
    budget_limit = args.budget_limit
    data_path = args.data_path
    num_classes = args.num_classes
    num_channels = args.num_channels
    image_size = args.image_size
    dataset_size = args.dataset_size
    optimizer_name = args.optimizer_name
    version = args.search_space_version
    dataset_download = args.dataset_download

    rootdir = os.path.dirname(os.path.abspath(__file__))
    metadataset = QuickTuneMetaDataset(aggregate_data=False, version=metadataset_version,
                                 load_only_dataset_descriptors= True)
    ss = SearchSpace(version=version)

    datasets = metadataset.get_datasets()
    train_splits, test_splits, val_splits = SPLITS[split_id]
    datasets_for_split = CostMetaLearner.get_splits(None, datasets, train_splits=train_splits,
                                                         test_splits=test_splits,
                                                         val_splits=val_splits)


    if args.dataset_name is None:
        create_optimizer = True
        for dataset_name in datasets_for_split["test"]:
            print(dataset_name)
            run_quicktune(dataset_name, ss, metadataset, budget_limit, experiment_id,
                                       create_optimizer=create_optimizer,
                                       meta_train=meta_train,
                                       cost_aware=cost_aware,
                                       split_id=split_id,
                                       verbose=verbose,
                                       rootdir=rootdir,
                                       train_iter=train_iter)
            create_optimizer = False #active only meta-training for first dataset
    else:

        assert num_classes is not None
        assert num_channels is not None
        assert image_size is not None
        assert dataset_size is not None

        task_info = {"train_split": args.train_split,
                     "val_split": args.val_split,
                     "dataset": args.dataset_name,
                     "num_classes": args.num_classes}

        metafeatures = prepare_metafeatures(dataset_size, num_classes, image_size, num_channels).to(device)

        run_quicktune(args.dataset_name, ss, metadataset, budget_limit, experiment_id,
                      create_optimizer=False,
                      meta_train=meta_train,
                      cost_aware=cost_aware,
                      split_id=split_id,
                      verbose=verbose,
                      data_path=data_path,
                      task_info=task_info,
                      metafeatures=metafeatures,
                      optimizer_name=optimizer_name,
                      rootdir=rootdir,
                      train_iter=train_iter,
                      dataset_download=dataset_download
                      )

