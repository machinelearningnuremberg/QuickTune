import logging
import time
from syne_tune import Reporter
from argparse import ArgumentParser
import datetime
import os
from finetune_utils.eval_autofinetune import eval_finetune_conf
from meta_album.dataset import AVAILABLE_MTLBM_DATASETS, DEFAULT_FOLDER, AVAILABLE_SETS
import json

def get_task_info(version, set, dataset):

    path = os.path.dirname(__file__)
    with open(os.path.join(path,  "..", "..", "..", DEFAULT_FOLDER, version, set, dataset, "info.json")) as f:
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

if __name__ == "__main__":
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--version', type=str)
    parser.add_argument('--set', type=str)
    parser.add_argument('--verbose', type=int)


    args, _ = parser.parse_known_args()
    report = Reporter()
    experiment_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rootdir = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(rootdir, "..", "..", "..", "datasets", "meta-album")
    verbose = args.verbose
    epochs = args.epochs
    output = os.path.join(rootdir, "..", "..", "..", "experiments", "output", "temp", "synetune", experiment_id,
                          args.dataset)

    task_info, dataset = get_task_info(args.version, args.set, args.dataset)

    for step in range(epochs):
        perf, cost, status = eval_finetune_conf(args.__dict__, task_info, budget = step, experiment = experiment_id,
                                                  data_path = data_path,
                                                  output = output,
                                                  verbose= verbose)
        report(epoch=step+1, mean_loss = -perf)