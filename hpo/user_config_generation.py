import os
import json
from meta_album.dataset import AVAILABLE_MTLBM_DATASETS, DEFAULT_FOLDER, AVAILABLE_SETS
from hpo.optimizers.aft_metadataset import AFTMetaDataset


def get_task_info(version, set, dataset):

    path = os.path.dirname(__file__)
    with open(
        os.path.join(path, "..", DEFAULT_FOLDER, version, set, dataset, "info.json")
    ) as f:
        info_json = json.load(f)
    train_split = "train"
    val_split = "val"
    dataset = "mtlbm/{}/{}/{}".format(version, set, dataset)
    num_classes = info_json["total_categories"]
    dataset_size = (
        info_json["median_images_per_category"] * info_json["total_categories"]
    )

    task_info = {
        "train_split": train_split,
        "val_split": val_split,
        "dataset": dataset,
        "num_classes": num_classes,
        "dataset_size": dataset_size,
    }

    dataset = dataset.replace("/", "_")
    return task_info, dataset


super_sets = ["micro", "mini", "extended"]
sets = ["set0", "set1", "set2"]
cmd_list = []
other = False

for super_set in super_sets:

    metadataset = AFTMetaDataset(
        aggregate_data=False, set=super_set, load_only_dataset_descriptors=True
    )

    datasets = metadataset.get_datasets()

    for dataset in datasets:
        _, super_set, set, new_dataset = dataset.split("/")

        task_info, new_dataset = get_task_info(super_set, set, new_dataset)
        print(task_info)

        num_classes = task_info["num_classes"]
        dataset_size = task_info["dataset_size"]
        cmd = (
            f"--data_path datasets/meta-album "
            f"--dataset_name {dataset} "
            + f"--num_classes {num_classes} "
            + f"--num_channels 3 "
            + f"--image_size 128 "
            + f"--verbose "
            + f"--dataset_size {dataset_size} "
            + f"--experiment_id qt-rebuttal-reviewer4-win "
            + f"--optimizer_name dyhpo-{super_set} "
            + f"--budget_limit 86000 --search_space_version v8"
        )
        print(cmd)
        cmd_list.append(cmd)

# write in txt
with open("RB06.args", "w") as f:
    for cmd in cmd_list:
        f.write(cmd + "\n")
