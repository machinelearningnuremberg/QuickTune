import matplotlib.pyplot as plt
import os
from transferability_measure.Task2Vec import models, datasets, task2vec, task_similarity
from timm.data import create_dataset
from finetune_utils.utils import get_icgen_dataset_info_json, get_number_of_classes, get_dataset_path
from torchvision import  transforms
from finetune_utils.available_datasets import all_datasets
from hpo.create_datasets_x_models_args import handle_splits
from meta_album.dataset import AVAILABLE_MTLBM_DATASETS, DEFAULT_FOLDER, AVAILABLE_SETS
import torch
import torchvision
import json

def get_embedding(dataset_name, metadataset_name, augmentation_id = 1, epochs=1, max_samples=None, method="montecarlo", network="resnet34"):

    class_map = ""
    dataset_download = False
    batch_size = 2
    seed = 42
    epochs_repeats = 0
    input_size = 224

    if metadataset_name == "zap":
        dataset_augmentation_path = f"datasets/tfds-icgen/{augmentation_id}"
        dataset = f"tfdsicgn/{dataset_name}"
        data_dir = "datasets/tfds-icgen/core_datasets"
        train_split, test_split = handle_splits(dataset_augmentation_path, dataset.split("/")[1])
        if test_split != "":
            train_split = train_split + "+" + test_split
        else:
            train_split = train_split
        is_gray = True if "mnist" in dataset_name else False
        in_chans = 1 if is_gray else 3
        dataset_aug_path = get_dataset_path(dataset_augmentation_path, dataset)
        icgen_dataset_info = get_icgen_dataset_info_json(dataset_aug_path, dataset)
        dataset_aug_path = get_dataset_path(dataset_augmentation_path, dataset)
        icgen_dataset_info = get_icgen_dataset_info_json(dataset_aug_path, dataset)
        num_classes = icgen_dataset_info["number_classes"]
        image_resolution = icgen_dataset_info["resolution"]
        extra_dataset_info = {"icgen_dataset_info": icgen_dataset_info}
    else:
        with open(os.path.join( DEFAULT_FOLDER, dataset_name, "info.json")) as f:
            info_json = json.load(f)
        dataset = f"mtlbm/{dataset_name}"
        train_split = "train"
        data_dir = f"datasets/meta-album"
        data_dir = os.path.join(*data_dir.split("/"))
        in_chans = 3
        is_gray = False
        num_classes = info_json["total_categories"]
        extra_dataset_info = {}
        image_resolution = None

    dataset_train = create_dataset(
        dataset,
        root=data_dir,
        split=train_split,
        is_training=True,
        class_map=class_map,
        download=dataset_download,
        batch_size=batch_size,
        seed=seed,
        repeats=epochs_repeats,
        input_img_mode="L" if in_chans == 1 else "RGB",
        **extra_dataset_info
    )
    data_transforms = []

    if image_resolution is None:
        image_resolution = dataset_train.get_image_size()[0]

    if is_gray == 1:
        data_transforms.extend([
                                transforms.Resize(input_size),
                                transforms.CenterCrop(input_size),
                                torchvision.transforms.Grayscale(num_output_channels=3),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        in_chans = 1
    else:
        data_transforms.extend([
                                transforms.Resize(input_size),
                                transforms.CenterCrop(input_size),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        in_chans = 3
    data_transforms = transforms.Compose(data_transforms)
    dataset_train.transform = data_transforms
    probe_network = models.get_model(network, pretrained=True, num_classes=num_classes).cuda()
    loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=1)
    embedding = task2vec.Task2Vec(probe_network, max_samples=max_samples, skip_layers=6, loader=loader, method=method, method_opts={"epochs": epochs}).embed(dataset_train)
    #in_chans = dataset_train[0][0].shape[0]
    descriptors = [len(dataset_train), num_classes, image_resolution, in_chans]

    return embedding, descriptors

if __name__ == "__main__":

    method = "montecarlo"
    max_samples = 10000
    epochs = 10
    network = "resnet18"
    include_zap = True
    draw_matrix = False
    selected_datasets = all_datasets
    path = os.path.dirname(__file__)

    dataset_descriptors = []
    hessians = []
    embeddings = []
    dataset_names = []

    if include_zap:
        meta_features_path = os.path.join(path, "..", "experiments", "output", "dataset-meta-features", "zap")
        for dataset_name in selected_datasets:
            for id in range(1, 16):
                embedding, descriptors = get_embedding(dataset_name, "zap",
                                                        augmentation_id=id,
                                                        epochs=epochs,
                                                        max_samples=max_samples,
                                                        method=method,
                                                        network=network)
                embeddings.append(embedding)
                dataset_names.append(dataset_name+"-"+str(id))
                hessians.append(embedding.hessian.tolist())
                dataset_descriptors.append(descriptors)

    else:
        meta_features_path = os.path.join(path, "..", "experiments", "output", "dataset-meta-features", "meta-album")
        for version in ["micro", "mini", "extended"]:
            for set in AVAILABLE_SETS:
                for dataset in AVAILABLE_MTLBM_DATASETS[set]:
                    dataset_name = f"{version}/{set}/{dataset}"
                    print(dataset_name)
                    try:
                        torch.cuda.empty_cache()
                        embedding, descriptors = get_embedding(dataset_name, "mtlbm",
                                                                epochs=epochs,
                                                                max_samples=max_samples,
                                                                method=method,
                                                                network=network)
                        embeddings.append(embedding)
                        dataset_names.append(dataset_name)
                        hessians.append(embedding.hessian.tolist())
                        dataset_descriptors.append(descriptors)
                    except Exception as e:
                        print(e)
                        print(f"Dataset {dataset_name} not found")

    os.makedirs(meta_features_path, exist_ok=True)
    meta_features_info = {"hessians": hessians,
                        "dataset_names": dataset_names,
                           "dataset_descriptors": dataset_descriptors}

    #write meta features as json
    with open(os.path.join(meta_features_path, "meta-features.json"), "w") as f:
        json.dump(meta_features_info, f)

    if draw_matrix:
        task_similarity.plot_distance_matrix(embeddings, dataset_names)
        temp_path = os.path.join(path, "..", "plots", "similarity", method, network, str(max_samples))
        os.makedirs(temp_path, exist_ok=True)
        plt.savefig(os.path.join(temp_path, "plot-10epochs_zap.png"))