import openml
import os
import argparse

#openml.config.cache_directory = os.path.expanduser('/work/dlclarge1/pineda-zap-data/tfds-icgen/meta-album/')
#openml.config.cache_directory = os.path.expanduser('/home/sebastian/Documents/Code/AutoFinetune/data/micro/')

dataset_ids_per_split_and_version = {
                            "micro" :   {   "set0": [44241, 44238, 44239, 44242, 44237, 44246, 44245, 44244, 44240, 44243],
                                            "set1": [44313, 44248, 44249, 44314, 44312, 44315, 44251, 44250, 44247, 44252 ],
                                            "set2": [44275, 44276, 44272, 44273, 44278, 44277, 44279, 44274, 44271, 44280 ]
                                },
                            "mini"  :   {   "set0": [44285, 44282, 44283, 44286, 44281, 44290, 44289, 44288, 44284, 44287],
                                            "set1": [44298, 44292, 44293, 44299, 44297, 44300, 44295, 44294, 44291, 44296],
                                            "set2": [44305, 44306, 44302, 44303, 44308, 44307, 44309, 44304, 44301, 44310]
                                },
                            "extended":  {
                                            "set0": [44320, 44317, 44318, 44321, 44316, 44324, 44323, 44322, 44319],
                                            "set1": [44331, 44326, 44327, 44332, 44330, 44333, 44329, 44328, 44325],
                                            "set2": [44338, 44340, 44335, 44336, 44342, 44341, 44343, 44337, 44334]
                                }
                            }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data')
    parser.add_argument("--test", action="store_true", default=False)

    args = parser.parse_args()

    versions = ["micro", "mini", "extended"]
    splits = ["set0", "set1", "set2"]
    os.makedirs(args.data_path, exist_ok=True)
    openml.config.cache_directory = os.path.expanduser(args.data_path)

    if args.test:
        dataset_id = dataset_ids_per_split_and_version["mini"]["set0"][0]
        print("Downloading dataset: ", dataset_id)
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)

    for version in versions:
        for split in splits:
            for dataset_id in dataset_ids_per_split_and_version[version][split]:
                print("Downloading dataset: ", dataset_id)
                dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_all_files=True)

             



