from pathlib import Path
import yaml
import os

N_AUGMENTATIONS = 15

# for coil100, throws a KeyError (changed target key in version 2) --> replace it by caltech101
# emnist/mnist -> deep_weeds
# emnist/byclass -> oxford_iiit_pet
# emnist/balanced -> visual_domain_decathlon/dtd
# perhaps add: sun397, food101, pets
all_datasets = ['cifar100', 'cycle_gan/vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb/devanagari',
                'cmaterdb/bangla', 'mnist', 'horses_or_humans', 'kmnist', 'cycle_gan/horse2zebra', 'cycle_gan/facades',
                'cycle_gan/apple2orange', 'imagenet_resized/32x32', 'cycle_gan/maps', 'omniglot', 'imagenette', 'oxford_iiit_pet',
                'svhn_cropped', 'colorectal_histology', 'caltech101', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers',
                'cycle_gan/ukiyoe2photo', 'cassava', 'fashion_mnist', 'deep_weeds', 'cmaterdb/telugu', 'malaria', 'eurosat/rgb',
                'visual_domain_decathlon/dtd', 'cars196', 'cycle_gan/iphone2dslr_flower', 'cycle_gan/summer2winter_yosemite', 'cats_vs_dogs']

# Same as the AutoFolio inner CV splits
train_splits = [['cifar100', 'cycle_gan/vangogh2photo', 'uc_merced', 'cifar10', 'cmaterdb/devanagari', 'cmaterdb/bangla', 'mnist'],
                ['horses_or_humans', 'kmnist', 'cycle_gan/horse2zebra', 'cycle_gan/facades', 'cycle_gan/apple2orange', 'cycle_gan/maps', 'imagenette'],
                ['oxford_iiit_pet', 'svhn_cropped', 'caltech101', 'stanford_dogs', 'rock_paper_scissors', 'tf_flowers', 'cycle_gan/ukiyoe2photo'],
                ['cassava', 'fashion_mnist', 'deep_weeds', 'cmaterdb/telugu', 'malaria', 'eurosat/rgb', 'visual_domain_decathlon/dtd'],
                ['cars196', 'cycle_gan/iphone2dslr_flower', 'cycle_gan/summer2winter_yosemite', 'cats_vs_dogs', 'imagenet_resized/32x32', 'omniglot', 'colorectal_histology']]

# Domains
objects = ['cifar100', 'cifar10', 'horses_or_humans', 'cycle_gan/horse2zebra', 'cycle_gan/facades',
           'cycle_gan/apple2orange', 'imagenette', 'caltech101', 'stanford_dogs', 'rock_paper_scissors',
           'tf_flowers', 'cassava', 'fashion_mnist', 'cars196', 'cats_vs_dogs', 'imagenet_resized/32x32',
           ]

ocr = ['cmaterdb/devanagari', 'cmaterdb/bangla', 'mnist', 'kmnist',
       'cmaterdb/telugu', 'omniglot', 'svhn_cropped']

medical = ['colorectal_histology', 'malaria']
aerial = ['uc_merced', 'cycle_gan/maps', 'eurosat/rgb']
other = ['cycle_gan/ukiyoe2photo', 'cycle_gan/vangogh2photo', 'cycle_gan/summer2winter_yosemite', 'cycle_gan/iphone2dslr_flower']

GROUPS = {
    'all':     all_datasets,
    'objects': objects,
    'ocr':     ocr,
    'medical': medical,
    'aerial':  aerial,
    'other':   other
    }

if __name__ == '__main__':
    
    for group, elements in GROUPS.items():
        print('Dataset group {} contains {} dataset(s). These are ->'.format(group, len(elements)))
        print('\n'.join(elements))
        print('=' * 60)
    
    import pandas as pd
    import numpy as np
    
    folds = [float(i + 1) for i, datasets in enumerate(train_splits) for _ in datasets for n in range(N_AUGMENTATIONS)]
    train_splits_index = []
    for k in train_splits:
        train_splits_index += k
    
    train_splits_index_augmented = [str(n) + "-" + dataset for dataset in train_splits_index for n in range(N_AUGMENTATIONS)]
    folds_df = pd.DataFrame(folds, index=train_splits_index_augmented, columns=["fold"])
    folds_df.to_csv("../../data/meta_dataset/inner_CV_folds.csv", index=True)