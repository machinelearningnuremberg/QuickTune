# QuickTune

## Prepare environment
Create environment and install requirements:

```bash
conda -n quick_tune python=3.9
conda activate quick_tune
pip install -r requirements_qt.txt
```

Install torch and gpytorch version:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install gpytorch -c gpytorch
```



## Fine-tune Network (without HPO)


Download a dataset. In this example, we will use a dataset from meta-album.

```bash
mkdir data && cd data
mkdir mtlbm && cd mtlbm
wget https://rewind.tf.uni-freiburg.de/index.php/s/pGyowo3WBp7f33S/download/PLT_VIL_Micro.zip
unzip PLT_VIL_Micro.zip
```

You can fine-tune network by providing any hyperparameter as follows:

```bash
mkdir output 
python finetune.py data \
					--model dla46x_c \
					--pct_to_freeze 0.8\
					--dataset "mtlb/PLT_VIL_Micro"\
					--train-split train \
					--val-split val  \
					--experiment test_experiment \
					--output output \
					--pretrained \
					--num_classes 20\
					--epochs 50
```


## Run Quick-Tune on Meta-dataset

Download QuickTune meta-dataset:

```bash
mkdir data && cd data
wget https://rewind.tf.uni-freiburg.de/index.php/s/oMxC5sfrkA53ESo/download/qt_metadataset.zip
unzip QT_metadataset.zip
```

Run example:
```
mkdir output
python bash_scripts/run_example.sh
```


## Run on a new dataset

For running on a new dataset, make sure to wrap the folder as specified in Timm Library. You try on included datasets such as Inaturalist:

```
python hpo/optimizers/quicktune/user_interface.py --data_path datasets/inat 
						--dataset_name torch/inaturalist 
						--num_classes 13 
						--num_channels 3 
						--image_size 128 
						--verbose 
						--dataset_size 2700000 
						--train_split kingdom/2021_train_mini 
						--val_split kingdom/validation 
						--experiment_id  qt-test
						--optimizer_name qt-extended 
						--budget_limit 88000
```


