import torch
from torch import nn
import numpy as np
import logging

from hpo.optimizers.dyhpo.dyhpo import FeatureExtractor


class CostPredictor(nn.Module):
    def __init__(
        self,
        input_dim_hps=71,
        output_dim_feature_extractor=1,
        input_dim_curves=1,
        hidden_dim=64,
        output_dim_metafeatures=8,
        input_dim_metafeatures=7684,
    ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(
            input_dim_hps=input_dim_hps,
            output_dim=output_dim_feature_extractor,
            input_dim_curves=input_dim_curves,
            hidden_dim=hidden_dim,
            output_dim_metafeatures=output_dim_metafeatures,
            input_dim_metafeatures=input_dim_metafeatures,
        )
        self.output_dim_feature_extractor = output_dim_feature_extractor
        self.fc1 = nn.Linear(output_dim_feature_extractor, 1)
        self.act = nn.ReLU()

    def forward(self, hps, budgets, curves, metafeatures=None):
        x = self.feature_extractor(hps, budgets, curves, metafeatures)
        x = self.act(x)
        x = self.act(self.fc1(x))
        return x


class CostMetaLearner:
    """Meta-learns the cost predictor."""

    def __init__(
        self,
        cost_predictor,
        metadataset,
        learning_rate=0.0001,
        train_iter=1000,
        test_iter=10,
        test_freq=20,
        batch_size=32,
        num_splits=5,
        train_splits=(0, 1, 2),
        test_splits=(3,),
        val_splits=(4,),
        max_budget=50,
        seed=42,
        checkpoint_path=None,
        curve_type="perf",
        device="cuda",
    ):

        self.cost_predictor = cost_predictor
        self.metadataset = metadataset
        self.datasets = self.metadataset.get_datasets()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.test_freq = test_freq
        self.num_splits = num_splits
        self.train_splits = train_splits
        self.test_splits = test_splits
        self.val_splits = val_splits
        self.max_budget = max_budget
        self.seed = seed
        self.criterion = nn.MSELoss()
        self.datasets = self.get_splits(
            self,
            train_splits=train_splits,
            test_splits=test_splits,
            val_splits=val_splits,
            seed=seed,
        )

        self.rnd_gens = {
            "train": np.random.default_rng(seed),
            "test": np.random.default_rng(seed),
            "val": np.random.default_rng(seed),
        }
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.finetuning = False
        self.info_dict = None
        self.curve_type = curve_type

        self.logger = logging.getLogger("Cost Predictor Meta-learner")
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    @staticmethod
    def get_splits(
        self,
        datasets=None,
        seed=42,
        train_splits=(0, 1, 2),
        test_splits=(3,),
        val_splits=(4,),
    ):

        if datasets is None:
            datasets = self.datasets

        rnd_gen = np.random.default_rng(seed)
        rnd_gen.shuffle(datasets)

        splits = {"train": [], "val": [], "test": []}
        num_splits = len(train_splits) + len(test_splits) + len(val_splits)
        for i, dataset in enumerate(datasets):
            split_id = i % num_splits
            if split_id in train_splits:
                splits["train"].append(dataset)
            elif split_id in test_splits:
                splits["test"].append(dataset)
            elif split_id in val_splits:
                splits["val"].append(dataset)
            else:
                raise ValueError("Dataset not assigned to any split")
        return splits

    def get_batch_for_finetuning(self):

        assert self.info_dict is not None, "No info dict available"
        info_hps = self.info_dict["hp"]
        info_budgets = self.info_dict["epochs"]
        info_curves = self.info_dict["overhead"]
        aggregated_info = {}

        hps = []
        targets = []
        curves = []
        budgets = []
        metafeatures = []

        for hp_index, budget, curve in zip(info_hps, info_budgets, info_curves):
            if hp_index not in aggregated_info:
                aggregated_info[hp_index] = []
            aggregated_info[hp_index].append(curve)

            hps.append(self.metadataset.get_hyperparameters(hp_index).tolist())
            budgets.append(budget)
            temp_curve = aggregated_info[hp_index][:-1]
            curves.append(temp_curve + [0] * (self.max_budget - len(temp_curve)))
            targets.append(aggregated_info[hp_index][-1])
            metafeatures.append(self.metadataset.get_metafeatures().tolist())

        curves = torch.FloatTensor(curves).to(self.device) / 100
        hps = torch.FloatTensor(hps).to(self.device)
        metafeatures = torch.FloatTensor(metafeatures).to(self.device) / 10000
        budgets = torch.FloatTensor(budgets).to(self.device) / self.max_budget
        targets = torch.FloatTensor(targets).to(self.device) / 100

        return hps, targets, budgets, curves, metafeatures

    def get_batch(self, mode="train", dataset=None):

        targets = []
        curves = []
        hps = []
        metafeatures = []
        budgets = []

        if dataset is None:
            dataset = self.rnd_gens[mode].choice(self.datasets[mode])

        if self.metadataset.version == "zap":
            augmentation_id = self.rnd_gens[mode].integers(1, 16)
        else:
            augmentation_id = None
        self.metadataset.set_dataset_name(dataset, augmentation_id=augmentation_id)

        for _ in range(self.batch_size):
            num_hps_candidates = self.metadataset.get_num_hyperparameters()
            hp_index = self.rnd_gens[mode].integers(num_hps_candidates)
            budget = self.rnd_gens[mode].integers(1, self.max_budget)
            temp_curves = []

            curve_eval_time = self.metadataset.get_curve(
                hp_index=hp_index, budget=budget, curve_name="eval_time"
            )
            curve_train_time = self.metadataset.get_curve(
                hp_index=hp_index, budget=budget, curve_name="train_time"
            )
            curve_time = [
                curve_eval_time[i] + curve_train_time[i]
                for i in range(len(curve_eval_time))
            ]

            if self.curve_type == "time":
                curve = curve_time
            else:
                curve = self.metadataset.get_curve(hp_index=hp_index, budget=budget)

            target = curve_time[-1]
            curve = curve[:-1]
            temp_curves.append(curve + [0] * (self.max_budget - len(curve)))
            targets.append(target)
            curves.append(temp_curves)
            budgets.append(budget)
            hps.append(self.metadataset.get_hyperparameters(hp_index=hp_index).tolist())
            metafeatures.append(self.metadataset.get_metafeatures().tolist())

        curves = torch.FloatTensor(curves).to(self.device) / 100
        hps = torch.FloatTensor(hps).to(self.device)
        metafeatures = torch.FloatTensor(metafeatures).to(self.device) / 10000
        budgets = torch.FloatTensor(budgets).to(self.device) / self.max_budget
        targets = torch.FloatTensor(targets).to(self.device) / 100

        return hps, targets, budgets, curves, metafeatures

    def save_checkpoint(self):
        torch.save(self.cost_predictor.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.cost_predictor.load_state_dict(torch.load(self.checkpoint_path))

    def meta_train(self):
        self.logger.info("Training cost predictor...")
        self.cost_predictor.train()
        self.cost_predictor.to(self.device)
        optimizer = torch.optim.Adam(
            self.cost_predictor.parameters(), lr=self.learning_rate
        )
        for i in range(self.train_iter):
            optimizer.zero_grad()
            if self.finetuning:
                hps, targets, budgets, curves, metafeatures = (
                    self.get_batch_for_finetuning()
                )
            else:
                hps, targets, budgets, curves, metafeatures = self.get_batch(
                    mode="train"
                )
            predictions = self.cost_predictor(hps, budgets, curves, metafeatures)
            loss = self.criterion(predictions.reshape(targets.shape), targets)
            loss.backward()
            optimizer.step()
            if i % self.test_freq == 0:
                self.meta_val()
        self.cost_predictor.eval()

    def meta_val(self):
        self.cost_predictor.eval()
        self.cost_predictor.to(self.device)
        losses = []
        for i in range(self.test_iter):
            if self.finetuning:
                hps, targets, budgets, curves, metafeatures = (
                    self.get_batch_for_finetuning()
                )
            else:
                hps, targets, budgets, curves, metafeatures = self.get_batch(mode="val")
            predictions = self.cost_predictor(hps, budgets, curves, metafeatures)
            loss = self.criterion(predictions.reshape(targets.shape), targets)
            losses.append(loss.item())
        mean = np.mean(losses)
        self.logger.info(f"Val loss:  {mean}")
        self.cost_predictor.train()
