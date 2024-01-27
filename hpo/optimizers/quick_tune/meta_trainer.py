import torch
from torch import nn
import numpy as np
import logging

class MetaTrainer:

    "Meta learns the performance predictor"

    def __init__(self, model,
                        metadataset,
                        device,
                        train_iter = 100,
                        test_iter = 50,
                        test_freq = 20,
                        batch_size = 32,
                        num_splits = 5,
                        learning_rate = 0.001,
                        train_splits = (0,1,2),
                        test_splits = (3,),
                        val_splits = (4,),
                        max_budget = 50,
                        seed = 42,
                        with_scheduler = True,
                        include_metafeatures = False
                 ):

        self.model = model
        self.metadataset = metadataset
        self.model.learning_rate = learning_rate
        self.model.restart_optimization()
        self.test_freq = test_freq
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.test_criterion = nn.MSELoss()
        self.datasets = self.metadataset.get_datasets()
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        self.max_budget = max_budget
        self.with_scheduler = with_scheduler
        self.include_metafeatures = include_metafeatures
        self.datasets = self.get_splits(train_splits=train_splits,
                                        test_splits=test_splits,
                                        val_splits=val_splits,
                                        seed=seed)
        self.rnd_gens = {'train': np.random.default_rng(seed),
                            'test': np.random.default_rng(seed),
                            'val': np.random.default_rng(seed)}

        self.logger = logging.getLogger('Peformance Predictor Trainer')
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

    def get_splits(self,
                   datasets=None,
                   seed=42,
                   train_splits=(0, 1, 2),
                   test_splits=(3,),
                   val_splits=(4,)):

        if datasets is None:
            datasets = self.datasets

        rnd_gen = np.random.default_rng(seed)
        rnd_gen.shuffle(datasets)

        splits = {'train': [],
                  'val': [],
                  'test': []}
        num_splits = len(train_splits) + len(test_splits) + len(val_splits)
        for i, dataset in enumerate(datasets):
            split_id = i % num_splits
            if split_id in train_splits:
                splits['train'].append(dataset)
            elif split_id in test_splits:
                splits['test'].append(dataset)
            elif split_id in val_splits:
                splits['val'].append(dataset)
            else:
                raise ValueError("Dataset not assigned to any split")
        return splits

    def get_batch(self, mode = 'train', dataset = None):
        targets = []
        curves = []
        hps = []
        metafeatures = []
        budgets = []

        if dataset is None:
            dataset = self.rnd_gens[mode].choice(self.datasets[mode])
        #self.metadataset.set_dataset_name(dataset)
        if self.metadataset.set == "zap":
            augmentation_id = self.rnd_gens[mode].integers(1,16)
        else:
            augmentation_id = None
        self.metadataset.set_dataset_name(dataset, augmentation_id=augmentation_id)


        for _ in range(self.batch_size):
            num_hps_candidates = self.metadataset.get_num_hyperparameters()
            hp_index = self.rnd_gens[mode].integers(num_hps_candidates)
            budget = self.rnd_gens[mode].integers(1,self.max_budget)
            temp_curves = []
            curve = self.metadataset.get_curve(hp_index=hp_index, budget=budget)
            budget = len(curve)
            target = curve[-1]
            curve = curve[:-1]
            temp_curves.append(curve + [0] * (self.max_budget - len(curve)))
            targets.append(target)
            curves.append(temp_curves)
            budgets.append(budget)
            hps.append(self.metadataset.get_hyperparameters(hp_index=hp_index).tolist())
            metafeatures.append(self.metadataset.get_metafeatures().tolist())

        curves = torch.FloatTensor(curves).to(self.device)/100
        hps = torch.FloatTensor(hps).to(self.device)
        metafeatures = torch.FloatTensor(metafeatures).to(self.device)/1000 #TODO: Proper scaling
        budgets = torch.FloatTensor(budgets).to(self.device)/self.max_budget
        targets = torch.FloatTensor(targets).to(self.device)/100

        return hps, targets, budgets, curves, metafeatures

    def meta_train(self):

        self.model.model.train()
        self.model.likelihood.train()
        self.model.feature_extractor.train()
        min_eval_error = np.inf

        if self.with_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model.optimizer, self.train_iter, eta_min=1e-7)
        else:
            scheduler = None

        self.model.save_checkpoint()
        for i in range(self.train_iter):
            hps, targets, budgets, curves, metafeatures = self.get_batch()

            if not self.include_metafeatures:
                metafeatures = None

            self.model.train_step(hps, targets, budgets, curves, i, metafeatures)
            if i % self.test_freq == 0:
                val_error = self.meta_val()

                if val_error < min_eval_error:
                    min_eval_error = val_error
                    self.model.save_checkpoint()
                    self.logger.info(f"Saving checkpoint in meta-learning with error: {min_eval_error}")

            if scheduler is not None:
                scheduler.step()

        final_val_error = self.final_meta_val()
        return final_val_error

    def final_meta_val(self):
        self.model.load_checkpoint()
        return self.meta_val()

    def meta_val(self):
        self.model.model.eval()
        self.model.likelihood.eval()
        self.model.feature_extractor.eval()

        #set seed of random generator
        self.rnd_gens["val"] = np.random.default_rng(self.seed)

        cum_error = 0
        for i in range(self.test_iter):
            batch_train = self.get_batch('val')
            batch_test = self.get_batch('val', self.metadataset.dataset_name)

            batch_train_name = ["X_train", "y_train", "train_budgets", "train_curves", "train_metafeatures"]
            batch_test_name = ["X_test", "y_test", "test_budgets", "test_curves", "test_metafeatures"]

            batch_train = dict(zip(batch_train_name, batch_train))
            batch_test = dict(zip(batch_test_name, batch_test))

            if not self.include_metafeatures:
                batch_train["train_metafeatures"] = None
                batch_test["test_metafeatures"] = None

            means, stds, costs = self.model.predict_pipeline(batch_train, batch_test, to_numpy=False)
            y_test = batch_test["y_test"]
            with torch.no_grad():
                cum_error += self.test_criterion(y_test, means.reshape(y_test.shape))

        val_loss = cum_error / self.test_iter
        self.logger.info(f"Val loss: {val_loss}")

        self.model.model.train()
        self.model.likelihood.train()
        self.model.feature_extractor.train()

        return cum_error / self.test_iter

