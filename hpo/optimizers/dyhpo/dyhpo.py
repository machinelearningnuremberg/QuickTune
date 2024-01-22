from copy import deepcopy
import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import cat
import gpytorch
import copy


class GPRegressionModel(gpytorch.models.ExactGP):
    """
    A simple GP model.
    """
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        """
        Constructor of the GPRegressionModel.
        Args:
            train_x: The initial train examples for the GP.
            train_y: The initial train labels for the GP.
            likelihood: The likelihood to be used.
        """
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DyHPO:
    """
    The DyHPO DeepGP model.
    """
    def __init__(
        self,
        device: torch.device,
        dataset_name: str = 'unknown',
        output_path: str = '.',
        seed: int = 11,
        output_dim = None,
        feature_extractor = None,
        batch_size = 64,
        nr_epochs = 1000,
        early_stopping_patience = 10,
        learning_rate = 0.001,
        include_metafeatures = True,
        meta_checkpoint = None
    ):
        """
        The constructor for the DyHPO model.
        Args:

            device: The device where the experiments will be run on.
            dataset_name: The name of the dataset for the current run.
            output_path: The path where the intermediate/final results
                will be stored.
            seed: The seed that will be used to store the checkpoint
                properly.
        """
        super(DyHPO, self).__init__()
        self.feature_extractor = feature_extractor
        self.original_feature_extractor = copy.deepcopy(self.feature_extractor)
        self.batch_size = batch_size
        self.nr_epochs = nr_epochs
        self.early_stopping_patience = early_stopping_patience
        self.refine_epochs = 50
        self.learning_rate = learning_rate
        self.dev = device
        self.seed = seed
        self.output_dim = output_dim
        self.model, self.likelihood, self.mll = \
            self.get_model_likelihood_mll(
                self.output_dim
            )

        self.model.to(self.dev)
        self.likelihood.to(self.dev)
        self.feature_extractor.to(self.dev)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.learning_rate}],
        )

        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_nr_points = 10
        # keeping track of the total hpo iterations. It will be used during the optimization
        # process to switch from fully training the model, to refining.
        self.iterations = 0
        # flag for when the optimization of the model should start from scratch.
        self.restart = True

        self.logger = logging.getLogger(__name__)

        self.checkpoint_path = os.path.join(
            output_path,
            'checkpoints',
            f'{dataset_name}',
            f'{self.seed}',
        )

        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.metafeatures = None
        self.checkpoint_file = os.path.join(
            self.checkpoint_path,
            'checkpoint.pth'
        )
        self.cost_aware = False
        self.include_metafeatures = include_metafeatures
        self.meta_checkpoint = meta_checkpoint

    def set_cost_predictor(self, cost_predictor):
        self.cost_predictor = cost_predictor
        self.cost_aware = True


    def restart_optimization(self):
        """
        Restart the surrogate model from scratch.
        """
        if self.meta_checkpoint is None:
            self.feature_extractor = copy.deepcopy(self.original_feature_extractor).to(self.dev)
            self.model, self.likelihood, self.mll = \
                self.get_model_likelihood_mll(
                    self.output_dim,
                )

        else:
            self.load_checkpoint(self.meta_checkpoint)

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.learning_rate}]
        )

    def get_model_likelihood_mll(
        self,
        train_size: int,
    ) -> Tuple[GPRegressionModel, gpytorch.likelihoods.GaussianLikelihood, gpytorch.mlls.ExactMarginalLogLikelihood]:
        """
        Called when the surrogate is first initialized or restarted.
        Args:
            train_size: The size of the current training set.
        Returns:
            model, likelihood, mll - The GP model, the likelihood and
                the marginal likelihood.
        """
        train_x = torch.ones(train_size, train_size).to(self.dev)
        train_y = torch.ones(train_size).to(self.dev)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.dev)
        model = GPRegressionModel(train_x=train_x, train_y=train_y, likelihood=likelihood).to(self.dev)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(self.dev)

        return model, likelihood, mll


    def train_step(self, X_train, y_train, train_budgets, train_curves, epoch_nr, meta_features=None):
        nr_examples_batch = X_train.size(dim=0)
        # if only one example in the batch, skip the batch.
        # Otherwise, the code will fail because of batchnorm
        if nr_examples_batch == 1:
            return False

        # Zero backprop gradients
        self.optimizer.zero_grad()

        if meta_features is not None:
            projected_x = self.feature_extractor(X_train, train_budgets, train_curves, meta_features)
        else:
            projected_x = self.feature_extractor(X_train, train_budgets, train_curves)
        self.model.set_train_data(projected_x, y_train, strict=False)
        output = self.model(projected_x)
        training_errored = False

        try:
            # Calc loss and backprop derivatives
            loss = -self.mll(output, self.model.train_targets)
            loss_value = loss.detach().to('cpu').item()
            mse = gpytorch.metrics.mean_squared_error(output, self.model.train_targets)
            self.logger.debug(
                f'Epoch {epoch_nr} - MSE {mse:.5f}, '
                f'Loss: {loss_value:.3f}, '
                f'lengthscale: {self.model.covar_module.base_kernel.lengthscale.item():.3f}, '
                f'noise: {self.model.likelihood.noise.item():.3f}, '
            )
            loss.backward()
            self.optimizer.step()
        except Exception as training_error:
            self.logger.error(f'The following error happened while training: {training_error}')
            # An error has happened, trigger the restart of the optimization and restart
            # the model with default hyperparameters.
            self.restart = True
            training_errored = True

        return training_errored


    def train_pipeline(self, data: Dict[str, torch.Tensor], load_checkpoint: bool = False):
        """
        Train the surrogate model.
        Args:
            data: A dictionary which has the training examples, training features,
                training budgets and in the end the training curves.
            load_checkpoint: A flag whether to load the state from a previous checkpoint,
                or whether to start from scratch.
        """
        self.iterations += 1
        self.logger.debug(f'Starting iteration: {self.iterations}')
        # whether the state has been changed. Basically, if a better loss was found during
        # this optimization iteration then the state (weights) were changed.
        weights_changed = False

        if load_checkpoint:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                self.logger.error(f'No checkpoint file found at: {self.checkpoint_file}'
                                  f'Training the GP from the beginning')

        self.model.train()
        self.likelihood.train()
        self.feature_extractor.train()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.learning_rate}],
        )

        X_train = data['X_train']
        train_budgets = data['train_budgets']
        train_curves = data['train_curves']
        y_train = data['y_train']

        initial_state = self.get_state()
        training_errored = False

        if self.restart:
            self.restart_optimization()
            nr_epochs = self.nr_epochs
            # 2 cases where the statement below is hit.
            # - We are switching from the full training phase in the beginning to refining.
            # - We are restarting because our refining diverged
            if self.initial_nr_points <= self.iterations:
                self.restart = False
        else:
            nr_epochs = self.refine_epochs

        # where the mean squared error will be stored
        # when predicting on the train set
        mse = 0.0

        for epoch_nr in range(0, nr_epochs):
            if self.include_metafeatures:
                batch_metafeatures = self.metafeatures.repeat(X_train.size(dim=0), 1)
            else:
                batch_metafeatures = None
            training_errored = self.train_step(X_train, y_train, train_budgets, train_curves, epoch_nr, batch_metafeatures)


        """
        # metric too high, time to restart, or we risk divergence
        if mse > 0.15:
            if not self.restart:
                self.restart = True
        """
        if training_errored:
            self.save_checkpoint(initial_state)
            self.load_checkpoint()

    def predict_pipeline(
        self,
        train_data: Dict[str, torch.Tensor],
        test_data: Dict[str, torch.Tensor],
        to_numpy: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            train_data: A dictionary that has the training
                examples, features, budgets and learning curves.
            test_data: Same as for the training data, but it is
                for the testing part and it does not feature labels.
        Returns:
            means, stds: The means of the predictions for the
                testing points and the standard deviations.
        """
        self.model.eval()
        self.feature_extractor.eval()
        self.likelihood.eval()

        batch_metafeatures_train = self.metafeatures.repeat(train_data['X_train'].size(dim=0), 1)
        batch_metafeatures_test = self.metafeatures.repeat(test_data['X_test'].size(dim=0), 1)

        batch_metafeatures_train_for_surrogate = None
        batch_metafeatures_test_for_surrogate = None

        if self.include_metafeatures:
            batch_metafeatures_train_for_surrogate = batch_metafeatures_train
            batch_metafeatures_test_for_surrogate = batch_metafeatures_test


        with torch.no_grad(): # gpytorch.settings.fast_pred_var():
            projected_train_x = self.feature_extractor(
                train_data['X_train'],
                train_data['train_budgets'],
                train_data['train_curves'],
                train_data.get('train_metafeatures', batch_metafeatures_train_for_surrogate)
            )
            self.model.set_train_data(inputs=projected_train_x, targets=train_data['y_train'], strict=False)
            projected_test_x = self.feature_extractor(
                test_data['X_test'],
                test_data['test_budgets'],
                test_data['test_curves'],
                test_data.get('test_metafeatures', batch_metafeatures_test_for_surrogate)
            )
            preds = self.likelihood(self.model(projected_test_x))

            if self.cost_aware:
                costs = self.cost_predictor(
                    test_data['X_test'],
                    test_data['test_budgets'],
                    test_data['test_curves'],
                     batch_metafeatures_test
                )
            else:
                costs = None
        if to_numpy:
            means = preds.mean.detach().to('cpu').numpy().reshape(-1, )
            stds = preds.stddev.detach().to('cpu').numpy().reshape(-1, )
        else:
            means = preds.mean.reshape(-1, )
            stds = preds.stddev.reshape(-1, )
        return means, stds, costs

    def load_checkpoint(self, checkpoint_file = None):
        """
        Load the state from a previous checkpoint.
        """
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_file
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['gp_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.original_feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    def save_checkpoint(self, state: Dict =None,
                              checkpoint_file: str = None):
        """
        Save the given state or the current state in a
        checkpoint file.
        Args:
            state: The state to save, if none, it will
            save the current state.
        """
        if checkpoint_file is None:
            checkpoint_file = self.checkpoint_file


        if state is None:
            torch.save(
                self.get_state(),
                checkpoint_file,
            )
        else:
            torch.save(
                state,
                checkpoint_file,
            )

    def get_state(self) -> Dict[str, Dict]:
        """
        Get the current state of the surrogate.
        Returns:
            current_state: A dictionary that represents
                the current state of the surrogate model.
        """
        current_state = {
            'gp_state_dict': deepcopy(self.model.state_dict()),
            'feature_extractor_state_dict': deepcopy(self.feature_extractor.state_dict()),
            'likelihood_state_dict': deepcopy(self.likelihood.state_dict()),
        }

        return current_state

class ConvNet(nn.Module):
    def __init__(self, input_dim=1, output_dim=16):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 8, 3, 1, padding="same")
        self.conv2 = nn.Conv1d(8, 8, 3, 1, padding="same")
        self.fc1 = nn.Linear(200, output_dim)
        self.dropout1 = nn.Dropout1d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.MaxPool1d(2)(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = nn.ReLU()(x)
        x = self.fc1(x)
        output = nn.ReLU()(x)

        return output


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        # create the input layer
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # create hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            hidden_layer = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
            self.hidden_layers.append(hidden_layer)

        # create the output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # apply input layer
        x = self.input_layer(x)
        x = nn.ReLU()(x)

        # apply hidden layers
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.ReLU()(x)

        # apply output layer
        x = self.output_layer(x)
        return x

class FeatureExtractor(nn.Module):
    def __init__(self, configuration = {},
                        input_dim_hps = None,
                        output_dim = 32,
                        input_dim_curves = 1,
                        output_dim_curves=16,
                        hidden_dim=128,
                        input_dim_metafeatures=7684,
                        output_dim_metafeatures=0,
                        encoder_dim_ranges = None,
                        encoder_num_layers = 1):
        super().__init__()

        self.input_dim = configuration.get("input_dim_hps", input_dim_hps) \
                        + configuration.get("output_dim_curves", output_dim_curves) \
                        + configuration.get("output_dim_metafeatures", output_dim_metafeatures)
        self.output_dim = configuration.get("output_dim", output_dim)
        self.hidden_dim = configuration.get("hidden_dim", hidden_dim)
        self.input_dim_curves = configuration.get("input_dim_curves", input_dim_curves)
        self.output_dim_curves = configuration.get("output_dim_curves", output_dim_curves)
        self.output_dim_metafeatures = configuration.get("output_dim_metafeatures", output_dim_metafeatures)

        assert self.input_dim is not None, "input_dim_hps must be specified"

        self.encoder_dim_ranges = encoder_dim_ranges
        self.encoders = torch.nn.ModuleList([])
        if encoder_dim_ranges is not None:
            new_input_dim = 0
            for dim_range in encoder_dim_ranges:
                self.encoders.append(MLP(dim_range[1] - dim_range[0], [hidden_dim] * encoder_num_layers, hidden_dim))
                new_input_dim += hidden_dim
            self.input_dim = new_input_dim + self.output_dim_curves + self.output_dim_metafeatures +1

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.curve_embedder = ConvNet(input_dim=self.input_dim_curves,
                                      output_dim=self.output_dim_curves)
        self.fc_metafeatures = nn.Linear(input_dim_metafeatures, self.output_dim_metafeatures)




    def forward(self, hps, budgets, curves, metafeatures=None):

        budgets = torch.unsqueeze(budgets, dim=1)
        if curves.dim() == 2:
            curves = torch.unsqueeze(curves, dim=1)

        if self.encoder_dim_ranges is not None:
            x = []
            for i, encoder in enumerate(self.encoders):
                dim1, dim2 = self.encoder_dim_ranges[i]
                x.append(encoder(hps[:, dim1:dim2]))
            x.append(budgets)
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat((hps, budgets), dim=1)

        curves_emb = self.curve_embedder(curves)

        if metafeatures is not None:
            metafeatures_emb = self.fc_metafeatures(metafeatures)
            x = torch.cat([x, metafeatures_emb], dim=1)

        x = torch.cat([x, curves_emb], dim=1)

        x = self.fc1(x)
        x = nn.LeakyReLU()(x)
        output = self.fc3(x)
        #output = nn.ReLU()(x)
        return output

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class CostPredictor(nn.Module):
    def __init__(self, input_dim_hps = 71,
                        output_dim_feature_extractor = 1,
                        input_dim_curves = 1,
                        hidden_dim=64,
                        output_dim_metafeatures=8,
                        input_dim_metafeatures=7684
                 ):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim_hps = input_dim_hps,
                                                  output_dim = output_dim_feature_extractor,
                                                  input_dim_curves = input_dim_curves,
                                                  hidden_dim=hidden_dim,
                                                  output_dim_metafeatures=output_dim_metafeatures,
                                                  input_dim_metafeatures=input_dim_metafeatures)
        self.output_dim_feature_extractor = output_dim_feature_extractor
        self.fc1 = nn.Linear(output_dim_feature_extractor, 1)
        self.act = nn.ReLU()

    def forward(self, hps, budgets, curves, metafeatures=None):
        x = self.feature_extractor(hps, budgets, curves, metafeatures)
        x = self.act(x)
        x = self.act(self.fc1(x))
        return x


class CostPredictorTrainer:
    def __init__(self, cost_predictor,
                        metadataset,
                        learning_rate = 0.0001,
                        train_iter = 1000,
                        test_iter = 10,
                        test_freq = 20,
                        batch_size = 32,
                        num_splits = 5,
                        train_splits = (0,1,2),
                        test_splits = (3,),
                        val_splits = (4,),
                        max_budget = 50,
                        seed = 42,
                        checkpoint_path = None,
                        curve_type = "perf",
                        device = "cuda"):

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
        self.datasets = self.get_splits(self,train_splits=train_splits,
                                        test_splits=test_splits,
                                        val_splits=val_splits,
                                        seed=seed)

        self.rnd_gens = {'train': np.random.default_rng(seed),
                            'test': np.random.default_rng(seed),
                            'val': np.random.default_rng(seed)}
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.finetuning = False
        self.info_dict = None
        self.curve_type = curve_type

    @staticmethod
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

    def get_batch_for_finetuning(self):

        #TODO: confirm that this is correct when the curve to train on is not the performance
        assert self.info_dict is not None, "No info dict available"
        info_hps = self.info_dict['hp']
        info_budgets = self.info_dict['epochs']
        info_curves = self.info_dict['overhead']
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
            curves.append(temp_curve+[0]*(self.max_budget-len(temp_curve)))
            targets.append(aggregated_info[hp_index][-1])
            metafeatures.append(self.metadataset.get_metafeatures().tolist())

        curves = torch.FloatTensor(curves).to(self.device)/100
        hps = torch.FloatTensor(hps).to(self.device)
        metafeatures = torch.FloatTensor(metafeatures).to(self.device)/10000
        budgets = torch.FloatTensor(budgets).to(self.device)/self.max_budget
        targets = torch.FloatTensor(targets).to(self.device)/100

        return  hps, targets, budgets, curves, metafeatures

    def get_batch(self, mode = 'train', dataset = None):

        targets = []
        curves = []
        hps = []
        metafeatures = []
        budgets = []

        if dataset is None:
            dataset = self.rnd_gens[mode].choice(self.datasets[mode])

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

            curve_eval_time = self.metadataset.get_curve(hp_index=hp_index, budget=budget, curve_name="eval_time")
            curve_train_time = self.metadataset.get_curve(hp_index=hp_index, budget=budget, curve_name="train_time")
            curve_time = [curve_eval_time[i] + curve_train_time[i] for i in range(len(curve_eval_time))]

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

        curves = torch.FloatTensor(curves).to(self.device)/100
        hps = torch.FloatTensor(hps).to(self.device)
        metafeatures = torch.FloatTensor(metafeatures).to(self.device)/10000
        budgets = torch.FloatTensor(budgets).to(self.device)/self.max_budget
        targets = torch.FloatTensor(targets).to(self.device)/100

        return hps, targets, budgets, curves, metafeatures

    def save_checkpoint(self):
        torch.save(self.cost_predictor.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.cost_predictor.load_state_dict(torch.load(self.checkpoint_path))

    def train(self):
        print("Training cost predictor...")
        self.cost_predictor.train()
        self.cost_predictor.to(self.device)
        optimizer = torch.optim.Adam(self.cost_predictor.parameters(), lr=self.learning_rate)
        for i in range(self.train_iter):
            optimizer.zero_grad()
            if self.finetuning:
                hps, targets, budgets, curves, metafeatures = self.get_batch_for_finetuning()
            else:
                hps, targets, budgets, curves, metafeatures = self.get_batch(mode='train')
            predictions = self.cost_predictor(hps, budgets, curves, metafeatures)
            loss = self.criterion(predictions.reshape(targets.shape), targets)
            loss.backward()
            optimizer.step()
            if i % self.test_freq == 0:
                self.val()
        self.cost_predictor.eval()

    def val(self):
        self.cost_predictor.eval()
        self.cost_predictor.to(self.device)
        losses = []
        for i in range(self.test_iter):
            if self.finetuning:
                hps, targets, budgets, curves, metafeatures = self.get_batch_for_finetuning()
            else:
                hps, targets, budgets, curves, metafeatures = self.get_batch(mode='val')
            predictions = self.cost_predictor(hps, budgets, curves, metafeatures)
            loss = self.criterion(predictions.reshape(targets.shape), targets)
            losses.append(loss.item())
        print("Val loss: ", np.mean(losses))
        self.cost_predictor.train()

class MetaTrainer:

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
        metafeatures = torch.FloatTensor(metafeatures).to(self.device)/1000
        budgets = torch.FloatTensor(budgets).to(self.device)/self.max_budget
        targets = torch.FloatTensor(targets).to(self.device)/100

        return hps, targets, budgets, curves, metafeatures

    def meta_train(self):

        self.model.model.train()
        self.model.likelihood.train()
        self.model.feature_extractor.train()
        min_eval_error = 100

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
                    print("Saving checkpoint in meta-learning with error: ", min_eval_error)

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

        print("Val error: ", cum_error / self.test_iter)

        self.model.model.train()
        self.model.likelihood.train()
        self.model.feature_extractor.train()

        return cum_error / self.test_iter

