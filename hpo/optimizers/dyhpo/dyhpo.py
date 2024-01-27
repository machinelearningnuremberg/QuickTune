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
    The DyHPO DeepGP model. This version of DyHPO also includes a Cost Predictor
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

