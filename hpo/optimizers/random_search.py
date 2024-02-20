import numpy as np


class RandomSearchOptimizer:

    def __init__(self, metadataset, seed):
        self.hp_candidates = (
            metadataset.get_hyperparameters_candidates().values.tolist()
        )
        self.metadataset = metadataset
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.converged_configs = []

    def suggest(self):
        hp_index = self.rng.randint(0, len(self.hp_candidates))
        budget = self.metadataset.get_curve_len(hp_index)
        return hp_index, budget

    def observe(self, hp_index, budget, performance_curve, observed_cost):
        return 0
