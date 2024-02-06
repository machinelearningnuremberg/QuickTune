import os
import pandas as pd
import yaml
import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CAT_COLS = ['auto_augment', 'dataset', 'experiment',  'model', 'opt', 'opt_betas', 'sched']
COLS_TO_DROP = [ "project_name", "device", "project"]
NON_CAT_COLS = ['amp', 'batch_size', 'bss_reg', 'cotuning_reg', 'cutmix', 'decay_epochs', 'decay_rate', 'delta_reg',
                'drop', 'linear_probing', 'lr', 'mixup', 'mixup_prob', 'momentum', 'num_classes', 'patience_epochs',
                'pct_to_freeze', 'ra_magnitude', 'ra_num_ops', 'random_augment', 'smoothing', 'sp_reg', 'stoch_norm',
                'trivial_augment', 'warmup_epochs', 'warmup_lr', 'weight_decay', 'max_eval_top1', 'max_eval_top5',
                'curve_length', 'final_batch_size', 'invalid_loss_value', 'max_memory_allocated', 'clip_grad', 'layer_decay']

class QuickTuneMetaDataset:

    def __init__(self, version="micro",
                        path=None,
                        curves_to_load=None,
                        preprocess_args=True,
                        aggregate_data=False,
                        drop_constant_args=True,
                        impute_numerical_args=True,
                        encode_categorical_args=True,
                        standardize_numerical_args=True,
                        load_only_dataset_descriptors=False,
                        verbose=False,
                        model_args_first=True,
                        target_model=None,
                        action_on_model=None,
                        subsample_models_in_hub=None):

        self.drop_constant_args = drop_constant_args
        self.impute_numerical_args = impute_numerical_args
        self.encode_categorical_args = encode_categorical_args
        self.standardize_numerical_args = standardize_numerical_args
        self.version = version
        self.dataset_name = None
        self.preset_metafeatures = None
        self.verbose = verbose
        self.load_only_dataset_descriptors = load_only_dataset_descriptors
        self.model_args_first = model_args_first
        self.target_model = target_model
        self.action_on_model = action_on_model
        self.subsample_models_in_hub = subsample_models_in_hub

        if path is None:
            self.path = "/work/dlclarge1/pineda-aft-curves"
        else:
            self.path = path

        if curves_to_load is None:
            curves_to_load = ["eval_top1", "eval_time", "train_time"]
        if verbose:
            print(f"curves_to_load: {curves_to_load}")

        self.curves_names = ["epoch", "train_loss", "train_head_grad_norm", "train_backbone_grad_norm",\
                              "train_time", "eval_loss", "eval_top1", "eval_top5", "eval_time", "lr"]

        if aggregate_data:
            self.aggregate_curves()
        self.load_curves(curves_to_load)
        self.load_args(preprocess_args)
        self.load_metafeatures()

        if self.version == "zap":
            self.obtain_augmentation_id()

    def obtain_augmentation_id(self):
        augmentation_id = [int(x.split("_")[-2]) for x in self.args_df.index]
        self.args_df["augmentation_id"] = augmentation_id



    def aggregate_curves(self):
        files = os.listdir(f"{self.path}/curves/{self.version}")
        aggregated_curves = {}
        aggregated_args = {}

        for name in self.curves_names:
            aggregated_curves[name] = {}

        for file in files:
            try:
                run = file[:-4]
                with open(f"{self.path}/args/{self.version}/{run}.yaml", 'r') as stream:
                    args = yaml.safe_load(stream)
                curves_data = pd.read_csv(f"{self.path}/curves/{self.version}/{file}", nrows=52, index_col=0)

                if curves_data.shape[0] < 2:
                    if (curves_data["eval_top1"][0] == 0.0) or (np.isnan(curves_data["eval_top1"][0])):
                        continue
                elif (len(curves_data)==26) and (curves_data["eval_top1"][24]-curves_data["eval_top1"][25] > 20)\
                                            and args["amp"]:
                    curves_data.drop([25], inplace=True)

                dataset = args["dataset"]
                if dataset not in aggregated_curves["eval_top1"].keys():
                    for name in self.curves_names:
                        aggregated_curves[name][dataset] = {}

                max_eval_top1 = curves_data["eval_top1"].max()
                max_eval_top5 = curves_data["eval_top5"].max()

                if isinstance(max_eval_top1, float) and isinstance(max_eval_top5, float):
                    args["max_eval_top1"] = max_eval_top1
                    args["max_eval_top5"] = max_eval_top5
                else:
                    continue

                for name in self.curves_names:
                    aggregated_curves[name][dataset][run] = curves_data[name].values.tolist()

                aggregated_args[run] = args
            except Exception as e:
                print(f"Error in {file}")
                print(e)

        for name in self.curves_names:
            with open(f"{self.path}/qt_metadataset/{self.version}/{name}.json", 'w') as outfile:
                json.dump(aggregated_curves[name], outfile)

        with open(f"{self.path}/qt_metadataset/{self.version}/args.json", 'w') as outfile:
                json.dump(aggregated_args, outfile)

    def load_curves(self, curve_names):
        self.curves = {}
        for name in curve_names:
            with open(f"{self.path}/qt_metadataset/{self.version}/{name}.json", 'r') as stream:
                self.curves[name] = json.load(stream)

    def get_superset(self):
        if self.version in ["micro", "mini", "extended"]:
            return "meta-album"
        else:
            return self.version

    def load_metafeatures(self):
        superset = self.get_superset()
        #read json file
        with open(f"{self.path}/qt_metadataset/dataset-meta-features/{superset}/meta-features.json", 'r') as stream:
            metafeatures = json.load(stream)

        if self.load_only_dataset_descriptors:
            metafeatures_df = pd.DataFrame(metafeatures["dataset_descriptors"])
        else:
            metafeatures_df = pd.concat([pd.DataFrame(metafeatures["hessians"]), pd.DataFrame(metafeatures["dataset_descriptors"])], axis=1)

        if superset == "meta-album":
            metafeatures_df.index = ["mtlbm/"+x for x in metafeatures["dataset_names"]]
        else:
            metafeatures_df.index = metafeatures["dataset_names"]
        self.metafeatures = metafeatures_df


    def load_args(self, preprocess=False, file_name="unnormalized_args_table"):

        #check if path exists
        if not os.path.exists(f"{self.path}/qt_metadataset/{self.version}/{file_name}.csv"):
            with open(f"{self.path}/qt_metadataset/{self.version}/args.json", 'r') as stream:
                self.args = json.load(stream)
            self.args_df = pd.DataFrame(self.args).T

            for col in self.args_df.columns:
                try:
                    if isinstance(self.args_df[col][0], int):
                        self.args_df[col] = self.args_df[col].fillna(-1)
                        self.args_df[col] = self.args_df[col].astype(int)
                    elif isinstance(self.args_df[col][0], float):
                        self.args_df[col] = self.args_df[col].fillna(-1)
                        self.args_df[col] = self.args_df[col].astype(float)
                    elif isinstance(self.args_df[col][0], bool):
                        self.args_df[col] = self.args_df[col].astype(bool)
                    else:
                        self.args_df[col] = self.args_df[col].astype(str)
                except Exception as e:
                    self.args_df[col] = self.args_df[col].astype(str)
                    print(e)
                    print("Leaving column as string.")

            if preprocess:
                self.args_df = self.preprocess_args(self.args_df)
            self.args_df.to_csv(f"{self.path}/qt_metadataset/{self.version}/{file_name}.csv")
        else:
            self.args_df = pd.read_csv(f"{self.path}/qt_metadataset/{self.version}/{file_name}.csv", index_col=0)

        self.check_valid_args()

        self.args_max = self.args_df.max()
        self.args_min = self.args_df.min()

        if self.standardize_numerical_args:
            self.args_mean = self.args_df.mean()
            self.args_std = self.args_df.std()
            for col in self.args_df.columns:
                if self.args_df[col].dtype == "float64" and not col.startswith("cat") \
                        and self.args_std[col] > 0 \
                        and col not in ["max_eval_top1", "max_eval_top5"]:
                    self.args_df[col] = (self.args_df[col] - self.args_mean[col]) / self.args_std[col]

        if self.model_args_first:
            hyperparams = self.args_df.columns.tolist()
            for col in hyperparams:
                if col.startswith("cat__model"):
                    hyperparams.remove(col)
                    hyperparams.insert(0, col)
            self.args_df = self.args_df[hyperparams]
            self.args_df.columns = hyperparams



    def check_valid_args(self):

        #self.args_df["initial_eval_loss"][self.args_df["initial_eval_loss"] > 100] = 100
        self.args_df["decay_epochs"][self.args_df["decay_epochs"] > 20] = 20

    def preprocess_args(self, args_df):
        index = args_df.index
        if self.drop_constant_args:
            constant_cols = []
            for col in list(args_df.columns):
                if len(args_df[col].astype(str).unique()) == 1:
                    constant_cols.append(col)

            for column in constant_cols+COLS_TO_DROP:
                if column in args_df.columns:
                    args_df = args_df.drop(columns=column)

        #cat_columns = args_df.select_dtypes(include=['object']).columns.tolist()
        cat_columns = CAT_COLS
        if self.encode_categorical_args:
            cat_columns.remove("experiment")

            cat_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            col_transformer = ColumnTransformer(transformers=[
                ('cat', cat_transformer, cat_columns)
            ])

            # Fit and transform the data
            df_cat_transformed = col_transformer.fit_transform(args_df).toarray()
            new_cat_columns = col_transformer.get_feature_names_out().tolist()
            df_cat_transformed = pd.DataFrame(df_cat_transformed, columns=new_cat_columns)
        else:
            df_cat_transformed = args_df.select_dtypes(include=['object'])

        #non_cat_columns = args_df.select_dtypes(include=['float64', 'int64', 'bool']).columns.tolist()
        non_cat_columns = NON_CAT_COLS
        args_df["clip_grad"][args_df["clip_grad"] == "None"] = np.nan
        args_df["layer_decay"][args_df["layer_decay"] == "None"] = np.nan

        if self.impute_numerical_args:
            imputer = SimpleImputer(strategy='constant', fill_value=-1)
            df_non_cat_transformed = imputer.fit_transform(args_df[non_cat_columns])
            df_non_cat_transformed = pd.DataFrame(df_non_cat_transformed, columns=non_cat_columns)
        else:
            df_non_cat_transformed = args_df.select_dtypes(include=['float64', 'int64', 'bool'])

        args_df = pd.concat([df_cat_transformed, df_non_cat_transformed], axis=1)
        args_df.index = index

        return args_df

    def preload_hyperparameters_candidates(self):
        assert self.dataset_name is not None, "Please specify a dataset name."
        args = list(self.dataset_args_df.columns)
        args_to_remove = ["experiment", "dataset", "distributed",
                          "rank", "world_size", "prefetcher", "epochs",
                          "curve_length", "device_count", "final_batch_size",
                          "invalid_loss_value", "max_memory_allocated", "test_mode",
                          "initial_eval_top1", "initial_eval_top5", "initial_eval_loss",
                          "num_classes", "max_eval_top1", "max_eval_top5"]

        for arg in self.dataset_args_df.columns:
            if arg.startswith("cat__dataset") or arg.startswith("cat__project") or (arg in args_to_remove):
                args.remove(arg)

        self.hyperparameters_candidates = self.dataset_args_df[args]
        self.hyperparameter_names = args


    def get_hyperparameters_candidates(self):
        return self.hyperparameters_candidates

    def get_datasets(self):
        curve_name = list(self.curves.keys())[0]
        return list(self.curves[curve_name].keys())

    def get_runs(self, dataset_name=None, augmentation_id=None):
        if dataset_name is None:
            dataset_name = self.dataset_name
        curve_name = list(self.curves.keys())[0]
        runs = list(self.curves[curve_name][dataset_name].keys())

        if augmentation_id is not None:
            adapted_dataset_name = "_".join(dataset_name.split("/")[1:]+[str(augmentation_id), ""])
            runs = [run for run in runs if adapted_dataset_name in run]

        return runs

    def set_dataset_name(self, dataset_name, augmentation_id = None):
        self.dataset_name = dataset_name
        ohe_dataset_name = f"cat__dataset_{dataset_name}"
        self.runs_list = self.get_runs(self.dataset_name, augmentation_id)
        self.dataset_args_df = self.args_df[self.args_df[ohe_dataset_name]==1]
        self.augmentation_id = augmentation_id

        if augmentation_id is not None:
            self.dataset_args_df = self.dataset_args_df.loc[self.args_df["augmentation_id"]==augmentation_id].copy()
            self.dataset_args_df = self.dataset_args_df.drop(columns=["augmentation_id"])

        if self.target_model is not None:
            if self.action_on_model == "omit_it": #omit one
                ohe_target_model_name = f"cat__model_{self.target_model}"
                self.dataset_args_df = self.dataset_args_df[self.dataset_args_df[ohe_target_model_name]!=1].copy()
            elif self.action_on_model == "omit_the_rest": #omit rest
                ohe_target_model_name = f"cat__model_{self.target_model}"
                self.dataset_args_df = self.dataset_args_df[self.dataset_args_df[ohe_target_model_name]==1].copy()

        if self.subsample_models_in_hub is not None:
            for model in self.model_vars:
                if model not in self.selected_models:
                    self.dataset_args_df = self.dataset_args_df[self.dataset_args_df[model]!=1]
        self.preload_hyperparameters_candidates()

    def get_incumbent_curve(self):
        assert self.dataset_name is not None, "Dataset name not set. Use set_dataset_name() to set it."
        best_index = self.get_incumbent_config_index()
        inc_run_id = self.dataset_args_df.index[best_index]
        return self.curves["eval_top1"][self.dataset_name][inc_run_id]

    def get_incumbent_config_index(self):
        assert self.dataset_name is not None, "Dataset name not set. Use set_dataset_name() to set it."
        best_index = self.dataset_args_df["max_eval_top1"].argmax()
        return best_index

    def get_incumbent_config_id(self):
        return self.get_incumbent_config_index()

    def get_best_performance(self):
        return self.dataset_args_df["max_eval_top1"].max()

    def get_worst_performance(self):
        min_perf = 100
        for run_id in self.curves["eval_top1"][self.dataset_name].keys():
            curve = self.curves["eval_top1"][self.dataset_name][run_id]
            try:
                min_perf = min(min_perf, min(curve))
            except Exception as e:
                print("Error in curve: ", curve, e)
        return min_perf

    def get_gap_performance(self):
        return self.get_best_performance() - self.get_worst_performance()

    def get_step_cost(self, hp_index, budget=None):
        '''
        Returns the cost of a step in the curve.
        hp_index: index of the hyperparameter configuration
        budget: budget of the step (time)
        '''

        cost = self.get_curve(hp_index, budget, curve_name="eval_time")[-1]+\
               self.get_curve(hp_index, budget, curve_name="train_time")[-1]
        return cost
    def get_curve_cost(self, hp_index, budget=None):
        cost = sum(self.get_curve(hp_index, budget, curve_name="eval_time"))+\
                sum(self.get_curve(hp_index, budget, curve_name="train_time"))
        return cost

    def get_performance(self, hp_index, budget=None, run_id=None, curve_name=None):
        #docstring
        '''
        Returns the performance of a hyperparameter configuration at a given budget.
        '''
        curve = self.get_curve(hp_index, budget, run_id, curve_name)
        return curve[-1]

    def get_curve(self, hp_index, budget=None, run_id=None, curve_name=None):
        '''
        Returns the performance curve of a hyperparameter configuration.
        hp_index: index of the hyperparameter configuration
        budget: budget of the curve (time)
        run_id: run id of the curve
        curve_name: name of the curve to return (default is 'eval_top1')
        '''
        assert self.dataset_name is not None, "Dataset name not set. Use set_dataset_name() to set it."

        if curve_name is None:
            if self.verbose:
                print("Curve name not set. Using eval_top1 by default.")
            curve_name = "eval_top1"

        if run_id is None:
            run_id = self.runs_list[hp_index]

        if budget is not None:
            return self.curves[curve_name][self.dataset_name][run_id][:budget]
        else:
            return self.curves[curve_name][self.dataset_name][run_id]

    def get_curve_len(self, *args, **kwargs):
        assert self.dataset_name is not None, "Dataset name not set. Use set_dataset_name() to set it."
        return len(self.get_curve(*args, **kwargs))

    def get_hyperparameters(self, hp_index=None):
        return self.hyperparameters_candidates.iloc[hp_index]

    def get_num_hyperparameters(self):
        return len(self.hyperparameters_candidates)

    def get_hyperparameters_names(self):
        return self.hyperparameter_names

    def preset_metafeatures(self, preset_metafeatures):
        self.preset_metafeatures = preset_metafeatures

    def get_metafeatures(self):
        #TODO: unify the dataset name pattern
        if self.augmentation_id is not None:
            adapted_dataset_name = "/".join(self.dataset_name.split("/")[1:])
            adapted_dataset_name = f"{adapted_dataset_name}-{self.augmentation_id}"
        else:
            adapted_dataset_name = self.dataset_name

        if self.preset_metafeatures is None:
            return self.metafeatures.loc[adapted_dataset_name].values
        else:
            return self.preset_metafeatures

    def set_action_on_model(self, target_model, action_on_model):
        #set action
        self.action_on_model = action_on_model
        self.target_model = target_model

    def reset_action_on_model(self):
        self.action_on_model = None
        self.target_model = None

    def set_subsample_models(self, subsample_models_in_hub):
        self.subsample_models_in_hub = subsample_models_in_hub
        self.model_vars = self.dataset_args_df.columns[self.dataset_args_df.columns.str.startswith("cat__model_")]
        self.selected_models = np.random.choice(self.model_vars, self.subsample_models_in_hub, replace=False)

if __name__ == "__main__":
    aggregate_data = False
    augmentation_id = None
    set = "micro"
    target_model = None
    action_on_model = None
    path = "data"
    loader = QTMetaDataset(aggregate_data=aggregate_data,
                            path=path,
                            set=set,
                            target_model=target_model,
                            action_on_model=action_on_model)
    datasets = loader.get_datasets()
    total_cost = 0
    total_len = 0
    total_curves = 0
    for dataset in datasets:
        #runs = loader.get_runs(datasets[1)
        loader.set_dataset_name(dataset, augmentation_id=augmentation_id)
        #curve = loader.get_curve(hp_index=0, budget=10)
        hps = loader.get_hyperparameters_candidates()
        #perf = loader.get_performance(hp_index=0, budget=10)
        #inc = loader.get_incumbent_curve()
        #inc_idx = loader.get_incumbent_config_id()
        #worst_perf = loader.get_worst_performance()

        for i in range(len(hps)):
            cost = loader.get_curve_cost(hp_index=i)
            total_cost += cost
            #cost = loader.get_curve_cost(hp_index=0)
            #gap = loader.get_gap_performance()
            curve_length = loader.get_curve_len(hp_index=0)
            total_len += curve_length
        total_curves += len(hps)
    print(total_cost)
    print(total_len)
    print(total_curves)

    #print(inc)




