import yaml
from yaml.loader import SafeLoader
import os
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

class SearchSpace:

    def __init__(self, version = "v1", sample_with_weights = False):
        self.file_name = f"search_space_{version}.yml"
        self.sample_with_weights = sample_with_weights
        self._build_search_space()
        self.n_possible_confs = None


    def weight_hp(self, idx, num_options, factor = 3):
        return ((num_options-idx)//factor)

    def _build_search_space(self):
        # Open the file and load the file
        path = os.path.dirname(__file__)
        search_space_file = os.path.join(path, "meta_data", self.file_name)
        self.cs = CS.ConfigurationSpace()
        with open(search_space_file) as f:
            self.data = yaml.load(f, Loader=SafeLoader)

        self.cs_info = {}

        #collect hyperparameters options
        conditional_hp_list = []
        hp_list = []

        for idx, hp in enumerate(self.data.keys()):
            num_options = 0
            values = []
            weights = []

            if isinstance(self.data[hp], list):
                values = self.data[hp]
                num_options = len(self.data[hp])
            elif isinstance(self.data[hp], dict):
                if "options" in self.data[hp].keys():
                    values = self.data[hp]["options"]
                    num_options = len(self.data[hp]["options"])
                    conditional_hp_list.append(hp)

            if self.sample_with_weights and hp in ["model", "pct_to_freeze"]:
                weights = [self.weight_hp(idx, num_options) for idx in range(len(values))]
                hp_list.append(CSH.CategoricalHyperparameter(hp, values, weights=weights))
            else:
                hp_list.append(CSH.CategoricalHyperparameter(hp, values))
            self.cs_info[hp] = {"options": values}

        #add hyperparameters to the search space
        self.cs.add_hyperparameters(hp_list)
        conditions_list = []
        for hp in conditional_hp_list:
            only_active_with = self.data[hp].get("only_active_with")
            for activator, values in only_active_with.items():
                conjunction_conditions = []
                if len(values)>1:
                    for value in values:
                        conjunction_conditions.append(CS.EqualsCondition(self.cs[hp], self.cs[activator],value))
                    conditions_list.append(CS.OrConjunction(*conjunction_conditions))
                else:
                    conditions_list.append(CS.EqualsCondition(self.cs[hp], self.cs[activator],values[0]))
        self.cs.add_conditions(conditions_list)

    def _build_args(self, configuration):
        args = ""
        for hp, value in configuration.items():
            if value == "None":
                pass
            elif value == False and type(value) == bool:
                pass
            elif value == True and type(value) == bool:
                args+=f" --{hp}"
            elif hp == "data_augmentation":
                if value != "auto_augment":
                    args+=f" --{value}"
            else:
                args+=f" --{hp} {value}"
        return args

    def sample_configuration(self, n=1, return_args=False):

        config = self.cs.sample_configuration(n)
        if return_args:
            args = self._build_args(config)
            return config, args
        return config

    def get_configuration_code (self, configuration):

        conf_as_dict = configuration.get_dictionary()
        code = 0
        cum_mul = 1
        for hp, value in conf_as_dict.items():
            options = self.cs_info[hp]["options"]
            idx = options.index(value)
            code += idx*cum_mul
            cum_mul *= len(options)
        
        assert code < self.get_num_possible_configurations()
        return code

    def get_num_possible_configurations (self):

        if self.n_possible_confs is None:
            cum_mul = 1
            for hp, value in self.cs_info.items():
                options = self.cs_info[hp]["options"]
                cum_mul *= len(options)      
            self.n_possible_confs = cum_mul 

        return self.n_possible_confs 

if __name__ == "__main__":
    ss = SearchSpace()
    configuration, args = ss.sample_configuration(return_args=True)
    code = ss.get_configuration_code(configuration)

    ss = SearchSpace("v2", sample_with_weights=True)
    configuration, args = ss.sample_configuration(return_args=True)
    code = ss.get_configuration_code(configuration)