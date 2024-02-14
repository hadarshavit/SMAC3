"""
Quadratic Function
^^^^^^^^^^^^^^^^^^

An example of applying SMAC to optimize a quadratic function.

We use the black-box facade because it is designed for black-box function optimization.
The black-box facade uses a :term:`Gaussian Process<GP>` as its surrogate model.
The facade works best on a numerical hyperparameter configuration space and should not
be applied to problems with large evaluation budgets (up to 1000 evaluations).
"""

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from matplotlib import pyplot as plt
import smac.runhistory.encoder
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario
import submitit

from smac.model.random_forest.random_forest import RandomForest
from sklearn.metrics import mean_squared_error
from ioh import get_problem, ProblemClass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
from skopt.learning import GradientBoostingQuantileRegressor, ExtraTreesRegressor
from smac.model.progressive_ensemble import ProgressiveEnsemble
from smac.model.random_forest.extra_trees import ExtraTrees
from smac.model.random_forest.gradient_boosting import GradientBoosting
from smac.model.random_forest.random_forest import RandomForest
from smac.callback.sawei_callback import get_sawei_kwargs
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
# from hpobench.container.benchmarks.nas.nasbench_101 import NASCifar10ABenchmark, NASCifar10BBenchmark, NASCifar10CBenchmark
# from hpobench.container.benchmarks.nas.nasbench_201 import ImageNetNasBench201Benchmark

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class QuadraticFunction:
    def __init__(self, model, task_id, seed) -> None:
        self.bench = TabularBenchmark(model, task_id, rng=seed)
        self.seed = seed
        self.configs = []
        self.evals = []

    @property
    def configspace(self) -> ConfigurationSpace:
        return self.bench.get_configuration_space(seed=self.seed)

    def train(self, config: Configuration, seed) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        # vs = []
        # for seed in self.bench._seeds_used():
        #     vs.append(self.bench.objective_function(config, seed=seed)['function_value'])
        # v = np.mean(vs)
        self.configs.append(config)
        v = self.bench.objective_function(config)['function_value']
        self.evals.append(v)
        return v

def run(model_name, task_id, surrogate_model):
    for seed in range(5):
        model = QuadraticFunction(model_name, task_id, seed)

        # Scenario object specifying the optimization "environment"
        scenario = Scenario(model.configspace, 
                            deterministic=True, 
                            n_trials=256, 
                            seed=seed,
                            output_directory=f'smac_out_hpobench/m{model_name}t{task_id}sm{surrogate_model}s{seed}')
        if surrogate_model == 'orig':
            smodel = None
        elif surrogate_model == 'ens30':
            smodel = ProgressiveEnsemble(models=[HPOFacade.get_model(scenario), GradientBoosting(model.configspace), ExtraTrees(model.configspace)],
                        configspace=model.configspace, n_models_to_keep=30)
        elif surrogate_model == 'ens3':
            smodel = ProgressiveEnsemble(models=[HPOFacade.get_model(scenario), GradientBoosting(model.configspace), ExtraTrees(model.configspace)],
                        configspace=model.configspace, n_models_to_keep=3)
        elif surrogate_model == 'gb':
            smodel = GradientBoosting(model.configspace)
        elif surrogate_model == 'et':
            smodel = ExtraTrees(model.configspace)
        # Now we use SMAC to find the best hyperparameters
        smac = HPOFacade(
            scenario,
            model.train,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
            model=smodel,
            # runhistory_encoder=smac.runhistory.encoder.RunHistoryEncoder()
        )
        smac._config_selector._retries = 1024 * 1024

        incumbent = smac.optimize()

        np.save(f'data_hpobench/m{model_name}t{task_id}sm{surrogate_model}s{seed}_evals.npy', model.evals)
        np.save(f'data_hpobench/m{model_name}t{task_id}sm{surrogate_model}s{seed}_configs.npy', model.configs)



if __name__ == "__main__":
    executor = submitit.AutoExecutor('logs_hpobench', 'slurm')
    executor.update_parameters(timeout_min=60 * 48, slurm_partition="Kathleen", slurm_array_parallelism=8, cpus_per_task=1, mem_gb=16, )
    with executor.batch():
        for model in [ 'lr', 'svm', 'xgb','nn', ]: # 'lr', 'svm', 'xgb''nn''rf'
            for task_id in [31, 53, 3917, 9952, 10101, 146818, 146821, 146822]:
                for surrogate_model in ['orig', 'ens30', 'ens3', 'gb', 'et']:
                    j = executor.submit(run, model, task_id, surrogate_model)
                    # j.result()
