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


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class QuadraticFunction:
    def __init__(self, fid, iid, dim) -> None:
        self.f = get_problem(fid, instance=iid, dimension=dim, problem_class=ProblemClass.BBOB)
        self.dim = dim
        self.evals = []
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        for i in range(self.dim):
            cs.add_hyperparameter(Float(f'x{i}', (-5, 5)))
        # x1 = Float("x1", (-5, 5))#, default=-5)
        # x2 = Float("x2", (-5, 5))#, default=-5)

        # cs.add_hyperparameters([x1, x2])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a quadratic function with a minimum we know to be at x=0."""
        v = self.f([config[f'x{i}'] for i in range(self.dim)])
        if len(self.evals) == 0  or self.evals[-1] > v:
            self.evals.append(v)
        else:
            self.evals.append(self.evals[-1])
        return self.f([config[f'x{i}'] for i in range(self.dim)])# x**2


def get_trajectory(smac: HPOFacade, sort_by: str = "trials") -> tuple[list[float], list[float]]:
    assert smac is not None
    rh = smac.runhistory
    trajectory = smac.intensifier.trajectory
    X: list[int | float] = []
    Y: list[float] = []

    for traj in trajectory:
        assert len(traj.config_ids) == 1
        config_id = traj.config_ids[0]
        config = rh.get_config(config_id)

        cost = rh.get_cost(config)
        if cost > 1e6:
            continue

        if sort_by == "trials":
            X.append(traj.trial)
        elif sort_by == "walltime":
            X.append(traj.walltime)
        else:
            raise RuntimeError("Unknown sort_by.")

        Y.append(cost)

    return X, Y


def run(dim, fid):
    data = []
    data_mses = []
    for sawei in [True, False]:
        for smac_model, mname in [(None, 'ens'), (None, 'orig')]:
            for t in range(5):
                xs, ys = [], []
                sawei_add = 'SAWEI' if sawei else ''

                model = QuadraticFunction(fid, t, dim)

                # Scenario object specifying the optimization "environment"
                scenario = Scenario(model.configspace, 
                                    deterministic=True, 
                                    n_trials=32 * dim, 
                                    seed=t,
                                    output_directory=f'smac_out_bbob/f{fid}d{dim}t{t}{mname}{sawei_add}')
                sawei_kwargs = get_sawei_kwargs() if sawei else {}
                # Now we use SMAC to find the best hyperparameters
                smac = HPOFacade(
                    scenario,
                    model.train,  # We pass the target function here
                    overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
                    model=None if mname == 'orig' else ProgressiveEnsemble(models=[HPOFacade.get_model(scenario), GradientBoosting(model.configspace), ExtraTrees(model.configspace)],
                                configspace=model.configspace),
                    **sawei_kwargs,  # Add SAWEI
                    # runhistory_encoder=smac.runhistory.encoder.RunHistoryEncoder()
                )

                incumbent = smac.optimize()

                # Get cost of default configuration
                default_cost = smac.validate(model.configspace.get_default_configuration())
                print(f"Default cost: {default_cost}")

                # Let's calculate the cost of the incumbent
                incumbent_cost = smac.validate(incumbent)
                print(f"Incumbent cost: {incumbent_cost}")

                # Let's plot it too
                # plot(smac.runhistory, incumbent)
                # print(smac._optimizer.intensifier.config_selector._runhistory_encoder)
                # X, Y = get_trajectory(smac)
                xs += list(range(len(model.evals)))
                ys += model.evals
                xys = zip(xs, ys)
                sawei_add = 'SAWEI' if sawei else ''
                xys = [(x, y - model.f.optimum.y, mname + sawei_add) for x, y in xys]
                data += xys
                # import pdb; pdb.set_trace()
                X, Y, X_configurations = smac._optimizer.intensifier.config_selector._collect_data()
                for l in smac._optimizer.intensifier.config_selector.models:
                    if mname == 'orig':
                        preds = RandomForest.load(f'{scenario.output_directory}/trained_models/{l}')[1].predict(X)[0]
                    else:
                        preds = ProgressiveEnsemble.load(f'{scenario.output_directory}/trained_models/{l}').predict(X)[0]

                    # preds = smac._optimizer.intensifier.config_selector._model.predict(X)[0]
                    mse = np.sqrt(mean_squared_error(Y, preds))
                    data_mses.append([l, mse, mname + sawei_add])
                    
                # print(xys)
                # print(model.evals)
        
    df = pd.DataFrame(data)
    df.columns = ['iter', 'cost', 'type']
    df.to_csv(f'data_bbob/opt_f{fid}d{dim}.csv')
    sns.lineplot(df, x='iter', y='cost', hue='type')
    plt.yscale("log")
    plt.savefig(f'figs_bbob/opt_f{fid}d{dim}.png')
    plt.clf()
    plt.cla()

    df = pd.DataFrame(data_mses)
    df.columns = ['iter', 'cost', 'type']
    df.to_csv(f'data_bbob/acc_f{fid}d{dim}.csv')
    sns.lineplot(df, x='iter', y='cost', hue='type')
    plt.yscale("log")
    plt.savefig(f'figs_bbob/acc_f{fid}d{dim}.png')
    plt.clf()
    plt.cla()


if __name__ == "__main__":
    executor = submitit.AutoExecutor('logs', 'slurm')
    executor.update_parameters(timeout_min=60 * 48, slurm_partition="Kathleen", slurm_array_parallelism=256, cpus_per_task=1, mem_gb=16, )
    with executor.batch():
        for dim in [8, 16, 32, 64, 128]:
            for fid in range(1, 25):
                executor.submit(run, dim, fid)

