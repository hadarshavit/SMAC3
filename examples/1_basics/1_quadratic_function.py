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

from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario

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


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class QuadraticFunction:
    def __init__(self, iid, dim) -> None:
        self.f = get_problem(15, instance=iid, dimension=dim, problem_class=ProblemClass.BBOB)
        self.dim = dim
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
        x1 = config["x1"]
        x2 = config["x2"]

        print(self.f([config[f'x{i}'] for i in range(self.dim)]))
        return self.f([config[f'x{i}'] for i in range(self.dim)])# x**2


# def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
#     plt.figure()

#     # Plot ground truth
#     x = list(np.linspace(-5, 5, 100))
#     y = [xi * xi for xi in x]
#     plt.plot(x, y)

#     # Plot all trials
#     for k, v in runhistory.items():
#         config = runhistory.get_config(k.config_id)
#         x = config["x"]
#         y = v.cost  # type: ignore
#         plt.scatter(x, y, c="blue", alpha=0.1, zorder=9999, marker="o")

#     # Plot incumbent
#     plt.scatter(incumbent["x"], incumbent["x"] * incumbent["x"], c="red", zorder=10000, marker="x")

#     plt.savefig('fig.png')


if __name__ == "__main__":
    data = []

    for t in range(5):
        model = QuadraticFunction(t, 50)

        # Scenario object specifying the optimization "environment"
        scenario = Scenario(model.configspace, deterministic=True, n_trials=250, seed=t)

        # Now we use SMAC to find the best hyperparameters
        smac = HPOFacade(
            scenario,
            model.train,  # We pass the target function here
            overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
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
        print(smac._optimizer.intensifier.config_selector._runhistory_encoder)
        X, Y, X_configurations = smac._optimizer.intensifier.config_selector._collect_data()
        gb_models = []
        et_models = []
        # print(Y)
        for i, (model, Xi, Yi) in enumerate(smac._optimizer.intensifier.config_selector.models):
            iter = Xi.shape[0]
            smac._optimizer.intensifier.config_selector._model._rf = model
            mae = mean_squared_error(Y, smac._optimizer.intensifier.config_selector._model.predict(X)[0])
            data.append([mae, iter, 'original'])

            clf = GradientBoostingQuantileRegressor()
            clf.fit(Xi, Yi)
            gb_models.append(clf)

            clf = ExtraTreesRegressor(criterion='squared_error', max_features=1)
            clf.fit(Xi, Yi)
            et_models.append(clf)

            model_predictions = {}
            for j, (model, _, _) in enumerate(smac._optimizer.intensifier.config_selector.models[:i + 1]):
                smac._optimizer.intensifier.config_selector._model._rf = model
                model_predictions[j] = smac._optimizer.intensifier.config_selector._model.predict(Xi)[0]
            
            for j, model in enumerate(gb_models):
                model_predictions[f'gb{j}'] = model.predict(Xi).reshape(-1, 1)
            
            for j, model in enumerate(et_models):
                model_predictions[f'et{j}'] = model.predict(Xi).reshape(-1, 1)
             
            weights, traj, _ = weighted_ensemble_caruana(model_predictions=model_predictions, 
                                                      targets=Yi, 
                                                      size=min(5, i + 1), 
                                                      metric=mean_squared_error, 
                                                      select=min, 
                                                      seed=1)
            model_predictions = {}
            for j, (model, _, _) in enumerate(smac._optimizer.intensifier.config_selector.models[:i + 1]):
                smac._optimizer.intensifier.config_selector._model._rf = model
                model_predictions[j] = smac._optimizer.intensifier.config_selector._model.predict(X)[0]

            for j, model in enumerate(gb_models):
                model_predictions[f'gb{j}'] = model.predict(X).reshape(-1, 1)
            
            for j, model in enumerate(et_models):
                model_predictions[f'et{j}'] = model.predict(X).reshape(-1, 1)

            current = np.zeros_like(Y, dtype=np.float64)
            for key, w in weights.items():
                current += model_predictions[key] * w
            mae = mean_squared_error(Y, current)
            data.append([mae, iter, 'ens'])
    df = pd.DataFrame(data)
    df.columns = ['mae', 'iter', 'type']

    f = sns.lineplot(df, x='iter', y='mae', hue='type')
    plt.savefig('f1d50.png')


