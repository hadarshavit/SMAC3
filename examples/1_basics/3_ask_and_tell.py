"""
Ask-and-Tell
^^^^^^^^^^^^

This examples show how to use the Ask-and-Tell interface.
"""

from ConfigSpace import Configuration, ConfigurationSpace, Float

from smac import HyperparameterOptimizationFacade, Scenario
from smac.runhistory.dataclasses import TrialValue
from smac.runhistory.encoder import RunHistoryEncoder
from smac.model.random_forest.random_forest import RandomForest
from smac.acquisition.function.expected_improvement import EI


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class Rosenbrock2D:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-3)
        x1 = Float("x1", (-5, 10), default=-4)
        cs.add_hyperparameters([x0, x1])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """The 2-dimensional Rosenbrock function as a toy model.
        The Rosenbrock function is well know in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimium is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = config["x0"]
        x2 = config["x1"]

        cost = 100.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
        return cost


if __name__ == "__main__":
    model = Rosenbrock2D()

    # Scenario object
    scenario = Scenario(model.configspace, deterministic=False, n_trials=100, seed=1, output_directory='asktell')

    intensifier = HyperparameterOptimizationFacade.get_intensifier(
        scenario,
        max_config_calls=1,  # We basically use one seed per config only
    )

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,
        intensifier=intensifier,
        overwrite=True,
        runhistory_encoder=RunHistoryEncoder(scenario),
        model=RandomForest(
            log_y=False,
            n_trees=10,
            bootstrapping=True,
            ratio_features=1.0,
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=2**20,
            configspace=scenario.configspace,
            instance_features=scenario.instance_features,
            seed=scenario.seed,
        ),
        acquisition_function=EI(log=False)
    )

    min_cost = 1e8
    # We can ask SMAC which trials should be evaluated next
    for _ in range(100):
        info = smac.ask()
        assert info.seed is not None

        cost = model.train(info.config, seed=info.seed)
        min_cost = min(cost, min_cost)
        value = TrialValue(cost=cost, time=0.5, additional_info={'method': 'asktell'})

        smac.tell(info, value)

    # After calling ask+tell, we can still optimize
    # Note: SMAC will optimize the next 90 trials because 10 trials already have been evaluated
    # incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    # incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {min_cost}")
    # print(f"Incumbent cost: {incumbent_cost}")
