from __future__ import annotations

from typing import Iterator

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.maximizer import AbstractAcquisitionMaximizer
from smac.acquisition.maximizer.helpers import ChallengerList
from smac.random_design.abstract_random_design import AbstractRandomDesign
from smac.utils.configspace import transform_continuous_designs

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class NSGAII(AbstractAcquisitionMaximizer):
    """Get candidate solutions via `DifferentialEvolutionSolvers` from scipy.

    According to scipy 1.9.2 documentation:

    'Finds the global minimum of a multivariate function.
    Differential Evolution is stochastic in nature (does not use gradient methods) to find the minimum,
    and can search large areas of candidate space, but often requires larger numbers of function
    evaluations than conventional gradient-based techniques.
    The algorithm is due to Storn and Price [1].'

    [1] Storn, R and Price, K, Differential Evolution - a Simple and Efficient Heuristic for Global
     Optimization over Continuous Spaces, Journal of Global Optimization, 1997, 11, 341 - 359.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction
    challengers : int, defaults to 50000
        Number of challengers.
    max_iter: int | None, defaults to None
        Maximum number of iterations that the DE will perform.
    strategy: str, defaults to "best1bin"
        The strategy to use for the DE.
    polish: bool, defaults to True
        Whether to polish the final solution using L-BFGS-B.
    mutation: tuple[float, float], defaults to (0.5, 1.0)
        The mutation constant.
    recombination: float, defaults to 0.7
        The recombination constant.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        max_iter: int = 1000,
        challengers: int = 50000,
        strategy: str = "best1bin",
        polish: bool = True,
        mutation: tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: int = 0,
    ):
        super().__init__(configspace, acquisition_function, challengers, seed)
        # raise NotImplementedError("DifferentialEvolution is not yet implemented.")
        self.max_iter = max_iter
        self.strategy = strategy
        self.polish = polish
        self.mutation = mutation
        self.recombination = recombination

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:
        # n_points is not used here, but is required by the interface

        configs: list[tuple[float, Configuration]] = []

        class ProblemWrapper(Problem):
            def __init__(self, configspace, acquisition_function):  # type: ignore
                self._configspace = configspace
                self._acquisition_function = acquisition_function
                super().__init__(n_var=len(self._configspace), n_obj=3, n_ieq_constr=0, xl=0.0, xu=1.0, vtype=float)

            def _evaluate(self, x, out, *args, **kwargs):  # type: ignore
                assert self._acquisition_function is not None
                acqs = -self._acquisition_function(
                    transform_continuous_designs(design=x, origin="NSGAII", configspace=self._configspace)
                )
                out["F"] = np.swapaxes(acqs, 0, 1).squeeze(-1)
                out["G"] = np.zeros((x.shape[0], 0))
                return out

        algorithm = NSGA2(pop_size=100)

        res = minimize(
            ProblemWrapper(self._configspace, self._acquisition_function),
            algorithm,
            ("n_gen", 100),
            seed=self._seed,
            verbose=False,
        )

        configs = transform_continuous_designs(
            design=res.X, origin="Acquisition Function Maximizer: NSGAII", configspace=self._configspace
        )
        np.random.shuffle(configs)

        return configs

    def maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int | None = None,
        random_design: AbstractRandomDesign | None = None,
    ) -> Iterator[Configuration]:
        """Maximize acquisition function using `_maximize`, implemented by a subclass.

        Parameters
        ----------
        previous_configs: list[Configuration]
            Previous evaluated configurations.
        n_points: int, defaults to None
            Number of points to be sampled & number of configurations to be returned. If `n_points` is not specified,
            `self._challengers` is used. Semantics depend on concrete implementation.
        random_design: AbstractRandomDesign, defaults to None
            Part of the returned ChallengerList such that we can interleave random configurations
            by a scheme defined by the random design. The method `random_design.next_iteration()`
            is called at the end of this function.

        Returns
        -------
        challengers : Iterator[Configuration]
            An iterable consisting of configurations.
        """
        if n_points is None:
            n_points = self._challengers

        def next_configs_by_order_reutrned() -> list[Configuration]:
            assert n_points is not None
            # since maximize returns a tuple of acquisition value and configuration,
            # and we only need the configuration, we return the second element of the tuple
            # for each element in the list
            return self._maximize(previous_configs, n_points)

        challengers = ChallengerList(
            self._configspace,
            next_configs_by_order_reutrned,
            random_design,
        )

        if random_design is not None:
            random_design.next_iteration()

        return challengers
