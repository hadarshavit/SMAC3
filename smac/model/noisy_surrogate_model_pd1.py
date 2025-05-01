from __future__ import annotations

import numpy as np
import pandas as pd
import ioh
from functools import partial

from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class Schedule:
    def __init__(self, min_val, max_val) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.total_sum = 1.0
        self.grid_sum = self.get_integral()

    def __call__(self, val):
        val = ((val - self.min_val)) / (self.max_val - self.min_val) 
        val = self._calc(val)
        return val / self.grid_sum
    
    def _calc(self, val):
        raise NotImplementedError()

    def get_integral(self):
        raise NotImplementedError()


class ConstantSchedule(Schedule):
    def _calc(self, val):
        return np.ones_like(val)
    
    def get_integral(self):
        return 1.0
    
    
class LinearSchedule(Schedule):
    def _calc(self, val):  
        return val
    
    def get_integral(self):
        return 0.5

class ExponentialSchedule(Schedule):
    def __init__(self, min_val, max_val, exp) -> None:
        self.exp = exp
        super().__init__(min_val, max_val)

    def _calc(self, val):
        return (1 - self.exp ** val) / (1 - self.exp)

    def get_integral(self):
        return 1 / (1 - self.exp) + 1 / (np.log(self.exp))


class PolynomialSchedule(Schedule):
    def __init__(self, min_val, max_val, base) -> None:
        self.base = base
        super().__init__(min_val, max_val)

    def _calc(self, val):
        return val ** self.base
    
    def get_integral(self):
        return 1 / (self.base + 1)

class CosineSchedule(Schedule):
    def _calc(self, val):
        return (np.cos(np.pi + val * np.pi) + 1) / 2

    def get_integral(self):
        return 0.5
    

class TanhSchedule(Schedule):
    def _calc(self, val):
        return (np.tanh(4 * val - 2)) / 2 + 0.5
    
    def get_integral(self):
        return 0.5
    
class HardTanhSchedule(Schedule):
    def _calc(self, val):
        return (np.tanh(8 * val - 4)) / 2 + 0.5
    
    def get_integral(self):
        return 0.5
    

class BBOBNoiseLevel:
    def __init__(self, f: ioh.ProblemType, base_noise, schedule) -> None:
        self.f = f
        self.base_noise = base_noise
        self.schedule = schedule

    def __call__(self, xs, ground_truths):
        raise NotImplementedError()


class BBOBEqualNoise(BBOBNoiseLevel):
    def __call__(self, xs, ground_truths):
        return np.ones_like(ground_truths) * self.base_noise
    

class BBOBValueBased(BBOBNoiseLevel):
    def __init__(self, f: ioh.ProblemType, base_noise, schedule) -> None:
        super().__init__(f, base_noise, schedule)

    def __call__(self, xs, ground_truths):
        return self.schedule(ground_truths) * self.base_noise


class BBOBDistanceToOptimum(BBOBNoiseLevel):
    def __init__(self, f: ioh.ProblemType, base_noise, schedule, distance_function, opt_x) -> None:
        super().__init__(f, base_noise, schedule)
        self.distance_function = distance_function
        self.opt_x = opt_x

    # def get_max_distance(self):
    #     max_dist_point = [5 if x < 0 else -5 for x in self.f.optimum.x]
    #     max_dist = self.distance_function(max_dist_point, self.f.optimum.x)
    #     return max_dist

    def __call__(self, xs, ground_truths):
        distances = np.apply_along_axis(partial(self.distance_function, self.opt_x), 1, xs)
        return self.schedule(distances) * self.base_noise
    

class NoisySurrogateModelPD1(AbstractModel):
    def __init__(self, noise_type, target_function, min_noise, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise_type = noise_type
        self.target_function = target_function
        self.min_noise = min_noise

    def _train(self, X: np.ndarray, Y: np.ndarray) -> NoisySurrogateModelPD1:
        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray.")
        if not isinstance(Y, np.ndarray):
            raise NotImplementedError("Y has to be of type np.ndarray.")

        logger.debug("(Pseudo) fit model to data.")
        return self

    def _predict(
        self,
        X: np.ndarray,
        covariance_type: str | None = "diagonal",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if covariance_type != "diagonal":
            raise ValueError("`covariance_type` can only take `diagonal` for this model.")
        
        if not isinstance(X, np.ndarray):
            raise NotImplementedError("X has to be of type np.ndarray.")

        confs = [{hp: conf_array[i] for i, hp in enumerate(self._configspace.keys())} for conf_array in X]
        confs = pd.DataFrame(confs)
        confs['epoch'] = self.target_function.end

        bounds = self.target_function.Result.metric_defs['valid_error_rate'].bounds
        ground_truth = self.target_function.surrogates['valid_error_rate'].predict(confs).clip(*bounds)
        # ground_truth = np.array([self.target_function.query(conf).error for conf in confs])
        noise_levels = self.noise_type(X, ground_truth) + self.min_noise
        

        # if (noise_levels <= 0).any():
        #     import pdb; pdb.set_trace()
        mu_noise = np.random.normal(loc=0, scale=noise_levels, size=ground_truth.shape)
        sigma_noise = np.random.normal(loc=0, scale=noise_levels, size=ground_truth.shape)

        return ground_truth + mu_noise, np.power(np.abs(mu_noise) + np.abs(sigma_noise), 2)
