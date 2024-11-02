from __future__ import annotations

import numpy as np
import ioh
from functools import partial

from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class Schedule:
    def __init__(self, grid, grid_sum) -> None:
        self.min_val = grid.min()
        self.max_val = grid.max()
        self.total_sum = grid_sum
        self.grid_sum = self.get_grid_sum(grid)

    def __call__(self, val):
        raise NotImplementedError()
    
    def get_grid_sum(self, grid):
        raise NotImplementedError()


class LinearSchedule(Schedule):
    def __call__(self, val):
        # import pdb; pdb.set_trace()
        val = ((val - self.min_val)) / (self.max_val - self.min_val) 
        # val[val < 1e-5] = 1e-5
        return self.total_sum * val / self.grid_sum
    
    def get_grid_sum(self, grid):
        val = ((grid - self.min_val)) / (self.max_val - self.min_val) 
        return val.sum()
    

class ExponentialSchedule(Schedule):
    def __init__(self, grid, grid_sum, exp) -> None:
        self.exp = exp
        super().__init__(grid, grid_sum)

    def __call__(self, val):
        val = ((val - self.min_val)) / (self.max_val - self.min_val) 
        # val[val < 1e-5] = 1e-5
        val = 1 - self.exp ** val
        return self.total_sum * val / self.grid_sum
    
    def get_grid_sum(self, grid):
        val = ((grid - self.min_val)) / (self.max_val - self.min_val) 
        val = 1 - self.exp ** val
        return val.sum()

class PolynomialSchedule(Schedule):
    def __init__(self, grid, grid_sum, base) -> None:
        self.base = base
        super().__init__(grid, grid_sum)

    def __call__(self, val):
        val = ((val - self.min_val)) / (self.max_val - self.min_val) 
        # val[val < 1e-5] = 1e-5
        val = val ** self.base
        return self.total_sum  * val / self.grid_sum
    
    def get_grid_sum(self, grid):
        val = ((grid - self.min_val)) / (self.max_val - self.min_val) 
        val = val ** self.base
        return val.sum()
    

class CosineSchedule(Schedule):
    def __init__(self, grid, grid_sum) -> None:
        super().__init__(grid, grid_sum)

    def __call__(self, val):
        val = ((val - self.min_val)) / (self.max_val - self.min_val) 
        # val[val < 1e-5] = 1e-5
        val = (np.cos(np.pi + val * np.pi) + 1) / 2
        return self.total_sum  * val / self.grid_sum
    
    def get_grid_sum(self, grid):
        val = ((grid - self.min_val)) / (self.max_val - self.min_val) 
        val = (np.cos(np.pi + val * np.pi) + 1) / 2
        return val.sum()


class TanhSchedule(Schedule):
    def __init__(self, grid, grid_sum) -> None:
        super().__init__(grid, grid_sum)

    def __call__(self, val):
        val = ((val - self.min_val)) / (self.max_val - self.min_val) 
        # val[val < 1e-5] = 1e-5
        val = (np.tanh(4 * val - 2)) / 2 + 0.5
        return self.total_sum  * val / self.grid_sum
    
    def get_grid_sum(self, grid):
        val = ((grid - self.min_val)) / (self.max_val - self.min_val) 
        val = (np.tanh(4 * val - 2)) / 2 + 0.5
        return val.sum()
    
class HardTanhSchedule(Schedule):
    def __init__(self, grid, grid_sum) -> None:
        super().__init__(grid, grid_sum)

    def __call__(self, val):
        val = ((val - self.min_val)) / (self.max_val - self.min_val) 
        # val[val < 1e-5] = 1e-5
        val = (np.tanh(4 * val - 2)) / 2 + 0.5
        return self.total_sum  * val / self.grid_sum
    
    def get_grid_sum(self, grid):
        val = ((grid - self.min_val)) / (self.max_val - self.min_val) 
        val = (np.tanh(8 * val - 4)) / 2 + 0.5
        return val.sum()
    

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
    def __init__(self, f: ioh.ProblemType, base_noise, schedule, distance_function) -> None:
        super().__init__(f, base_noise, schedule)
        self.distance_function = distance_function

    # def get_max_distance(self):
    #     max_dist_point = [5 if x < 0 else -5 for x in self.f.optimum.x]
    #     max_dist = self.distance_function(max_dist_point, self.f.optimum.x)
    #     return max_dist

    def __call__(self, xs, ground_truths):
        distances = np.apply_along_axis(partial(self.distance_function, self.f.optimum.x), 1, xs)
        return self.schedule(distances) * self.base_noise
    

class NoisySurrogateModel(AbstractModel):
    def __init__(self, noise_type, target_function, min_noise, **kwargs) -> None:
        super().__init__(**kwargs)
        self.noise_type = noise_type
        self.target_function = target_function
        self.min_noise = min_noise

    def _train(self, X: np.ndarray, Y: np.ndarray) -> NoisySurrogateModel:
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

        ground_truth = np.array(self.target_function(X))
        noise_levels = self.noise_type(X, ground_truth) + self.min_noise
        

        # if (noise_levels <= 0).any():
        #     import pdb; pdb.set_trace()
        mu_noise = np.random.normal(loc=0, scale=noise_levels, size=ground_truth.shape)
        sigma_noise = np.random.normal(loc=0, scale=noise_levels, size=ground_truth.shape)

        return ground_truth + mu_noise, np.power(np.abs(mu_noise) + np.abs(sigma_noise), 2)
