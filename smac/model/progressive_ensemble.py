from typing import Dict, List

import numpy as np
from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana
from ConfigSpace import ConfigurationSpace
from numpy import ndarray
from sklearn.metrics import mean_squared_error

from smac.model.abstract_model import AbstractModel


class ProgressiveEnsemble(AbstractModel):
    def __init__(self, models, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.models_classes: List[AbstractModel] = models
        self.trained_models = {}
        self.cnt = 0
        self.weights = {}
        import traceback

        traceback.print_stack()

    def _predict(self, X: ndarray, covariance_type: str | None = "diagonal") -> tuple[ndarray, ndarray | None]:
        current_means = np.zeros((X.shape[0], 1), dtype=np.float64)
        current_vars = np.zeros((X.shape[0], 1), dtype=np.float64)
        # print(X.shape)
        for model_name, w in self.weights.items():
            # print(model_name)
            model, mc = self.trained_models[model_name]
            mc.set_model(model)
            means, vars = mc.predict(X)
            # print(means.shape, current_means.shape, X.shape)
            current_means += means * w
            current_vars += vars * w

        return current_means, current_vars

    def _train(self, X: ndarray, Y: ndarray):
        for mc in self.models_classes:
            mc.reset_model()
            mc.train(X, Y)
            self.trained_models[f"{mc.__class__.__name__}{self.cnt}"] = (mc.copy_model(), mc)

        model_predictions = {}
        for key, (model, mc) in self.trained_models.items():
            mc.set_model(model)
            model_predictions[key] = mc.predict(X)[0]
            mc.reset_model()

        weights, _, _ = weighted_ensemble_caruana(
            model_predictions=model_predictions,
            targets=Y,
            size=min(5, self.cnt + 1),
            metric=mean_squared_error,
            select=min,
            seed=1,
        )
        self.weights = weights

        print(len(self.trained_models))
        self.cnt += 1
