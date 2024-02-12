from typing import Dict, List
import os
import lzma
import pickle

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
        self.model_keys = []
        self._fitted = False

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
            self.model_keys.append(f"{mc.__class__.__name__}{self.cnt}")
        
        while len(self.model_keys) > 30:
            key = self.model_keys.pop(0)
            del self.trained_models[key]

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
        self._fitted = True
    
    def save(self, path) -> None:
        os.makedirs(path)
        keys_classes = []

        for key, (model, mc) in self.trained_models.items():
            model_path = os.path.join(path, key)
            mc.set_model(model)
            mc.save(model_path)
            keys_classes.append((key, mc.__class__))
        weights_path = os.path.join(path, 'weights')

        with lzma.open(weights_path, 'wb') as f:
            pickle.dump(self.weights, f)

        models_keys = os.path.join(path, 'keys')

        
        with lzma.open(models_keys, 'wb') as f:
            pickle.dump(keys_classes, f)
        
        trained_models = self.trained_models
        mcs = self.models_classes

        self.trained_models = None
        self.models_classes = None

        main_cls = os.path.join(path, 'main_class.pkl')

        with lzma.open(main_cls, 'wb') as f:
            pickle.dump(self, f)
        
        self.trained_models = trained_models
        self.models_classes = mcs

    @staticmethod
    def load(path):
        main_cls = os.path.join(path, 'main_class.pkl')
        with lzma.open(main_cls, 'rb') as f:
            cls = pickle.load(f)

        models_keys_path = os.path.join(path, 'keys')
        with lzma.open(models_keys_path, 'rb') as f:
            model_keys = pickle.load(f)
        
        cls.trained_models = {}
        for key, mc in model_keys:
            model_path = os.path.join(path, key)
            cls.trained_models[key] = mc.load(model_path)
        
        return cls
