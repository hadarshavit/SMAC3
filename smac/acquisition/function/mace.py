from __future__ import annotations

from typing import Any

import numpy as np

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.confidence_bound import LCB
from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.function.probability_improvement import PI
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class MACE(AbstractAcquisitionFunction):
    def __init__(
        self,
        ei_xi: float = 0.0,
        pi_xi: float = 0.0,
        lcb_beta: float = 1.0,
        log: bool = False,
    ) -> None:
        self.ei = EI(xi=ei_xi, log=log)
        self.pi = PI(xi=pi_xi)
        self.lcb = LCB(beta=lcb_beta)
        self._model: AbstractModel | None = None

    @property
    def name(self) -> str:
        """Return the used surrogate model in the acquisition function."""
        return "MACE"

    @property
    def meta(self) -> dict[str, Any]:
        """Updates the surrogate model."""
        meta = super().meta
        meta.update({"ei": self.ei.meta, "pi": self.pi.meta, "lcb": self.lcb.meta})
        return meta

    def _update(self, **kwargs: Any) -> None:
        ei_kwargs = {k: v for k, v in kwargs.items() if k.startswith("ei_")}
        pi_kwargs = {k: v for k, v in kwargs.items() if k.startswith("pi_")}
        lcb_kwargs = {k: v for k, v in kwargs.items() if k.startswith("lcb_")}
        ei_kwargs["eta"] = kwargs.get("eta")
        pi_kwargs["eta"] = kwargs.get("eta")
        lcb_kwargs["num_data"] = kwargs.get("num_data")

        self.ei._update(**ei_kwargs)
        self.pi._update(**pi_kwargs)
        self.lcb._update(**lcb_kwargs)

    @property
    def model(self) -> AbstractModel | None:
        """Return the used surrogate model in the acquisition function."""
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        """Updates the surrogate model."""
        self._model = model
        self.ei.model = model
        self.pi.model = model
        self.lcb.model = model

    def _compute(self, X: np.ndarray) -> np.ndarray:
        ei = self.ei._compute(X)
        pi = self.pi._compute(X)
        lcb = self.lcb._compute(X)
        return np.array([ei, pi, lcb])
