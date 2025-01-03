from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import power_transform

from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryPowerTransform(RunHistoryEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._instances is not None and len(self._instances) > 1:
            raise NotImplementedError("Handling more than one instance is not supported for sqrt scaled cost.")

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform the response values using PowerTransform."""
        if self._min_y <= 0:
            values = power_transform(values / values.std(), method="yeo-johnson")
        else:
            values = power_transform(values / values.std(), method="box-cox")
            if values.std() < 0.5:
                values = power_transform(values / values.std(), method="yeo-johnson")

        return values
