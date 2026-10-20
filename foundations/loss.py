import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        n = y_true.shape[0]

        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        output = -1/n * np.sum(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

        return round(output, 4)

    def categorical_cross_entropy(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:

        n = y_true.shape[0]

        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        output = -1/n * np.sum(
            y_true * np.log(y_pred)
        )

        return round(output, 4)