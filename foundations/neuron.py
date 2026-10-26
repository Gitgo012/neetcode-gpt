import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(
        self,
        x: NDArray[np.float64],
        w: NDArray[np.float64],
        b: float,
        activation: str
    ) -> float:

        z = np.dot(x, w) + b

        if activation == "sigmoid":
            y = 1 / (1 + np.exp(-z))
        elif activation == "relu":
            y = max(0, z)
        else:
            raise ValueError("Unsupported activation function")

        return round(float(y), 5)