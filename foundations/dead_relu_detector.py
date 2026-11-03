import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = []
        outputs = []

        def hook(module, input, output):
            outputs.append(output.detach())

        handles = []

        # Register hooks on every ReLU layer
        for layer in model.modules():
            if isinstance(layer, nn.ReLU):
                handles.append(layer.register_forward_hook(hook))

        # Execute the complete forward pass
        with torch.no_grad():
            model(x)

        # Calculate dead neuron fraction for each ReLU
        for output in outputs:
            dead_neurons = (output == 0).all(dim=0)
            dead_fraction = dead_neurons.float().mean().item()

            dead_fractions.append(round(dead_fraction, 4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        # Given dead fractions per ReLU layer, suggest a fix.
        # Check in this order:
        # 1. 'use_leaky_relu' if any layer has dead fraction > 0.5
        # 2. 'reinitialize' if the first layer has dead fraction > 0.3
        # 3. 'reduce_learning_rate' if dead fraction strictly increases
        #    with depth AND the last layer's fraction > 0.1
        # 4. 'healthy' if max dead fraction < 0.1
        # 5. 'healthy' otherwise
        # 1. Any layer has dead fraction > 0.5
        if any(x > 0.5 for x in dead_fractions):
            return "use_leaky_relu"

        # 2. First layer has dead fraction > 0.3
        if dead_fractions and dead_fractions[0] > 0.3:
            return "reinitialize"

        # 3. Dead fraction strictly increases with depth
        if len(dead_fractions) >= 2:
            increasing = all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )

            if increasing and dead_fractions[-1] > 0.1:
                return "reduce_learning_rate"

        # 4. Maximum dead fraction < 0.1
        if not dead_fractions or max(dead_fractions) < 0.1:
            return "healthy"

        # 5. Otherwise
        return "healthy"
