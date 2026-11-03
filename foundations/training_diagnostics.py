import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:

        stats = []
        outputs = []

        def hook(module, input, output):
            outputs.append(output.detach())

        handles = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                handles.append(layer.register_forward_hook(hook))

        with torch.no_grad():
            model(x)

        for handle in handles:
            handle.remove()

        for output in outputs:

            mean = output.mean().item()
            std = output.std().item()

            dead_neurons = (output <= 0).all(dim=0)
            dead_fraction = dead_neurons.float().mean().item()

            stats.append({
                "mean": round(mean, 4),
                "std": round(std, 4),
                "dead_fraction": round(dead_fraction, 4)
            })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        stats = []

        model.zero_grad()

        output = model(x)

        loss_fn = nn.MSELoss()
        loss = loss_fn(output, y)

        loss.backward()

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:

        # 1. Dead neurons
        if any(s["dead_fraction"] > 0.5 for s in activation_stats):
            return "dead_neurons"

        # 2. Exploding gradients
        if any(s["norm"] > 1000 for s in gradient_stats):
            return "exploding_gradients"

        # 3. Vanishing gradients (last layer)
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Check activation standard deviation
        if any(s["std"] < 0.1 for s in activation_stats):
            return "vanishing_gradients"

        if any(s["std"] > 10.0 for s in activation_stats):
            return "exploding_gradients"

        # 5. Otherwise healthy
        return "healthy"