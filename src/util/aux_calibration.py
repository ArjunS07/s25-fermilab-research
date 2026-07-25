"""Gradient-scale calibration for mass-shell joint transport objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


def shared_backbone_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Trainable tangent-backbone parameters excluding the final velocity readout."""
    selected = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "tangent_backbone." in name
        and "tangent_backbone.readout." not in name
    ]
    if not selected:
        raise ValueError("no shared tangent-backbone parameters found")
    return selected


def _gradient_products(
    losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> dict[tuple[str, str], float]:
    names = list(losses)
    gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
    for index, name in enumerate(names):
        gradients[name] = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=index < len(names) - 1,
            allow_unused=True,
        )

    products: dict[tuple[str, str], float] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index:]:
            product = 0.0
            for grad_left, grad_right in zip(gradients[left], gradients[right]):
                if grad_left is not None and grad_right is not None:
                    product += float(
                        torch.sum(
                            grad_left.detach().to(torch.float64)
                            * grad_right.detach().to(torch.float64)
                        ).cpu()
                    )
            products[(left, right)] = product
    return products


@dataclass
class GradientCalibration:
    """Accumulate deterministic gradient inner products without updating weights."""

    names: tuple[str, ...] = ("base", "gram", "total_momentum")
    batches: int = 0
    products: dict[tuple[str, str], float] = field(default_factory=dict)

    def update(
        self,
        losses: dict[str, torch.Tensor],
        parameters: list[torch.nn.Parameter],
    ) -> None:
        if tuple(losses) != self.names:
            raise ValueError(f"expected losses {self.names}, got {tuple(losses)}")
        batch_products = _gradient_products(losses, parameters)
        for key, value in batch_products.items():
            self.products[key] = self.products.get(key, 0.0) + value
        self.batches += 1

    def result(
        self,
        single_fraction: float = 0.2,
        combined_fraction: float = 0.1,
        minimum_weight: float = 1e-8,
        maximum_weight: float = 10.0,
    ) -> dict:
        if self.batches == 0:
            raise ValueError("cannot calibrate without batches")

        mean_products = {
            key: value / self.batches for key, value in self.products.items()
        }
        norms = {
            name: math.sqrt(max(mean_products[(name, name)], 0.0))
            for name in self.names
        }
        if norms["base"] == 0.0:
            raise ValueError("base gradient norm is zero")

        def weight(name: str, fraction: float) -> float:
            if norms[name] == 0.0:
                raise ValueError(f"{name} gradient norm is zero")
            raw = fraction * norms["base"] / norms[name]
            return min(max(raw, minimum_weight), maximum_weight)

        cosines = {}
        for left_index, left in enumerate(self.names):
            for right in self.names[left_index + 1:]:
                denominator = norms[left] * norms[right]
                value = (
                    mean_products[(left, right)] / denominator
                    if denominator > 0
                    else float("nan")
                )
                cosines[f"{left}__{right}"] = value

        return {
            "batches": self.batches,
            "gradient_rms_norm": norms,
            "gradient_cosine": cosines,
            "target_gradient_fraction": {
                "single_auxiliary": single_fraction,
                "combined_each": combined_fraction,
            },
            "weight_bounds": [minimum_weight, maximum_weight],
            "weights": {
                "gram_only": weight("gram", single_fraction),
                "total_momentum_only": weight("total_momentum", single_fraction),
                "combined_gram": weight("gram", combined_fraction),
                "combined_total_momentum": weight(
                    "total_momentum", combined_fraction
                ),
            },
        }
