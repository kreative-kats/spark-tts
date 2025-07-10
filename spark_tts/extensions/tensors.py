"""Provides tensor manipulation methods for use in the mishka experiment."""

import torch


def sanitize_weights(weights: list[float] | int) -> list[float]:
    """Returns a list of weights of target length adding up to 1.0."""
    if type(weights) is int:
        return [1.0 / weights] * weights

    if sum(weights) == 1.0:
        return weights

    sum_weights = sum(weights)
    return [w / sum_weights for w in weights]


def combine_tensors(
    tensors: list[torch.Tensor],
    weights: list[float] | None = None,
    as_int: bool = False,
) -> torch.Tensor:
    """Returns a weighted sum of the input tensors."""
    if len(tensors) == 1:
        return tensors[0]

    weights = sanitize_weights(weights or len(tensors))

    weighted_tensor = None
    for tensor, weight in zip(tensors, weights, strict=True):
        if weighted_tensor is None:
            weighted_tensor = tensor.to(torch.float32) * weight
            continue
        weighted_tensor += tensor.to(torch.float32) * weight

    if not as_int:
        return weighted_tensor

    return torch.round(weighted_tensor).to(torch.int32)


__all__ = ["combine_tensors"]
