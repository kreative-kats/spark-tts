"""Provides utility methods for the mishka experiment."""

from hashlib import md5
from pathlib import Path

import torch


def get_prompt_segments(prompt_text: str, segment_size: int) -> list[str]:
    """Returns a list with prompt text segments with clipped length."""
    words = prompt_text.split()
    return [
        " ".join(words[i : i + segment_size])
        for i in range(0, len(words), segment_size)
    ]


def get_id_from_path(path: Path) -> str:
    """Returns a unique identifier as a md5 hash of a file system path."""
    return md5(str(path).encode(), usedforsecurity=False).hexdigest()


def get_id_from_path_and_weight(path: Path, weight: float) -> str:
    """Returns a unique identifier as a md5 hash of a file system path."""
    path_weight = f"{path}|{weight:.4f}"
    return md5(str(path_weight).encode(), usedforsecurity=False).hexdigest()


def get_id_from_paths_and_weights(paths: list[Path], weights: list[float]) -> str:
    """Returns a unique identifier as a md5 hash of a file system path."""
    identifiers = sorted(
        get_id_from_path_and_weight(path, weight)
        for path, weight in zip(paths, weights, strict=True)
    )
    identifier = ".".join(identifiers)
    return md5(identifier.encode(), usedforsecurity=False).hexdigest()


def select_torch_device() -> torch.device:
    """Returns the best torch device matching the system profile."""
    if torch.accelerator.is_available():
        return torch.accelerator.current_accelerator()

    return torch.device("cpu")


__all__ = [
    "get_id_from_path",
    "get_id_from_path_and_weight",
    "get_id_from_paths_and_weights",
    "get_prompt_segments",
    "select_torch_device",
]
