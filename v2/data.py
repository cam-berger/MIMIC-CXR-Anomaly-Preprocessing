"""PyTorch datasets over the v2 manifest's PNGs."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """Images (and optionally targets with a label mask) for a list of manifest rows."""

    def __init__(
        self,
        paths: Sequence[str],
        transform: Callable,
        targets: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ):
        self.paths = [str(p) for p in paths]
        self.transform = transform
        self.targets = None if targets is None else torch.as_tensor(targets, dtype=torch.float32)
        self.mask = None if mask is None else torch.as_tensor(mask, dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.paths[i]) as img:
            x = self.transform(img.convert("L"))
        if self.targets is None:
            return x, i
        return x, self.targets[i], self.mask[i], i


def image_paths(out_dir: Path, rows) -> list[str]:
    return [str(Path(out_dir) / rel) for rel in rows["image"]]


def loader_kwargs(workers: int, device, persistent: bool = True) -> dict:
    """DataLoader settings that work on CUDA, MPS and CPU (macOS spawns workers)."""
    kwargs = {"num_workers": workers, "pin_memory": getattr(device, "type", str(device)) == "cuda"}
    if workers > 0:
        kwargs["persistent_workers"] = persistent
        kwargs["prefetch_factor"] = 4
    return kwargs
