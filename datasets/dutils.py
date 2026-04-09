"""Dataset helpers for VAE training on Chr20 Hi-C maps."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split

from .Data_loader_PKL import Chr20MultimodalDataset


class HicOnlyVAE(torch.utils.data.Dataset):
    """Wrap multimodal dataset; VAE trains on Hi-C maps (1, 100, 100) only."""

    def __init__(self, base: Chr20MultimodalDataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        hic, *_ = self.base[idx]
        return hic, 0


def get_dataset(
    cache_dir: str,
    enformer_tsv: str,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    split_seed: int = 42,
):
    """
    Build train / held-out test splits for the Chr20 VAE (Hi-C channel only).

    Returns
    -------
    train_dataset, test_dataset, in_channels
        ``in_channels`` is 1 for single-channel Hi-C input to ``AutoencoderKL``.
    """
    full = Chr20MultimodalDataset(cache_dir, enformer_tsv)
    wrapped = HicOnlyVAE(full)
    n = len(wrapped)
    train_size = int(train_frac * n)
    val_size = int(val_frac * n)
    test_size = n - train_size - val_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, _val_dataset, test_dataset = random_split(
        wrapped, [train_size, val_size, test_size], generator=generator
    )
    return train_dataset, test_dataset, 1


def get_vae_dataloaders(
    cache_dir: str,
    enformer_tsv: str,
    batch_size: int,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    split_seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
):
    """
    Same splits as ``get_dataset``, plus ``DataLoader``s matching ``train_vae.py`` (shuffle train only).

    If ``pin_memory`` is None, it defaults to CUDA being available.
    If ``persistent_workers`` is None, it defaults to ``num_workers > 0``.
    """
    train_dataset, test_dataset, in_channels = get_dataset(
        cache_dir,
        enformer_tsv,
        train_frac=train_frac,
        val_frac=val_frac,
        split_seed=split_seed,
    )
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kw)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kw)
    return train_loader, test_loader, in_channels
