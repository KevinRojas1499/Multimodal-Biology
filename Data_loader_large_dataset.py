#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal data loader for aligned chr19/chr20 cached data.

Loads these files from one cache directory:
- hic_cache_chr19_chr20_w100_coords.pkl
- rnaseq_raw_chr19_chr20_400bins_aligned.pkl
- atac_raw_chr19_chr20_val7_val15_400bins_aligned.pkl
- enformer_exact_index_chr19_chr20_2mb_multichr.pkl
- enformer_embedding_catalog_chr19_chr20.json

Enformer behavior:
- No external TSV path argument is required.
- Embedding TSV locations are read from the catalog JSON.
- Dataset can return either:
  1) pooled Enformer vectors (fixed 512-dim, default), or
  2) exact token matrices (variable length 1954/1955).
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def safe_pickle_load(fp: Path):
    """Load pickle with numpy compatibility fallback."""
    try:
        with open(fp, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        if not hasattr(np, "_core"):
            np.core = np.core
        sys.modules.setdefault("numpy._core", np.core)
        with open(fp, "rb") as f:
            return pickle.load(f)


class EnformerExactStore:
    """In-memory access to exact Enformer token arrays by chromosome."""

    def __init__(self, catalog_path: Path):
        if not catalog_path.exists():
            raise FileNotFoundError(f"Enformer catalog not found: {catalog_path}")

        with open(catalog_path, "r") as f:
            self.catalog = json.load(f)

        self.values_by_chrom: Dict[str, np.ndarray] = {}
        self.embedding_dim: Optional[int] = None

        print(f"Loading Enformer embeddings from catalog: {catalog_path}")
        for chrom, info in self.catalog.items():
            tsv_path = Path(info["tsv_path"])
            if not tsv_path.exists():
                raise FileNotFoundError(f"TSV for {chrom} not found: {tsv_path}")

            emb_dim = int(info["embedding_dim"])
            emb_cols = [f"emb_{i}" for i in range(emb_dim)]

            df = pd.read_csv(tsv_path, sep="\t", usecols=emb_cols)
            values = df.to_numpy(dtype=np.float32, copy=False)
            self.values_by_chrom[chrom] = values

            if self.embedding_dim is None:
                self.embedding_dim = emb_dim
            elif self.embedding_dim != emb_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, got {emb_dim} for {chrom}"
                )

            print(f"  {chrom}: loaded {values.shape[0]:,} rows x {values.shape[1]} dims")

        if self.embedding_dim is None:
            raise ValueError("Catalog is empty; no Enformer embeddings available.")

    def get_tokens(self, chrom: str, start_idx: int, end_idx: int) -> np.ndarray:
        values = self.values_by_chrom.get(chrom)
        if values is None:
            raise KeyError(f"No Enformer data loaded for chromosome: {chrom}")
        return values[int(start_idx):int(end_idx)]


class Chr19Chr20MultimodalDataset(Dataset):
    """
    Aligned multimodal dataset from cached files.

    Returns by default:
        hic:      (1, 100, 100)
        enformer: (512,) pooled token embedding
        atac:     (400, 2)
        rna:      (400, 1)

    If enformer_mode='tokens', returns exact tokens:
        enformer: (L, 512), where L is usually 1954 or 1955
    """

    def __init__(
        self,
        cache_dir: str,
        enformer_mode: str = "mean",
        atac_binarize: bool = False,
    ):
        cache_dir = Path(cache_dir)
        if not cache_dir.exists():
            raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

        self.cache_dir = cache_dir
        self.enformer_mode = enformer_mode
        self.atac_binarize = atac_binarize

        if self.enformer_mode not in {"mean", "tokens"}:
            raise ValueError("enformer_mode must be 'mean' or 'tokens'.")

        hic_path = cache_dir / "hic_cache_chr19_chr20_w100_coords.pkl"
        rna_path = cache_dir / "rnaseq_raw_chr19_chr20_400bins_aligned.pkl"
        atac_path = cache_dir / "atac_raw_chr19_chr20_val7_val15_400bins_aligned.pkl"
        index_path = cache_dir / "enformer_exact_index_chr19_chr20_2mb_multichr.pkl"
        catalog_path = cache_dir / "enformer_embedding_catalog_chr19_chr20.json"

        self.hic_coords = np.asarray(safe_pickle_load(hic_path), dtype=np.float32)
        self.rna = np.asarray(safe_pickle_load(rna_path), dtype=np.float32)
        self.atac = np.asarray(safe_pickle_load(atac_path), dtype=np.float32)

        self.index_df = pd.read_pickle(index_path).reset_index(drop=True)
        self.enformer_store = EnformerExactStore(catalog_path)
        self.enformer_dim = int(self.enformer_store.embedding_dim)

        n = len(self.hic_coords)
        if not (n == len(self.rna) == len(self.atac) == len(self.index_df)):
            raise ValueError(
                "Sample count mismatch across cached files: "
                f"hic={len(self.hic_coords)}, rna={len(self.rna)}, atac={len(self.atac)}, idx={len(self.index_df)}"
            )

        print("Loaded cached multimodal dataset")
        print(f"  samples: {n:,}")
        print(f"  Hi-C shape: {self.hic_coords.shape}")
        print(f"  RNA shape:  {self.rna.shape}")
        print(f"  ATAC shape: {self.atac.shape}")
        print(f"  Enformer mode: {self.enformer_mode} | dim: {self.enformer_dim}")

    @staticmethod
    def _coords_to_dist(coords: np.ndarray) -> np.ndarray:
        """Convert (100,3) coords to old-loader style log-normalized sigmoid distance map."""
        dif = coords[:, None, :] - coords[None, :, :]
        d = np.linalg.norm(dif, axis=-1)
        lt = np.log1p(d)
        mn, mx = lt.min(), lt.max()
        if mx > mn:
            lt = (lt - mn) / (mx - mn) * 10 - 5
        return (1.0 / (1.0 + np.exp(-lt))).astype(np.float32)

    def _get_enformer_for_idx(self, idx: int) -> np.ndarray:
        row = self.index_df.iloc[int(idx)]
        if not bool(row["has_enformer"]):
            if self.enformer_mode == "tokens":
                return np.zeros((1, self.enformer_dim), dtype=np.float32)
            return np.zeros((self.enformer_dim,), dtype=np.float32)

        chrom = str(row["emb_source_chrom"])
        start_idx = int(row["emb_start_idx"])
        end_idx = int(row["emb_end_idx"])

        tokens = self.enformer_store.get_tokens(chrom, start_idx, end_idx)
        if self.enformer_mode == "tokens":
            return tokens.astype(np.float32, copy=False)
        return tokens.mean(axis=0, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.hic_coords)

    def __getitem__(self, idx: int):
        hic = self._coords_to_dist(self.hic_coords[idx])
        rna = self.rna[idx].astype(np.float32, copy=False)

        atac = self.atac[idx].astype(np.float32, copy=False)
        if self.atac_binarize:
            atac = (atac > 0).astype(np.float32)

        enformer = self._get_enformer_for_idx(idx)

        return (
            torch.from_numpy(hic).unsqueeze(0),
            torch.from_numpy(enformer),
            torch.from_numpy(atac),
            torch.from_numpy(rna),
        )

    def get_data_info(self) -> Dict:
        valid = self.index_df[self.index_df["has_enformer"]]
        token_counts = valid["n_emb_tokens"].value_counts().sort_index().to_dict()
        return {
            "n_samples": len(self),
            "hic_coords_shape": tuple(self.hic_coords.shape),
            "rna_shape": tuple(self.rna.shape),
            "atac_shape": tuple(self.atac.shape),
            "enformer_mode": self.enformer_mode,
            "enformer_dim": self.enformer_dim,
            "valid_enformer_samples": int(valid.shape[0]),
            "token_count_distribution": token_counts,
        }


def pad_enformer_tokens_collate(batch):
    """
    Collate function for enformer_mode='tokens'.

    Pads variable-length Enformer token tensors to max length in batch.
    Returns:
      hic, enformer_padded, enformer_mask, atac, rna
    """
    hic_list, enf_list, atac_list, rna_list = zip(*batch)

    hic = torch.stack(hic_list, dim=0)
    atac = torch.stack(atac_list, dim=0)
    rna = torch.stack(rna_list, dim=0)

    lengths = [x.shape[0] for x in enf_list]
    max_len = max(lengths)
    dim = enf_list[0].shape[1]

    enformer = torch.zeros((len(enf_list), max_len, dim), dtype=enf_list[0].dtype)
    mask = torch.zeros((len(enf_list), max_len), dtype=torch.bool)
    for i, x in enumerate(enf_list):
        enformer[i, : x.shape[0]] = x
        mask[i, : x.shape[0]] = True

    return hic, enformer, mask, atac, rna


def create_dataloaders(
    cache_dir: str,
    batch_size: int = 8,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
    enformer_mode: str = "mean",
    atac_binarize: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset = Chr19Chr20MultimodalDataset(
        cache_dir=cache_dir,
        enformer_mode=enformer_mode,
        atac_binarize=atac_binarize,
    )

    n = len(dataset)
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    collate_fn = pad_enformer_tokens_collate if enformer_mode == "tokens" else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    print("\nCreated dataloaders")
    print(f"  train: {len(train_ds)}")
    print(f"  val:   {len(val_ds)}")
    print(f"  test:  {len(test_ds)}")
    print(f"  batch_size={batch_size}, workers={num_workers}, enformer_mode={enformer_mode}")

    return train_loader, val_loader, test_loader


def load_full_dataset(
    cache_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    enformer_mode: str = "mean",
    atac_binarize: bool = False,
) -> DataLoader:
    dataset = Chr19Chr20MultimodalDataset(
        cache_dir=cache_dir,
        enformer_mode=enformer_mode,
        atac_binarize=atac_binarize,
    )
    collate_fn = pad_enformer_tokens_collate if enformer_mode == "tokens" else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return loader


if __name__ == "__main__":
    CACHE_DIR = "./Chr19_chr20_gsm3271348_3271349_cached"

    print("=" * 70)
    print("MULTIMODAL DATA LOADER - CACHE ONLY")
    print("=" * 70)

    train_loader, val_loader, test_loader = create_dataloaders(
        cache_dir=CACHE_DIR,
        batch_size=8,
        num_workers=0,
        enformer_mode="mean",  # or "tokens"
        atac_binarize=False,
    )

    batch = next(iter(train_loader))
    if len(batch) == 4:
        hic, enf, atac, rna = batch
        print("\nSample batch (mean mode)")
        print(f"  hic: {hic.shape}")
        print(f"  enformer: {enf.shape}")
        print(f"  atac: {atac.shape}")
        print(f"  rna: {rna.shape}")
    else:
        hic, enf, enf_mask, atac, rna = batch
        print("\nSample batch (tokens mode)")
        print(f"  hic: {hic.shape}")
        print(f"  enformer: {enf.shape}")
        print(f"  enformer_mask: {enf_mask.shape}")
        print(f"  atac: {atac.shape}")
        print(f"  rna: {rna.shape}")
