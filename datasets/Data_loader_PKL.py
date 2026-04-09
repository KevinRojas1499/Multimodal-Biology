#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multimodal Data Loader for Chr20 Dataset
========================================
Loads pre-processed Hi-C, ATAC-seq, and RNA-seq data along with Enformer embeddings.


Key Features:
- Loads all three modalities (HiC, ATAC, RNA) aligned to the same genomic regions
- Integrates Enformer embeddings for sequence context
- Returns PyTorch tensors ready for model training
- Provides data statistics and information helpers
- Supports easy train/val/test splits
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from torch.utils.data import Dataset, DataLoader, random_split

# ===================== Utility Functions =====================

def safe_pickle_load(fp):
    """Load pickle file with numpy/pandas compatibility handling."""
    try:
        with open(fp, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError):
        # Handle numpy version incompatibilities
        import numpy as np
        if not hasattr(np, "_core"):
            np.core = np.core
        import sys
        sys.modules.setdefault("numpy._core", np.core)
        with open(fp, "rb") as f:
            return pickle.load(f)


class EnformerLookup:
    """
    Fast lookup for pre-computed Enformer embeddings from TSV.
    
    Args:
        tsv_path: Path to .tsv.gz file with columns:
                  chrom | start | end | emb_0 | emb_1 | ... | emb_{D-1}
        pool_method: 'mean' or 'max' for aggregating overlapping bins
    """
    def __init__(self, tsv_path: str, pool_method: str = 'mean'):
        print(f"📚 Loading Enformer embeddings from: {tsv_path}")
        
        if tsv_path.endswith('.gz'):
            self.df = pd.read_csv(tsv_path, sep='\t', compression='gzip')
        else:
            self.df = pd.read_csv(tsv_path, sep='\t')
        
        # Infer embedding dimension
        emb_cols = [c for c in self.df.columns if c.startswith('emb_')]
        self.embed_dim = len(emb_cols)
        self.emb_cols = emb_cols
        
        # Resolution (bin size in bp)
        self.resolution = int(self.df['end'].iloc[0] - self.df['start'].iloc[0])
        
        assert pool_method in ['mean', 'max'], f"Invalid pool_method: {pool_method}"
        self.pool_method = pool_method
        
        # Pre-convert to numpy for fast indexing
        self.embeddings = self.df[emb_cols].values.astype(np.float32)  # (N, D)
        self.starts = self.df['start'].values
        self.ends = self.df['end'].values
        
        print(f"   ✅ Loaded {len(self.df):,} bins × {self.embed_dim} dims")
        print(f"   📏 Resolution: {self.resolution} bp | Pool: {pool_method}")
        print(f"   🧬 Coverage: chr{self.df['chrom'].iloc[0]} "
              f"{self.starts[0]:,} - {self.ends[-1]:,} bp")
    
    def get_embedding(self, chrom: str, start: int, end: int) -> np.ndarray:
        """
        Extract and pool embeddings for a genomic window.
        
        Args:
            chrom: chromosome name (e.g., 'chr20' or '20')
            start: window start (bp, 0-based)
            end: window end (bp, exclusive)
        
        Returns:
            pooled_embedding: (embed_dim,) numpy array
        """
        # Normalize chromosome name - ensure single 'chr' prefix
        chrom = str(chrom).strip()
        if chrom.startswith('chrchr'):
            chrom = chrom[3:]  # Remove duplicate 'chr'
        elif not chrom.startswith('chr'):
            chrom = f'chr{chrom}'
        
        # Find overlapping bins
        mask = (self.starts < end) & (self.ends > start)
        overlapping_embs = self.embeddings[mask]
        
        if len(overlapping_embs) == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        
        # Pool across bins
        if self.pool_method == 'mean':
            return overlapping_embs.mean(axis=0)
        else:
            return overlapping_embs.max(axis=0)
    
    def get_embedding_batch(self, coords: List[Tuple[str, int, int]]) -> np.ndarray:
        """Extract embeddings for multiple windows."""
        return np.stack([self.get_embedding(c, s, e) for c, s, e in coords], axis=0)


# ====================== Main Dataset Class ======================

class Chr20MultimodalDataset(Dataset):
    """
    PyTorch Dataset for aligned Hi-C, ATAC-seq, RNA-seq, and Enformer embeddings.
    
    All modalities are aligned to the same genomic regions on chromosome 20.
    
    Returns (in this order):
        hic:      (1, 100, 100) - log-normalised + sigmoid distance matrix (symmetric, diagonal=0)
        enformer: (D,)          - pooled Enformer embedding (sequence context)
        atac:     (400, 2)      - ATAC accessibility (binary: > 0)
        rna:      (400, 1)      - RNA expression (raw continuous values)
    """
    
    def __init__(self, cache_dir: str, enformer_tsv: str,
                 window_size: int = 500000, pool_method: str = 'mean'):
        """
        Args:
            cache_dir: Directory containing pickle files
            enformer_tsv: Path to Enformer embeddings TSV.GZ
            window_size: Genomic window size (bp) for extracting embeddings
            pool_method: 'mean' or 'max' for pooling Enformer bins
        """
        print("🔄 Loading multimodal dataset...")
        cache_dir = Path(cache_dir)
        assert cache_dir.exists(), f"Cache dir not found: {cache_dir}"

        # Load raw pickle files
        hic  = safe_pickle_load(cache_dir / "hic_cache_chr20_w100_coords.pkl")
        atac = safe_pickle_load(cache_dir / "atac_cache_chr20_5kb_bins_corrected.pkl")
        rna  = safe_pickle_load(cache_dir / "rnaseq_cache_chr20_5kb_bins.pkl")

        # Handle both old dictionary format and new direct array format
        if isinstance(hic, dict):
            self.hic_coords  = hic["hic_coordinates"].astype(np.float32)             # (N,100,3)
            self.genomic_coords = hic.get("genomic_coordinates", None)               # (N,3)
            self.atac_mats   = atac["atac_matrices"].astype(np.float32)              # (N,400,2)
            self.rna_mats    = (rna["rna_matrices"] if isinstance(rna, dict) else rna).astype(np.float32)
            print("📂 Loaded old cache format (dictionaries)")
        else:
            self.hic_coords  = hic.astype(np.float32)   # (N,100,3)
            self.genomic_coords = None
            self.atac_mats   = atac.astype(np.float32)  # (N,400,2)
            self.rna_mats    = rna.astype(np.float32)   # (N,400,1)
            print("📂 Loaded new cache format (numpy arrays)")

        assert len(self.hic_coords) == len(self.atac_mats) == len(self.rna_mats), (
            f"HiC/ATAC/RNA sample counts mismatch: "
            f"{len(self.hic_coords)}, {len(self.atac_mats)}, {len(self.rna_mats)}"
        )

        self.n_samples  = len(self.hic_coords)
        self.window_size = window_size

        # Load Enformer embeddings
        self.enformer_lookup = EnformerLookup(enformer_tsv, pool_method=pool_method)

        # Genomic coordinates: must exist in cache or as separate file
        if self.genomic_coords is None:
            coords_file = cache_dir / "chr20_genomic_coords.pkl"
            if coords_file.exists():
                self.genomic_coords = safe_pickle_load(coords_file)
                print(f"   ✅ Loaded genomic coordinates: {self.genomic_coords.shape}")
            else:
                raise FileNotFoundError(
                    f"Genomic coordinates required but not found!\n"
                    f"Expected in HiC cache dict or at: {coords_file}"
                )

        print(f"✅ Dataset loaded: {self.n_samples:,} samples with Enformer embeddings "
              f"({self.enformer_lookup.embed_dim}D)")
    
    @staticmethod
    def _coords_to_dist(coords: np.ndarray) -> np.ndarray:
        """Convert (100,3) 3-D coordinates to (100,100) log-normalised + sigmoid distance matrix."""
        dif = coords[:, None, :] - coords[None, :, :]
        d   = np.linalg.norm(dif, axis=-1)
        lt  = np.log1p(d)
        mn, mx = lt.min(), lt.max()
        if mx > mn:
            lt = (lt - mn) / (mx - mn) * 10 - 5
        return (1.0 / (1.0 + np.exp(-lt))).astype(np.float32)
    
    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single sample.

        Returns (same order as the MVAE training loop):
            hic:      (1, 100, 100) distance matrix
            enformer: (D,)          Enformer embedding
            atac:     (400, 2)      binary ATAC (> 0)
            rna:      (400, 1)      continuous RNA
        """
        # HiC: convert 3-D coordinates → distance matrix
        hic = self._coords_to_dist(self.hic_coords[idx])  # (100, 100)

        # ATAC: binarise at zero (any signal = 1)
        atac = (self.atac_mats[idx] > 0).astype(np.float32)  # (400, 2)

        # RNA: keep raw continuous values
        rna = self.rna_mats[idx].astype(np.float32)  # (400, 1)

        # Enformer: re-centre the window on the HiC window midpoint
        chrom, start, end = self.genomic_coords[idx]
        center       = (int(start) + int(end)) // 2
        window_start = max(0, center - self.window_size // 2)
        window_end   = window_start + self.window_size
        enformer     = self.enformer_lookup.get_embedding(chrom, window_start, window_end)  # (D,)

        return (
            torch.from_numpy(hic).unsqueeze(0),      # (1, 100, 100)
            torch.from_numpy(enformer),              # (D,)
            torch.from_numpy(atac),                  # (400, 2)
            torch.from_numpy(rna),                   # (400, 1)
        )
    
    def get_data_info(self) -> Dict:
        """Return dataset information."""
        return {
            'n_samples': self.n_samples,
            'hic_coords_shape': self.hic_coords.shape,
            'atac_shape': self.atac_mats.shape,
            'rna_shape': self.rna_mats.shape,
            'enformer_dim': self.enformer_lookup.embed_dim,
            'enformer_resolution': self.enformer_lookup.resolution,
            'window_size': self.window_size,
        }


# ==================== Helper Functions ====================

def create_dataloaders(cache_dir: str, enformer_tsv: str,
                       batch_size: int = 8,
                       train_split: float = 0.7,
                       val_split: float = 0.15,
                       test_split: float = 0.15,
                       num_workers: int = 0,
                       pin_memory: bool = True,
                       seed: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        cache_dir: Directory with pickle files
        enformer_tsv: Path to Enformer embeddings TSV
        batch_size: Batch size
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for GPU
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create dataset
    dataset = Chr20MultimodalDataset(cache_dir, enformer_tsv)
    
    # Compute split sizes
    n = len(dataset)
    train_size = int(train_split * n)
    val_size = int(val_split * n)
    test_size = n - train_size - val_size
    
    # Create splits
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"\n✅ Created dataloaders:")
    print(f"   Train: {len(train_ds)} samples ({train_split*100:.0f}%)")
    print(f"   Val:   {len(val_ds)} samples ({val_split*100:.0f}%)")
    print(f"   Test:  {len(test_ds)} samples ({test_split*100:.0f}%)")
    print(f"   Batch size: {batch_size} | Workers: {num_workers}")
    
    return train_loader, val_loader, test_loader


def load_full_dataset(cache_dir: str, enformer_tsv: str, 
                      batch_size: int = 8, 
                      num_workers: int = 0) -> DataLoader:
    """
    Load entire dataset as a single dataloader (no train/val/test split).
    
    Args:
        cache_dir: Directory with pickle files
        enformer_tsv: Path to Enformer embeddings TSV
        batch_size: Batch size
        num_workers: Number of workers
    
    Returns:
        dataloader for full dataset
    """
    dataset = Chr20MultimodalDataset(cache_dir, enformer_tsv)
    print(f"\n✅ Full dataset loaded: {len(dataset)} samples")
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return loader


# ======================= Example Usage =======================

if __name__ == "__main__":
    # Example: Load data with train/val/test splits
    CACHE_DIR = "./Chr20_all_cached"
    ENFORMER_TSV = "./Chr20_all_cached/enfemb_chr20_res1000_dim512.tsv.gz"
    
    print("=" * 70)
    print("MULTIMODAL DATA LOADER - Example Usage")
    print("=" * 70)
    
    # Option 1: Separate dataloaders
    print("\n📦 Creating train/val/test dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        cache_dir=CACHE_DIR,
        enformer_tsv=ENFORMER_TSV,
        batch_size=8,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        num_workers=0
    )
    
    # Inspect one batch  (note: return order = hic, enf, atac, rna)
    print("\n🔍 Sample batch from train set:")
    hic, enf, atac, rna = next(iter(train_loader))
    print(f"   HiC:      {hic.shape}  | dtype: {hic.dtype}")
    print(f"   Enformer: {enf.shape}  | dtype: {enf.dtype}")
    print(f"   ATAC:     {atac.shape} | dtype: {atac.dtype}")
    print(f"   RNA:      {rna.shape}  | dtype: {rna.dtype}")
    print(f"   Values - HiC: [{hic.min():.3f}, {hic.max():.3f}] | "
          f"ATAC: [{atac.min():.3f}, {atac.max():.3f}] | "
          f"RNA: [{rna.min():.3f}, {rna.max():.3f}]")
    
    # Option 2: Full dataset (no split)
    print("\n📦 Loading full dataset without splits...")
    full_loader = load_full_dataset(CACHE_DIR, ENFORMER_TSV, batch_size=16, num_workers=0)
    
    print("\n" + "=" * 70)
    print("✅ Data loader ready for model building!")
    print("=" * 70)
