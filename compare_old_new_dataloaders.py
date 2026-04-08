from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Data_loader_PKL import Chr20MultimodalDataset, load_full_dataset as load_old_full_dataset
from Data_loader_large_dataset import Chr19Chr20MultimodalDataset, load_full_dataset as load_new_full_dataset


OLD_CACHE_DIR = Path("./Chr20_all_cached")
OLD_ENFORMER_TSV = OLD_CACHE_DIR / "enfemb_chr20_res1000_dim512.tsv.gz"
NEW_CACHE_DIR = Path("./Chr19_chr20_gsm3271348_3271349_cached")
BATCH_SIZE = 4
OUT_PLOT = Path("./compare_old_new_dataloaders_modalities.png")


def _plot_sample_row(ax_row, hic, enf, atac, rna, title_prefix):
    hic_img = hic[0].detach().cpu().numpy()
    enf_vec = enf.detach().cpu().numpy()
    atac_mat = atac.detach().cpu().numpy()
    rna_vec = rna.detach().cpu().numpy().squeeze(-1)

    im0 = ax_row[0].imshow(hic_img, cmap="viridis", origin="lower", aspect="auto")
    ax_row[0].set_title(f"{title_prefix} Hi-C")
    ax_row[0].set_xlabel("Bead index")
    ax_row[0].set_ylabel("Bead index")
    plt.colorbar(im0, ax=ax_row[0], fraction=0.046, pad=0.04)

    ax_row[1].plot(atac_mat[:, 0], color="tab:orange", linewidth=1.0, label="ATAC ch0")
    ax_row[1].plot(atac_mat[:, 1], color="tab:purple", linewidth=1.0, alpha=0.8, label="ATAC ch1")
    ax_row[1].set_title(f"{title_prefix} ATAC")
    ax_row[1].set_xlabel("Bin index")
    ax_row[1].set_ylabel("Signal")
    ax_row[1].grid(alpha=0.2)
    ax_row[1].legend(loc="upper right", fontsize=8)

    ax_row[2].plot(rna_vec, color="seagreen", linewidth=1.0)
    ax_row[2].set_title(f"{title_prefix} RNA")
    ax_row[2].set_xlabel("Bin index")
    ax_row[2].set_ylabel("Signal")
    ax_row[2].grid(alpha=0.2)

    ax_row[3].plot(enf_vec, color="steelblue", linewidth=1.0)
    ax_row[3].set_title(f"{title_prefix} Enformer")
    ax_row[3].set_xlabel("Embedding dim")
    ax_row[3].set_ylabel("Value")
    ax_row[3].grid(alpha=0.2)


def summarize_modalities(hic, atac, rna, label):
    print(f"{label} Hi-C range:  {float(hic.min()):.4f} .. {float(hic.max()):.4f}")
    print(f"{label} ATAC range:  {float(atac.min()):.4f} .. {float(atac.max()):.4f}")
    print(f"{label} RNA range:   {float(rna.min()):.4f} .. {float(rna.max()):.4f}")


def main():
    print("Loading dataset info...")
    print()

    old_dataset = Chr20MultimodalDataset(str(OLD_CACHE_DIR), str(OLD_ENFORMER_TSV))
    new_dataset = Chr19Chr20MultimodalDataset(str(NEW_CACHE_DIR))

    print("Old loader info:")
    print(old_dataset.get_data_info())
    print()

    print("New loader info:")
    print(new_dataset.get_data_info())
    print()

    old_loader = load_old_full_dataset(
        str(OLD_CACHE_DIR),
        str(OLD_ENFORMER_TSV),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )
    new_loader = load_new_full_dataset(
        str(NEW_CACHE_DIR),
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    old_hic, old_enf, old_atac, old_rna = next(iter(old_loader))
    new_hic, new_enf, new_atac, new_rna = next(iter(new_loader))

    print("Old batch shapes:")
    print("  hic     ", tuple(old_hic.shape))
    print("  enformer", tuple(old_enf.shape))
    print("  atac    ", tuple(old_atac.shape))
    print("  rna     ", tuple(old_rna.shape))
    print()

    print("New batch shapes:")
    print("  hic     ", tuple(new_hic.shape))
    print("  enformer", tuple(new_enf.shape))
    print("  atac    ", tuple(new_atac.shape))
    print("  rna     ", tuple(new_rna.shape))
    print()

    summarize_modalities(old_hic, old_atac, old_rna, "Old batch")
    summarize_modalities(new_hic, new_atac, new_rna, "New batch")
    print()

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    _plot_sample_row(axes[0], old_hic[0], old_enf[0], old_atac[0], old_rna[0], "Old")
    _plot_sample_row(axes[1], new_hic[0], new_enf[0], new_atac[0], new_rna[0], "New")
    plt.tight_layout()
    fig.savefig(OUT_PLOT, dpi=150)
    plt.close(fig)
    print(f"Saved modality comparison figure to: {OUT_PLOT.resolve()}")


if __name__ == "__main__":
    main()
