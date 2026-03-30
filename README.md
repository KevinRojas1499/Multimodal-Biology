# Multimodal Biology

## Setup

Install dependencies with [uv](https://docs.astral.sh/uv/) (Python 3.12+):

```bash
uv sync
```

## Train Hi-C VAE

Log in to Weights & Biases once per machine (`wandb login`), then:

```bash
uv run train_vae.py --dir experiments/vae-hic --use-wandb
```

Checkpoints and outputs go under `--dir` (e.g. `experiments/vae-hic`). Use `uv run train_vae.py --help` for all options.
