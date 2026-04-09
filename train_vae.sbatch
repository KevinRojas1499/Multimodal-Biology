#!/bin/bash
#SBATCH --job-name=hic_vae
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "CUDA visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
PY

mkdir -p experiments/vae-hic

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-Multimodal_genomics}"
WANDB_ENTITY="${WANDB_ENTITY:-vsvivek99-university-of-pennsylvania}"
WANDB_GROUP="${WANDB_GROUP:-hic_vae_grid_longer_training}"

# Optional training overrides; keep aligned with train_vae.py defaults.
STARTING_RES="${STARTING_RES:-100}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
LATENT_CHANNELS="${LATENT_CHANNELS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
AL="${AL:-0.99999}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-hic_vae_b${NUM_BLOCKS}_z${LATENT_CHANNELS}_$(date +%Y%m%d_%H%M)}"

# Use a unique default output directory per run to avoid overwriting checkpoints/reconstructions.
RUN_STAMP="${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S)"
RUN_TAG_RAW="${WANDB_RUN_NAME:-hic_vae_${RUN_STAMP}}"
RUN_TAG="$(echo "$RUN_TAG_RAW" | tr -cs '[:alnum:]_.-' '_')"
OUT_DIR="${OUT_DIR:-experiments/vae-hic/${RUN_TAG}}"

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  echo "W&B logging: enabled"
  echo "W&B entity: $WANDB_ENTITY"
  echo "W&B project: $WANDB_PROJECT"
  echo "W&B group: $WANDB_GROUP"
  echo "W&B run name: $WANDB_RUN_NAME"
  WANDB_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [[ -n "$WANDB_ENTITY" ]]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
  if [[ -n "$WANDB_GROUP" ]]; then
    WANDB_ARGS+=(--wandb-group "$WANDB_GROUP")
  fi
  if [[ -n "$WANDB_RUN_NAME" ]]; then
    WANDB_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  fi
else
  echo "W&B logging: disabled"
fi

mkdir -p "$OUT_DIR"

echo "Training config: starting_res=$STARTING_RES num_blocks=$NUM_BLOCKS latent_channels=$LATENT_CHANNELS batch_size=$BATCH_SIZE num_epochs=$NUM_EPOCHS al=$AL out_dir=$OUT_DIR"

python -u train_vae.py \
  --cache_dir ./Chr20_all_cached \
  --enformer_tsv ./Chr20_all_cached/enfemb_chr20_res1000_dim512.tsv.gz \
  --starting_res "$STARTING_RES" \
  --num_blocks "$NUM_BLOCKS" \
  --latent_channels "$LATENT_CHANNELS" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --al "$AL" \
  --dir "$OUT_DIR" \
  "${WANDB_ARGS[@]}"
