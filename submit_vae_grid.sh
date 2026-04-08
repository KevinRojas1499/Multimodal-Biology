#!/bin/bash
set -euo pipefail

group_name="${WANDB_GROUP:-hic_vae_grid}"
num_epochs="${NUM_EPOCHS:-100}"
use_wandb="${USE_WANDB:-1}"

configs=(
  "2 4 hic_b2_z4"
  "3 4 hic_b3_z4"
  "3 8 hic_b3_z8"
  "2 8 hic_b2_z8"
)

for config in "${configs[@]}"; do
  read -r num_blocks latent_channels run_prefix <<<"$config"
  run_name="${run_prefix}_${group_name}"

  echo "Submitting ${run_name} with NUM_BLOCKS=${num_blocks} LATENT_CHANNELS=${latent_channels} NUM_EPOCHS=${num_epochs}"

  USE_WANDB="$use_wandb" \
  WANDB_GROUP="$group_name" \
  WANDB_RUN_NAME="$run_name" \
  NUM_BLOCKS="$num_blocks" \
  LATENT_CHANNELS="$latent_channels" \
  NUM_EPOCHS="$num_epochs" \
  sbatch train_vae.sbatch
done