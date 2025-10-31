# wandb offline mode
export WANDB_MODE=offline

# Set environment variables for the training script
DEFAULT_RUN_NAME="NVILA-Lite-8B-AgroSFT-lape" \
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=32 \
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2 \
ENABLE_LAPE=true \
NUM_SPATIAL_TOKENS=100 \
NUM_TEMPORAL_TOKENS=100 \
time bash scripts/NVILA-Lite/sft_with_lape.sh \
    ../VILA/.cache/huggingface/hub/models--Efficient-Large-Model--NVILA-Lite-8B/snapshots/ea3c8b6d50a417b6d5fed49a5d98f1a24c9f389d \
    agromind_lape \
    runs/train/nvila-8b-AgroSFT-lape
