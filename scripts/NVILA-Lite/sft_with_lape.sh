#!/bin/bash
# Modified sft.sh script with LAPE support
# Usage:
#   ./sft_with_lape.sh [STAGE_PATH] [DATA_MIXTURE] [OUTPUT_DIR]
# Environment variables:
#   ENABLE_LAPE=true/false (default: false)
#   NUM_SPATIAL_TOKENS=N (default: 100)
#   NUM_TEMPORAL_TOKENS=N (default: 100)

DEFAULT_RUN_NAME="vila-qwen2-vl-7b-sft"
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=32
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2

STAGE_PATH=${1:-"runs/train/nvila-8b-pretrain/model"}
DATA_MIXTURE=${2:-"nvila-pretrain"}
OUTPUT_DIR=${3:-"runs/train/nvila-8b-sft"}

# LAPE configuration (可选)
ENABLE_LAPE=${ENABLE_LAPE:-false}
NUM_SPATIAL_TOKENS=${NUM_SPATIAL_TOKENS:-100}
NUM_TEMPORAL_TOKENS=${NUM_TEMPORAL_TOKENS:-100}

# Ensure this repo's Python packages (llava, etc.) are preferred over any installed ones
export PYTHONPATH="$(pwd):${PYTHONPATH}"
echo "PYTHONPATH set to: $PYTHONPATH"

# Validate LAPE parameters
if [ "$ENABLE_LAPE" = "true" ]; then
    if ! [[ "$NUM_SPATIAL_TOKENS" =~ ^[0-9]+$ ]] || [ "$NUM_SPATIAL_TOKENS" -lt 1 ]; then
        echo "❌ Error: NUM_SPATIAL_TOKENS must be a positive integer"
        exit 1
    fi
    if ! [[ "$NUM_TEMPORAL_TOKENS" =~ ^[0-9]+$ ]] || [ "$NUM_TEMPORAL_TOKENS" -lt 1 ]; then
        echo "❌ Error: NUM_TEMPORAL_TOKENS must be a positive integer"
        exit 1
    fi
fi

source scripts/setups/train.sh

STAGE2_PATH=$1

# Print configuration summary
echo "===================================================="
echo "🔧 VILA SFT with LAPE Configuration"
echo "===================================================="
echo "📂 Stage Path: $STAGE_PATH"
echo "📊 Data Mixture: $DATA_MIXTURE"
echo "💾 Output Dir: $OUTPUT_DIR"
echo "🧠 LAPE Enabled: $ENABLE_LAPE"
if [ "$ENABLE_LAPE" = "true" ]; then
    echo "  🗺️  Spatial Tokens: $NUM_SPATIAL_TOKENS"
    echo "  ⏰ Temporal Tokens: $NUM_TEMPORAL_TOKENS"
fi
echo "===================================================="

# Build LAPE arguments conditionally
LAPE_ARGS=""
if [ "$ENABLE_LAPE" = "true" ]; then
    LAPE_ARGS="--enable_lape --num_spatial_tokens $NUM_SPATIAL_TOKENS --num_temporal_tokens $NUM_TEMPORAL_TOKENS"
    echo "🚀 LAPE enabled with spatial_tokens=$NUM_SPATIAL_TOKENS, temporal_tokens=$NUM_TEMPORAL_TOKENS"
else
    echo "📝 LAPE disabled (set ENABLE_LAPE=true to enable)"
fi

# Verify required files exist
if [ ! -d "$STAGE_PATH" ]; then
    echo "❌ Error: Stage path does not exist: $STAGE_PATH"
    exit 1
fi

if [ ! -f "scripts/zero3.json" ]; then
    echo "❌ Error: DeepSpeed config not found: scripts/zero3.json"
    exit 1
fi

# Check if we can find the training script
if [ ! -f "llava/train/train_mem.py" ]; then
    echo "❌ Error: Training script not found: llava/train/train_mem.py"
    exit 1
fi

# Warn about LAPE memory usage
if [ "$ENABLE_LAPE" = "true" ]; then
    echo "⚠️  WARNING: LAPE will increase memory usage due to additional embeddings"
    echo "   Recommended: Monitor GPU memory and adjust batch size if needed"
fi

echo "🚀 Starting training..."

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower ../VILA/.cache/huggingface/hub/models--Efficient-Large-Model--paligemma-siglip-so400m-patch14-448/snapshots/5d16503948d9699243d16e93fab44d2fa202371c \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --tune_vision_tower True \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 4096 \
        --gradient_checkpointing True \
        --dataloader_num_workers 16 \
        --vflan_no_system_prompt True \
        --report_to wandb \
        $LAPE_ARGS

# Training completion summary
echo ""
echo "===================================================="
echo "🎉 Training Completed!"
echo "===================================================="
echo "📂 Model saved to: $OUTPUT_DIR/model"
echo "🧠 LAPE was: $([ "$ENABLE_LAPE" = "true" ] && echo "ENABLED" || echo "DISABLED")"
if [ "$ENABLE_LAPE" = "true" ]; then
    echo "  🗺️  Used $NUM_SPATIAL_TOKENS spatial tokens"
    echo "  ⏰ Used $NUM_TEMPORAL_TOKENS temporal tokens"
fi
echo "===================================================="