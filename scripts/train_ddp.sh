#!/bin/bash
# DDP Training Launch Script
#
# Usage:
#   ./scripts/train_ddp.sh                    # Use all available GPUs
#   ./scripts/train_ddp.sh 2                  # Use 2 GPUs
#   ./scripts/train_ddp.sh 4 medium           # 4 GPUs, medium model

NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}
MODEL_SIZE=${2:-small}

echo "=========================================="
echo "DDP Training"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL_SIZE"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export OMP_NUM_THREADS=8

# Launch training
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    src/training/ddp_trainer.py \
    --model_size $MODEL_SIZE \
    --batch_size 8 \
    --max_steps 1000 \
    --mixed_precision bf16

echo "Training complete!"
