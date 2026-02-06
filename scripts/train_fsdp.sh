#!/bin/bash
# FSDP Training Launch Script
#
# Usage:
#   ./scripts/train_fsdp.sh                   # Use all GPUs, medium model
#   ./scripts/train_fsdp.sh 4                 # Use 4 GPUs
#   ./scripts/train_fsdp.sh 4 large           # 4 GPUs, large model
#   ./scripts/train_fsdp.sh 4 xl --cpu_offload  # With CPU offloading

NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}
MODEL_SIZE=${2:-medium}
shift 2 2>/dev/null || shift $# 2>/dev/null
EXTRA_ARGS="$@"

echo "=========================================="
echo "FSDP Training"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Model: $MODEL_SIZE"
echo "Extra args: $EXTRA_ARGS"
echo "=========================================="

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export OMP_NUM_THREADS=8

# Recommended NCCL settings for multi-GPU
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available

# Launch FSDP training
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    src/training/fsdp_trainer.py \
    --model_size $MODEL_SIZE \
    --batch_size 4 \
    --max_steps 1000 \
    --sharding FULL_SHARD \
    $EXTRA_ARGS

echo "Training complete!"
