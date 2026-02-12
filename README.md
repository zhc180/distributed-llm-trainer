# Distributed LLM Trainer

A learning project for distributed LLM training, inspired by [Hugging Face's Nanotron](https://github.com/huggingface/nanotron).

## 🎯 Learning Goals

By building this project, you'll understand:
- **Data Parallelism (DP)** - Split batches across GPUs
- **Tensor Parallelism (TP)** - Split model layers across GPUs
- **Pipeline Parallelism (PP)** - Split model stages across GPUs
- **FSDP/ZeRO** - Shard optimizer states and gradients
- **Mixed Precision** - BF16/FP16 training
- **Gradient Checkpointing** - Trade compute for memory

## 📚 Recommended Learning Path

### Week 1: Foundations
1. Read the [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
2. Understand DDP basics with `01_ddp_basics.py`
3. Run single-GPU training first

### Week 2: Advanced Parallelism
1. Implement FSDP with `02_fsdp_training.py`
2. Add mixed precision and gradient checkpointing
3. Benchmark and compare approaches

## 🏗️ Project Structure

```
distributed-llm-trainer/
├── README.md
├── configs/
│   ├── small_model.yaml      # GPT-2 124M config
│   ├── medium_model.yaml     # GPT-2 355M config
│   └── fsdp_config.yaml      # FSDP settings
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gpt.py            # GPT model implementation
│   │   └── config.py         # Model configuration
│   ├── training/
│   │   ├── __init__.py
│   │   ├── ddp_trainer.py    # Basic DDP trainer
│   │   ├── fsdp_trainer.py   # FSDP trainer
│   │   └── trainer_utils.py  # Shared utilities
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataloader.py     # Distributed data loading
│   └── utils/
│       ├── __init__.py
│       ├── logging.py        # W&B / TensorBoard logging
│       └── checkpoint.py     # Distributed checkpointing
├── scripts/
│   ├── train_ddp.sh          # Launch DDP training
│   └── train_fsdp.sh         # Launch FSDP training
├── benchmarks/
│   └── results.md            # Your benchmark results
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers datasets wandb
pip install flash-attn --no-build-isolation  # Optional, for faster attention
```

### 2. Download Datasets (TinyStories + OpenWebText sample)

Run these commands from `src/data` so the files land in the data folder:

```bash
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
```

### 3. Single GPU Training (Start Here!)

```bash
python src/training/ddp_trainer.py --config configs/small_model.yaml
```

### 4. Multi-GPU DDP Training

```bash
torchrun --nproc_per_node=2 src/training/ddp_trainer.py --config configs/small_model.yaml
```

### 5. FSDP Training

```bash
torchrun --nproc_per_node=4 src/training/fsdp_trainer.py --config configs/medium_model.yaml
```

### 6. Choose a Dataset (dummy / tinystories / openwebtext)

```bash
# TinyStories
python src/training/ddp_trainer.py \
  --dataset tinystories \
  --data_path src/data/TinyStoriesV2-GPT4-train.txt \
  --model_size small --max_steps 50 --batch_size 4

# OpenWebText sample
python src/training/ddp_trainer.py \
  --dataset openwebtext \
  --data_path src/data/owt_train.txt \
  --model_size small --max_steps 50 --batch_size 4

# Dummy (default)
python src/training/ddp_trainer.py --dataset dummy --model_size small --max_steps 50
```

### 7. Inference from a Checkpoint

```bash
python src/eval/infer.py \
  --checkpoint checkpoints/final.pt \
  --model_size small \
  --prompt "Once upon a time," \
  --max_new_tokens 50
```

## 📊 Key Concepts Explained

### Data Parallelism (DDP)
```
GPU 0: Full Model + Batch 0 → Gradients 0 ─┐
GPU 1: Full Model + Batch 1 → Gradients 1 ─┼─→ All-Reduce → Update
GPU 2: Full Model + Batch 2 → Gradients 2 ─┤
GPU 3: Full Model + Batch 3 → Gradients 3 ─┘
```

### FSDP (Fully Sharded Data Parallel)
```
GPU 0: Shard 0 of (Model + Optimizer + Gradients)
GPU 1: Shard 1 of (Model + Optimizer + Gradients)
GPU 2: Shard 2 of (Model + Optimizer + Gradients)
GPU 3: Shard 3 of (Model + Optimizer + Gradients)
         ↓ All-Gather before forward/backward
         ↓ Reduce-Scatter after backward
```

### Memory Comparison (GPT-2 355M, FP32)

| Method | Model Memory | Optimizer Memory | Total/GPU |
|--------|-------------|------------------|-----------|
| DDP | 1.4 GB | 2.8 GB | 4.2 GB |
| FSDP (4 GPUs) | 0.35 GB | 0.7 GB | ~1.1 GB |

## 🔧 Configuration

### Model Config (`configs/small_model.yaml`)
```yaml
model:
  vocab_size: 50257
  hidden_size: 768
  num_layers: 12
  num_heads: 12
  max_seq_len: 1024

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 6e-4
  max_steps: 10000
  warmup_steps: 1000

distributed:
  backend: "nccl"
  mixed_precision: "bf16"
```

## 📈 Benchmarking

Track these metrics:
- **Throughput**: tokens/second
- **Memory**: peak GPU memory per device
- **Scaling efficiency**: throughput vs. # GPUs

```python
# Example benchmark output
"""
Config: GPT-2 124M, batch_size=8, seq_len=1024
─────────────────────────────────────────────
GPUs │ Method │ Tokens/sec │ Memory/GPU │ Efficiency
  1  │ Single │    12,500  │   8.2 GB   │   100%
  2  │ DDP    │    24,100  │   8.2 GB   │    96%
  4  │ DDP    │    46,800  │   8.2 GB   │    94%
  4  │ FSDP   │    44,200  │   3.1 GB   │    88%
"""
```

## 📖 Resources

- [Nanotron GitHub](https://github.com/huggingface/nanotron)
- [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)

## ✅ Project Milestones

- [ ] Single GPU training working
- [ ] DDP training with 2+ GPUs
- [ ] FSDP training implemented
- [ ] Mixed precision (BF16) added
- [ ] Gradient checkpointing added
- [ ] W&B logging integrated
- [ ] Benchmark results documented
- [ ] README with architecture diagrams
