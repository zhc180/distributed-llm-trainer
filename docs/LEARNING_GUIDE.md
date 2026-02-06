# Learning Guide: Distributed LLM Training

## Implementation Order

Work through these exercises in order. Each builds on the last.
Test after each step with `python src/models/gpt.py`.

### Phase 1: Model Fundamentals (src/models/gpt.py)

Start here. These teach core PyTorch and transformer concepts.

#### Exercise 1: RMSNorm.forward()
- **File:** `src/models/gpt.py` → class `RMSNorm`
- **Concept:** Tensor operations, broadcasting, normalization
- **Difficulty:** Easy (1 line)
- **Test:** You can't test in isolation yet, but the math is simple:
  `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight`
- **Read first:** Why normalization matters in deep nets

#### Exercise 2: rotate_half()
- **File:** `src/models/gpt.py` → function `rotate_half`
- **Concept:** Tensor slicing for RoPE
- **Difficulty:** Easy (3 lines)
- **Test:** `rotate_half(torch.tensor([1, 2, 3, 4]))` should give `[-3, -4, 1, 2]`

#### Exercise 3: apply_rotary_pos_emb()
- **File:** `src/models/gpt.py` → function `apply_rotary_pos_emb`
- **Concept:** Rotary position encoding
- **Difficulty:** Easy (2 lines, once you understand the formula)
- **Read first:** RoPE paper (Su et al. 2021) or any blog explaining RoPE
- **Depends on:** Exercise 2

#### Exercise 4: MLP.forward() (SwiGLU)
- **File:** `src/models/gpt.py` → class `MLP`
- **Concept:** Gated activations, modern MLP design
- **Difficulty:** Easy (1 line)
- **Key insight:** SwiGLU = two parallel projections, one gated by SiLU

#### Exercise 5: Scaled Dot-Product Attention
- **File:** `src/models/gpt.py` → `CausalSelfAttention.forward()` (else branch)
- **Concept:** THE core mechanism of transformers
- **Difficulty:** Medium (5-8 lines)
- **Key insight:** Q @ K^T gives similarity scores, softmax normalizes,
  causal mask prevents cheating by looking at future tokens
- **Depends on:** Exercise 3 (RoPE is applied before attention)

#### Exercise 6: GPT._init_weights()
- **File:** `src/models/gpt.py` → class `GPT`
- **Concept:** Weight initialization, isinstance() pattern
- **Difficulty:** Easy (5 lines)
- **Key insight:** Bad init = training won't converge

#### Exercise 7: GPT.forward()
- **File:** `src/models/gpt.py` → class `GPT`
- **Concept:** Full model pipeline, gradient checkpointing, loss computation
- **Difficulty:** Medium (15-20 lines)
- **Key insight:** The shift in loss computation (predict NEXT token)
- **Depends on:** All previous exercises
- **Checkpoint:** After this, `python src/models/gpt.py` should run!

```bash
# TEST: This should print logits shape and loss value
python src/models/gpt.py
```

---

### Phase 2: Training Loop (src/training/ddp_trainer.py)

Now you have a working model. Time to train it.

#### Exercise 8: _setup_optimizer()
- **File:** `src/training/ddp_trainer.py` → class `DistributedTrainer`
- **Concept:** Parameter groups, weight decay, AdamW
- **Difficulty:** Easy-Medium (10-15 lines)
- **Key insight:** Not all parameters should be regularized

#### Exercise 9: get_lr() (Cosine Schedule)
- **File:** `src/training/ddp_trainer.py` → class `DistributedTrainer`
- **Concept:** Learning rate scheduling
- **Difficulty:** Easy (5-8 lines)
- **Key insight:** Warmup avoids early instability, cosine is smooth decay

#### Exercise 10: train_step() (Gradient Accumulation)
- **File:** `src/training/ddp_trainer.py` → class `DistributedTrainer`
- **Concept:** The full training step - this is where it all comes together
- **Difficulty:** Hard (20-30 lines)
- **Key insight:** Gradient accumulation = poor man's large batch
- **Depends on:** Exercises 8 and 9

#### Exercise 11: save_checkpoint() and load_checkpoint()
- **File:** `src/training/ddp_trainer.py` → class `DistributedTrainer`
- **Concept:** State dicts, DDP unwrapping, training resumption
- **Difficulty:** Easy-Medium (10 lines each)
- **Key insight:** DDP wraps model in .module, must unwrap before saving

```bash
# TEST: Single GPU training (no torchrun needed)
python src/training/ddp_trainer.py --model_size small --max_steps 50
```

---

### Phase 3: FSDP (src/training/fsdp_trainer.py)

The advanced stuff. Make sure DDP works first.

#### Exercise 12: _get_sharding_strategy()
- **File:** `src/training/fsdp_trainer.py` → class `FSDPTrainer`
- **Concept:** ZeRO stages, memory vs communication tradeoff
- **Difficulty:** Easy (5 lines - it's a dict lookup)
- **Key insight:** Understand WHAT each strategy shards

#### Exercise 13: _get_mixed_precision_policy()
- **File:** `src/training/fsdp_trainer.py` → class `FSDPTrainer`
- **Concept:** Mixed precision, BF16 vs FP16
- **Difficulty:** Easy (10 lines - if/elif/return)
- **Key insight:** BF16 is preferred for training (no loss scaling)

#### Exercise 14: _get_fsdp_policy()
- **File:** `src/training/fsdp_trainer.py` → class `FSDPTrainer`
- **Concept:** FSDP wrapping granularity
- **Difficulty:** Easy (3 lines)
- **Key insight:** Wrap at TransformerBlock level for optimal communication

#### Exercise 15: save_checkpoint() (FSDP)
- **File:** `src/training/fsdp_trainer.py` → class `FSDPTrainer`
- **Concept:** Distributed checkpointing, gathering sharded state
- **Difficulty:** Medium (15 lines)
- **Key insight:** Model is split across GPUs, must gather before saving
- **Depends on:** Exercises 12-14

```bash
# TEST: Requires multiple GPUs
torchrun --nproc_per_node=2 src/training/fsdp_trainer.py --model_size medium --max_steps 50
```

---

## Study Plan

### Day 1-2: Understand the Basics

**Read:**
1. [Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Sections 1-3
   - What is data parallelism?
   - What problem does distributed training solve?

2. Complete exercises 1-7 (model implementation)

### Day 3-4: DDP Deep Dive

**Read:**
1. [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
2. Ultrascale Playbook - Data Parallelism section

**Practice:** Complete exercises 8-11 (DDP trainer)

**Key Questions to Answer:**
- What is all-reduce? When does it happen?
- How does gradient synchronization work?
- What's the relationship between batch size and # GPUs?

### Day 5-7: FSDP Introduction

**Read:**
1. [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
2. Ultrascale Playbook - ZeRO section

**Practice:** Complete exercises 12-15 (FSDP trainer)

---

## 📊 Key Concepts Cheat Sheet

### Data Parallelism (DDP)
```
What: Replicate model on each GPU, split data
When: Model fits on single GPU
Memory: O(N) per GPU (N = model size)
Communication: All-reduce gradients after backward
```

### FSDP / ZeRO
```
What: Shard model, optimizer, gradients across GPUs
When: Model too large for single GPU
Memory: O(N/P) per GPU (P = # GPUs)
Communication: All-gather before forward/backward, reduce-scatter after
```

### ZeRO Stages
| Stage | Shards | Memory Savings |
|-------|--------|----------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~Nx (N = # GPUs) |

### Mixed Precision
```
BF16: Larger range, less precision. Best for training.
FP16: Smaller range, more precision. Needs loss scaling.
FP32: Full precision. Baseline, but slow.
```

### Gradient Checkpointing
```
What: Recompute activations instead of storing them
Trade-off: ~30% slower, ~70% less memory
When: Large models, limited GPU memory
```

---

## 🔧 Debugging Tips

### Out of Memory (OOM)
1. Reduce batch size
2. Enable gradient checkpointing
3. Use FSDP with `FULL_SHARD`
4. Enable CPU offloading

### Slow Training
1. Check GPU utilization: `nvidia-smi -l 1`
2. Profile with `torch.profiler`
3. Enable mixed precision
4. Check data loading (num_workers)

### NaN Loss
1. Lower learning rate
2. Enable gradient clipping
3. Check for numerical instability in attention
4. Use BF16 instead of FP16

---

## 📚 Essential Reading

### Papers
1. **ZeRO** - "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
2. **Megatron-LM** - "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
3. **GPipe** - "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism"

### Code References
1. [Nanotron](https://github.com/huggingface/nanotron) - Production distributed training
2. [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation
3. [Llama](https://github.com/meta-llama/llama) - Meta's LLM implementation

### Blogs
1. [Hugging Face - Ultrascale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook)
2. [PyTorch - FSDP Deep Dive](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
3. [Microsoft - DeepSpeed ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

---

## ✅ Week 1 Checklist

- [ ] Read Ultrascale Playbook sections 1-4
- [ ] Run single-GPU training successfully
- [ ] Run multi-GPU DDP training
- [ ] Run FSDP training
- [ ] Compare memory usage: DDP vs FSDP
- [ ] Measure throughput (tokens/sec)
- [ ] Document your benchmark results

---

## 🎯 Project Milestones

### Milestone 1: Basic Training (End of Week 1)
- [ ] Train GPT-2 Small on dummy data
- [ ] Multi-GPU DDP working
- [ ] Basic logging (loss, LR, throughput)

### Milestone 2: FSDP (End of Week 2)
- [ ] FSDP training working
- [ ] Activation checkpointing enabled
- [ ] Memory benchmarks documented
- [ ] Checkpoint save/load working

### Milestone 3: Optimization (Bonus)
- [ ] Mixed precision comparison (FP32 vs BF16)
- [ ] DeepSpeed integration
- [ ] Real dataset (OpenWebText or similar)
- [ ] W&B logging integration

---

## 💡 Interview Talking Points

After completing this project, you should be able to discuss:

1. **"Explain how DDP works"**
   - Gradient synchronization via all-reduce
   - Why batch size scales with # GPUs
   - Communication overhead

2. **"What is FSDP/ZeRO and when would you use it?"**
   - Sharding strategy trade-offs
   - Memory vs communication trade-off
   - When FSDP > DDP

3. **"How would you scale training to 1000 GPUs?"**
   - Tensor parallelism for large layers
   - Pipeline parallelism for depth
   - 3D parallelism strategies

4. **"What optimizations improve training throughput?"**
   - Mixed precision (BF16)
   - Gradient accumulation
   - Activation checkpointing
   - Communication overlap
