"""GPT model implementation for distributed training.

This is a clean, educational implementation inspired by:
- Karpathy's nanoGPT
- Hugging Face's nanotron
- PyTorch's native transformer
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from .config import GPTConfig
except ImportError:
    from config import GPTConfig


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More stable than LayerNorm for large models.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using Root Mean Square.

        TODO: Implement RMSNorm forward pass

        RMSNorm is simpler than LayerNorm - it skips the mean-centering step.
        Used in LLaMA, Gemma, and most modern LLMs.

        LayerNorm:  (x - mean) / sqrt(var + eps) * weight + bias
        RMSNorm:    x / sqrt(mean(x^2) + eps) * weight

        Steps:
        1. Compute the RMS (root mean square) of x along the last dimension:
           - Square each element: x.pow(2)
           - Take mean along last dim (keepdim=True so broadcasting works)
           - Add self.eps for numerical stability
           - Take reciprocal square root: torch.rsqrt()

        2. Multiply: x * rms * self.weight

        Shape reference:
            x: [batch, seq_len, hidden_size]
            self.weight: [hidden_size]  (learnable scale parameter)
            output: same shape as x

        Why RMSNorm over LayerNorm?
        - Simpler (no mean subtraction, no bias)
        - Slightly faster
        - Empirically works just as well for transformers

        Hint: This can be done in one line!
              torch.rsqrt() = 1 / sqrt()
        """
        raise NotImplementedError("Implement RMSNorm forward")


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Better extrapolation than learned position embeddings.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    TODO: Implement rotation for RoPE

    Given input x of shape [..., dim], split it into two halves:
    - x1 = first half (indices 0 to dim//2)
    - x2 = second half (indices dim//2 to dim)

    Return: concatenation of (-x2, x1) along the last dimension

    Example: if x = [a, b, c, d], return [-c, -d, a, b]

    Hint: Use slicing x[..., :half] and torch.cat()
    """
    raise NotImplementedError("Implement rotate_half")


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys.

    TODO: Implement RoPE application

    The RoPE formula applies rotation in 2D subspaces:
        q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)

    This encodes position information by rotating the query/key vectors
    based on their position in the sequence.

    Args:
        q: Query tensor
        k: Key tensor
        cos: Cosine of rotation angles (precomputed)
        sin: Sine of rotation angles (precomputed)

    Returns:
        Tuple of (rotated_q, rotated_k)

    Hint: Apply the same rotation formula to both q and k
    """
    raise NotImplementedError("Implement apply_rotary_pos_emb")


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional Flash Attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        # Q, K, V projections (can be fused for efficiency)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Output projection
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        self.rotary_emb = RotaryPositionEmbedding(self.head_dim, config.max_seq_len)

        # Flash attention flag
        self.use_flash = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))

        # Attention
        if self.use_flash:
            # Use PyTorch's native Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # TODO: Implement manual scaled dot-product attention
            #
            # Steps:
            # 1. Compute attention scores: Q @ K^T
            #    - Use torch.matmul(q, k.transpose(-2, -1))
            #    - Scale by 1/sqrt(head_dim) for stable gradients
            #
            # 2. Apply causal mask (prevent attending to future tokens)
            #    - Create upper triangular mask with torch.triu(..., diagonal=1)
            #    - Fill masked positions with -inf (so softmax gives 0)
            #
            # 3. Apply softmax to get attention weights (dim=-1)
            #    - Tip: Use dtype=torch.float32 for numerical stability, then cast back
            #
            # 4. Apply dropout to attention weights (use self.attn_dropout)
            #
            # 5. Compute output: attention_weights @ V
            #
            # Shape reference:
            #   q, k, v: [batch, num_heads, seq_len, head_dim]
            #   attn_weights: [batch, num_heads, seq_len, seq_len]
            #   attn_output: [batch, num_heads, seq_len, head_dim]

            raise NotImplementedError("Implement manual attention computation")

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU activation function.

        TODO: Implement SwiGLU

        SwiGLU is a gated activation used in modern LLMs (LLaMA, PaLM, etc.)
        It's more expressive than plain ReLU or GELU.

        Formula: output = down_proj(silu(gate_proj(x)) * up_proj(x))

        Where:
        - gate_proj(x): Projects x to intermediate_size, then applies SiLU
        - up_proj(x): Projects x to intermediate_size (no activation)
        - The two are multiplied element-wise (the "gating" mechanism)
        - down_proj: Projects back to hidden_size
        - Don't forget dropout at the end!

        Hint: F.silu() is the SiLU/Swish activation function

        Why gating? The gate learns WHICH features to pass through,
        while up_proj learns WHAT values to pass. This separation
        gives the model more expressivity.
        """
        raise NotImplementedError("Implement SwiGLU forward pass")


class TransformerBlock(nn.Module):
    """Single transformer block with pre-normalization."""

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size)
        self.attention = CausalSelfAttention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GPT(nn.Module):
    """GPT model for causal language modeling."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.hidden_size)

        # Output head (tied with embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights using normal distribution.

        TODO: Implement weight initialization

        Proper initialization is CRITICAL for training stability.
        Bad init → exploding/vanishing gradients → training fails.

        This method is called via self.apply(self._init_weights), which
        recursively applies it to every submodule in the model.

        Rules:
        1. For nn.Linear layers:
           - Initialize weight with normal distribution:
             torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
           - If the layer has a bias, zero it out:
             torch.nn.init.zeros_(module.bias)

        2. For nn.Embedding layers:
           - Initialize weight with normal distribution (same std as Linear)

        3. All other module types: do nothing (skip them)

        Why normal distribution?
        - Keeps activation variance stable across layers
        - initializer_range (typically 0.02) is tuned for transformer scale

        Hint: Use isinstance() to check module type
              Use in-place init functions (the ones ending with _)
        """
        raise NotImplementedError("Implement _init_weights")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """GPT forward pass: tokens in → logits (and optionally loss) out.

        TODO: Implement the full forward pass

        This is where everything comes together. The pipeline is:
            input_ids → embedding → transformer layers → norm → lm_head → logits

        Steps:
        1. Token embedding:
           - hidden_states = self.embed_tokens(input_ids)
           - Converts token IDs [batch, seq_len] → vectors [batch, seq_len, hidden_size]

        2. Pass through transformer layers (self.layers):
           - Loop over each layer in self.layers
           - If gradient checkpointing is enabled AND we're training:
               hidden_states = checkpoint(layer, hidden_states, attention_mask,
                                          use_reentrant=False)
             (This trades memory for compute by recomputing activations in backward)
           - Otherwise: hidden_states = layer(hidden_states, attention_mask)

        3. Final normalization:
           - hidden_states = self.norm(hidden_states)

        4. Language model head:
           - logits = self.lm_head(hidden_states)
           - Projects [batch, seq_len, hidden_size] → [batch, seq_len, vocab_size]

        5. Loss computation (only if labels is not None):
           - For next-token prediction, we need to SHIFT:
             * shift_logits = logits[..., :-1, :]   (all positions except last)
             * shift_labels = labels[..., 1:]         (all positions except first)
           - Why shift? At position i, we predict token at position i+1
           - Use F.cross_entropy on the flattened tensors:
             * shift_logits.view(-1, vocab_size) → [batch*seq_len, vocab_size]
             * shift_labels.view(-1) → [batch*seq_len]
             * ignore_index=-100 skips padding tokens
           - If labels is None, set loss = None

        Returns:
            (logits, loss) - loss is None if labels not provided

        Hint: Check self.gradient_checkpointing and self.training for step 2
              Use .contiguous() before .view() to avoid memory layout errors
        """
        raise NotImplementedError("Implement GPT forward pass")

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple autoregressive generation."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop to max_seq_len if needed
                idx_cond = input_ids if input_ids.size(1) <= self.config.max_seq_len else input_ids[:, -self.config.max_seq_len:]

                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    config = GPTConfig.gpt2_small()
    model = GPT(config)

    print(f"Model config: {config}")
    print(f"Estimated parameters: {config.num_parameters():,}")
    print(f"Actual parameters: {count_parameters(model):,}")

    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    logits, loss = model(input_ids, labels=labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
