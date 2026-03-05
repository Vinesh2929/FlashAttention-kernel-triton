"""
naive_attention.py

Just the raw attention math — no tricks, no tiling, nothing fancy.
This is what we're trying to beat with FlashAttention. The whole point
of this file is to be a correctness reference. Every output from the
Triton kernel should match what comes out of here.

The memory problem: for sequence length N and head dim d, we compute
a full (N x N) score matrix. That's N^2 elements living in HBM. At
N=8192 in float16 that's 128MB just for one head. FlashAttention never
materializes this — that's the whole idea.
"""

import torch
import math


def naive_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Standard scaled dot-product attention. No optimization whatsoever.

    Args:
        Q: query tensor of shape (batch, heads, seq_len, head_dim)
        K: key tensor of shape (batch, heads, seq_len, head_dim)
        V: value tensor of shape (batch, heads, seq_len, head_dim)

    Returns:
        output tensor of shape (batch, heads, seq_len, head_dim)

    The math is:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
    """
    # d is the head dimension — we scale by sqrt(d) to keep variance stable
    # without this, for large d the dot products get huge and softmax saturates
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    # QK^T — this is the (N x N) matrix that kills memory
    # shape: (batch, heads, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # softmax over the last dimension (keys dimension)
    # subtract max before exp for numerical stability — same idea as FlashAttention
    # torch.softmax already does this internally, so we're fine here
    attn_weights = torch.softmax(scores, dim=-1)

    # weighted sum of values
    # shape: (batch, heads, seq_len, head_dim)
    output = torch.matmul(attn_weights, V)

    return output


if __name__ == "__main__":
    # quick sanity check — just making sure the shapes are right
    torch.manual_seed(42)

    batch, heads, seq_len, head_dim = 2, 4, 128, 64

    # use float32 for the naive reference since we care about precision here
    Q = torch.randn(batch, heads, seq_len, head_dim)
    K = torch.randn(batch, heads, seq_len, head_dim)
    V = torch.randn(batch, heads, seq_len, head_dim)

    out = naive_attention(Q, K, V)

    print(f"Q shape:      {Q.shape}")
    print(f"K shape:      {K.shape}")
    print(f"V shape:      {V.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output mean:  {out.mean().item():.6f}")
    print(f"Output std:   {out.std().item():.6f}")
    print()
    print("naive attention looks good — output shape matches input shape")
