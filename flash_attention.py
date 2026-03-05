"""
flash_attention.py

The Triton kernel for FlashAttention forward pass. The key idea is tiling —
we process Q in blocks, and for each Q block we iterate over all K/V blocks.
We never write the full NxN attention matrix to HBM. Instead we keep running
statistics in SRAM (which is tiny but fast) and accumulate the output directly.

The three running stats we track per row of Q:
  m  = running max of the scores (for numerical stability)
  l  = running sum of exp(score - m) (the denominator of softmax)
  O  = running weighted sum of V (the actual output, unnormalized)

At the end, O / l gives us the true attention output. This is "online softmax"
— we never had to see all the scores at once to compute the softmax correctly.
"""

import torch
import triton
import triton.language as tl
import math


# autotune configs — triton benchmarks all of these and picks the winner for your GPU
# BLOCK_M = Q rows per program, BLOCK_N = K/V rows per inner loop step
# larger blocks = more SRAM usage but better arithmetic intensity
#
# num_stages is the big new addition here — it controls software pipelining
# with num_stages=1 the pattern is: load K tile → wait → compute → load V tile → wait → compute
# with num_stages=3 triton overlaps the next tile's load with the current tile's compute
# the GPU doesn't sit idle during memory fetches — it just keeps doing math
# num_stages=3 is usually the sweet spot; =4 helps when HBM latency is really high
autotune_configs = [
    # --- num_stages=2 (light pipelining, small SRAM footprint) ---
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=2),
    # --- num_stages=3 (deeper pipeline, next tile prefetched during current compute) ---
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 32,  "BLOCK_N": 64},  num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 32},  num_warps=8, num_stages=3),
    # --- num_stages=4 (max depth, pays off when memory latency dominates) ---
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64},  num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64},  num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64,  "BLOCK_N": 32},  num_warps=4, num_stages=4),
]


@triton.autotune(
    configs=autotune_configs,
    key=["N", "d", "causal"],  # re-tune when any of these change — causal/non-causal
                               # have different loop counts so they need separate tuning
)
@triton.jit
def _flash_attn_forward_kernel(
    Q_ptr,       # pointer to Q tensor
    K_ptr,       # pointer to K tensor
    V_ptr,       # pointer to V tensor
    O_ptr,       # pointer to output tensor
    # strides — how many elements to skip to move one step in each dimension
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N: tl.constexpr,          # sequence length
    d: tl.constexpr,          # head dimension
    scale,                    # 1 / sqrt(d) — passed in to avoid computing in kernel
    causal: tl.constexpr,     # NEW: whether to apply causal (lower-triangle) masking
                              # tl.constexpr means triton compiles two separate kernels —
                              # one for causal=True, one for False — zero runtime overhead
    BLOCK_M: tl.constexpr,    # rows of Q per program instance
    BLOCK_N: tl.constexpr,    # rows of K/V per inner loop step
):
    # figure out which (batch, head, Q-block) this program instance handles
    # each program handles BLOCK_M rows of Q for one (batch, head) pair
    pid_m  = tl.program_id(axis=0)   # which Q block
    pid_bh = tl.program_id(axis=1)   # which (batch, head) pair

    # decompose pid_bh into batch index and head index
    # we launch with total_heads = num_heads, so:
    #   pid_bh = batch_idx * num_heads + head_idx
    # but we just use it as a flat offset into the B*H dimension
    # the strides handle the actual memory layout

    # base offset into Q/K/V for this (batch, head)
    # pid_bh indexes into the flattened (batch * heads) dimension
    qk_base = pid_bh.to(tl.int64) * stride_qh
    k_base  = pid_bh.to(tl.int64) * stride_kh
    v_base  = pid_bh.to(tl.int64) * stride_vh
    o_base  = pid_bh.to(tl.int64) * stride_oh

    # the starting row of Q this block handles
    q_start = pid_m * BLOCK_M

    # offsets into the M (sequence) dimension for this Q block
    offs_m = q_start + tl.arange(0, BLOCK_M)   # shape: (BLOCK_M,)
    offs_d = tl.arange(0, d)                    # shape: (d,)

    # load the Q block into SRAM
    # mask out rows that go beyond the sequence length (handles N % BLOCK_M != 0)
    q_mask = offs_m[:, None] < N  # (BLOCK_M, 1) — broadcast over d
    Q_block_ptr = Q_ptr + qk_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    Q_block = tl.load(Q_block_ptr, mask=q_mask, other=0.0)  # (BLOCK_M, d)

    # initialize running statistics
    # m starts at -inf so any real score immediately becomes the new max
    m_block = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    # l starts at 0 — we haven't seen any exp scores yet
    l_block = tl.zeros((BLOCK_M,), dtype=tl.float32)
    # O starts at 0 — no weighted values accumulated yet
    O_block = tl.zeros((BLOCK_M, d), dtype=tl.float32)

    # inner loop: iterate over K and V blocks
    # for each K/V block, compute scores, update running stats, accumulate output
    #
    # causal mode cuts the loop roughly in half — row i can only attend to j <= i
    # the furthest column any Q row in this block can attend to is q_start + BLOCK_M - 1
    # so any kv block starting beyond that column is pure upper-triangle: all -inf, skip it
    # for a long sequence this is ~half the inner loop iterations just gone for free
    if causal:
        num_blocks_n = tl.cdiv(q_start + BLOCK_M, BLOCK_N)
    else:
        num_blocks_n = tl.cdiv(N, BLOCK_N)

    for block_n in range(num_blocks_n):
        kv_start = block_n * BLOCK_N
        offs_n = kv_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)

        # load K block — shape (BLOCK_N, d)
        kv_mask = offs_n[:, None] < N
        K_block_ptr = K_ptr + k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        K_block = tl.load(K_block_ptr, mask=kv_mask, other=0.0)  # (BLOCK_N, d)

        # load V block — shape (BLOCK_N, d)
        V_block_ptr = V_ptr + v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        V_block = tl.load(V_block_ptr, mask=kv_mask, other=0.0)  # (BLOCK_N, d)

        # compute raw scores: S = Q @ K^T / sqrt(d)
        # Q_block: (BLOCK_M, d), K_block: (BLOCK_N, d)
        # S shape: (BLOCK_M, BLOCK_N)
        S = tl.dot(Q_block, tl.trans(K_block)) * scale

        # mask out positions beyond sequence length so they don't affect the max
        # this is critical for correctness when N % BLOCK_N != 0
        col_mask = offs_n[None, :] < N  # (1, BLOCK_N)
        S = tl.where(col_mask, S, float("-inf"))

        # causal mask — within each tile, zero out positions where j > i
        # the tile-skipping above already eliminated fully upper-triangle blocks,
        # but the "diagonal" tile (where kv_start <= q_start < kv_start + BLOCK_N)
        # is a mix: some (i, j) pairs are valid and some aren't
        # this per-element mask handles that boundary correctly
        # since causal is tl.constexpr, this whole branch is compiled away for non-causal
        if causal:
            # offs_m[i] is the absolute row position, offs_n[j] is the absolute col position
            # we want to keep j <= i, so mask out j > i
            causal_mask = offs_n[None, :] <= offs_m[:, None]  # (BLOCK_M, BLOCK_N)
            S = tl.where(causal_mask, S, float("-inf"))

        # online softmax update
        # m_new = max(m_old, rowmax(S))
        m_new = tl.maximum(m_block, tl.max(S, axis=1))  # (BLOCK_M,)

        # correction factor — when max updates, all our old exp scores are wrong
        # they were computed relative to m_old, but now the reference is m_new
        # exp(m_old - m_new) rescales them correctly (always <= 1 since m_new >= m_old)
        correction = tl.exp(m_block - m_new)  # (BLOCK_M,)

        # P = exp(S - m_new) — probabilities for this block's scores
        # subtracting m_new before exp prevents overflow
        P = tl.exp(S - m_new[:, None])  # (BLOCK_M, BLOCK_N)

        # update running sum: rescale old sum, add new probabilities
        l_block = correction * l_block + tl.sum(P, axis=1)  # (BLOCK_M,)

        # update running output: rescale old output, add new weighted values
        # O_block is (BLOCK_M, d), correction needs to broadcast: (BLOCK_M, 1)
        O_block = correction[:, None] * O_block + tl.dot(P.to(Q_block.dtype), V_block)  # (BLOCK_M, d)

        # advance the max tracker
        m_block = m_new

    # divide by l to complete the softmax normalization
    # this is the "divide by sum(exp(x))" part that's hidden in the softmax formula
    # we deferred it until the very end to avoid dividing and re-multiplying
    O_block = O_block / l_block[:, None]  # (BLOCK_M, d)

    # write the output block back to HBM
    # mask out rows that go beyond N (same mask we used when loading Q)
    O_block_ptr = O_ptr + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(O_block_ptr, O_block.to(Q_block.dtype), mask=q_mask)


def flash_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    FlashAttention forward pass using the Triton kernel above.

    Computes the same thing as naive_attention — scaled dot-product attention —
    but without ever materializing the full NxN score matrix in HBM.

    Args:
        Q: query tensor, shape (batch, heads, seq_len, head_dim)
        K: key tensor, shape (batch, heads, seq_len, head_dim)
        V: value tensor, shape (batch, heads, seq_len, head_dim)

    Returns:
        output tensor, shape (batch, heads, seq_len, head_dim), same dtype as inputs
    """
    # validate — all three must have the same shape
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape"
    assert Q.dim() == 4, "expected (batch, heads, seq_len, head_dim)"

    batch, heads, N, d = Q.shape
    assert d in (16, 32, 64, 128, 256), f"head_dim must be a power of 2, got {d}"

    # triton needs contiguous tensors — make sure we have that
    Q = Q.contiguous()
    K = K.contiguous()
    V = V.contiguous()

    # allocate output in the same dtype as inputs
    O = torch.empty_like(Q)

    # 1/sqrt(d) scaling factor — compute once on CPU, pass as scalar to kernel
    scale = 1.0 / math.sqrt(d)

    # grid: one program per (Q_block, batch*head) pair
    # axis 0 = which Q block, axis 1 = which (batch, head)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), batch * heads)

    # flatten batch and heads into a single dimension for the kernel
    # the kernel uses strides to navigate the actual layout
    # stride_qh needs to cover one full (seq_len, head_dim) block
    _flash_attn_forward_kernel[grid](
        Q, K, V, O,
        # Q strides
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        # K strides
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        # V strides
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        # O strides
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        N=N,
        d=d,
        scale=scale,
    )

    return O


if __name__ == "__main__":
    # quick smoke test — make sure the kernel at least runs without crashing
    # for real correctness checking see tests/test_correctness.py
    torch.manual_seed(42)

    batch, heads, seq_len, head_dim = 1, 1, 128, 64
    device = "cuda"

    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)

    out = flash_attention(Q, K, V)

    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"Output mean:  {out.float().mean().item():.6f}")
    print(f"Output std:   {out.float().std().item():.6f}")
    print()
    print("kernel ran without errors — run test_correctness.py to check numerical accuracy")
