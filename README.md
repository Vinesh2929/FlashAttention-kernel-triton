# FlashAttention in Triton

A from-scratch implementation of the FlashAttention forward pass kernel, written in
[Triton](https://github.com/openai/triton). Built as a learning project — every line
is commented to explain the *why*, not just the *what*.

---

## What is FlashAttention?

Standard attention computes `softmax(QK^T / sqrt(d)) * V`. The problem is that
`QK^T` is an `N x N` matrix — for a sequence of length 4096 in float16, that's
**32MB per head**. At 32 heads over a batch, that's gigabytes of intermediate data
being written to GPU HBM (the main GPU memory), then read back. The math is not
expensive. The memory traffic is what kills you.

FlashAttention's insight is that you don't need to materialize the full `N x N`
matrix at all. You can tile it — process `Q` in blocks of `BLOCK_M` rows, and
for each block iterate over all `K/V` in chunks of `BLOCK_N`. Everything fits in
SRAM (the GPU's small on-chip cache), and you never write the score matrix to HBM.
Memory usage goes from `O(N^2)` to `O(N)`.

The catch: computing softmax requires seeing all the scores in a row before you can
normalize. FlashAttention solves this with **online softmax** — you maintain three
running statistics as you scan through K/V blocks: `m` (running max), `l` (running
sum of exp scores), and `O` (running weighted sum of V). When you find a new max,
you apply a correction factor `exp(m_old - m_new)` to rescale what you already
accumulated. At the end, dividing `O` by `l` gives you the exact same answer as
the standard formula. The correctness tests verify this.

---

## Key insight: tiling + online softmax

The two ideas work together:
- **Tiling** keeps data in SRAM and avoids the `N^2` HBM traffic
- **Online softmax** lets us produce the correct output incrementally without ever
  seeing all scores at once

The correction factor is the non-obvious part. When the running max `m` increases
from `m_old` to `m_new`, all the `exp(score - m_old)` values we computed earlier
are wrong — they should have been `exp(score - m_new)`. Multiplying by
`exp(m_old - m_new)` fixes them. Since `m_new >= m_old`, this factor is always
`<= 1`, which means we're always scaling down, never up. No overflow.

---

## Repo structure

```
flashattention-triton/
├── naive_attention.py        # vanilla PyTorch attention — correctness reference
├── flash_attention.py        # Triton kernel implementation
├── benchmark.py              # benchmarks naive vs flash vs torch SDPA
├── tests/
│   └── test_correctness.py   # numerical correctness tests
└── README.md
```

---

## Setup

```bash
pip install torch triton matplotlib
```

Requires Python 3.10+, a CUDA GPU, and Triton >= 2.1. Tested on Google Colab T4.

---

## Running the correctness tests

```bash
python tests/test_correctness.py
```

Tests all combinations of:
- sequence lengths: 128, 256, 512, 1024
- head dims: 32, 64, 128
- batch sizes: 1, 2, 4
- dtypes: float16, float32

Expected output (all 96 tests):
```
======================================================================
FlashAttention correctness tests
  comparing flash_attention vs naive_attention
  tolerance: atol=0.01, rtol=0.01
  device: Tesla T4
======================================================================

--- dtype = float16 ---
  [PASSED] batch= 1  seq_len=  128  head_dim=  32  dtype=fp16  |  max_diff=1.22e-03
  [PASSED] batch= 1  seq_len=  128  head_dim=  64  dtype=fp16  |  max_diff=9.77e-04
  ...
Results: 96/96 passed, 0 failed
All tests passed.
```

---

## Running the benchmark

```bash
python benchmark.py
```

This sweeps sequence lengths [512, 1024, 2048, 4096, 8192] and measures wall-clock
time, peak memory, and TFLOPS for all three implementations. Saves two plots:
`memory_usage.png` and `throughput.png`.

---

## Benchmark results

Results from a Google Colab T4 GPU (16GB HBM), float16, batch=1, heads=1, head_dim=64.

| Seq Len | Naive (ms) | Naive (MB) | Naive (TFLOPS) | Flash (ms) | Flash (MB) | Flash (TFLOPS) | SDPA (ms)  | SDPA (MB)  | SDPA (TFLOPS) |
|---------|------------|------------|----------------|------------|------------|----------------|------------|------------|---------------|
|     512 |      0.082 |       0.50 |          0.013 |      0.143 |       0.06 |          0.008 |      0.045 |       0.50 |          0.024 |
|    1024 |      0.134 |       2.00 |          0.032 |      0.271 |       0.06 |          0.016 |      0.065 |       2.00 |          0.066 |
|    2048 |      0.380 |       8.00 |          0.045 |      0.744 |       0.06 |          0.023 |      0.146 |       8.00 |          0.118 |
|    4096 |      1.257 |      32.00 |          0.054 |      2.453 |       0.06 |          0.028 |      0.455 |      32.00 |          0.150 |
|    8192 |      4.871 |     128.00 |          0.056 |      9.140 |       0.06 |          0.030 |      1.621 |     128.00 |          0.168 |

The memory story is exactly what FlashAttention promises — naive attention's memory
grows quadratically (0.5 MB at N=512 → 128 MB at N=8192, a 256x increase), while
flash attention stays nearly flat at ~0.06 MB across all sequence lengths.

On raw speed, torch's SDPA wins — it uses optimized fused kernels (and on Ampere+
GPUs uses FlashAttention-2 internally). Our Triton kernel is a clean educational
implementation; production-grade speed would need more tuning (better vectorization,
smarter prefetching, more autotuning configs).

---

## Plots

### Memory Usage vs Sequence Length
![memory usage](memory_usage.png)

The naive curve grows quadratically — that's the `N^2` score matrix.
FlashAttention stays flat because we never allocate it.

### Throughput (TFLOPS) vs Sequence Length
![throughput](throughput.png)

Torch SDPA is fastest because it uses highly optimized fused CUDA kernels.
Our kernel shows the same scaling behavior, just with lower absolute throughput
due to being a teaching implementation rather than a production one.
