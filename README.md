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

Runs 192 tests total — 96 non-causal (flash vs naive) and 96 causal
(flash vs torch SDPA). Prints `PASSED` / `FAILED` and max absolute diff for each.

---

## Running the benchmark

```bash
# baseline: naive vs flash vs torch SDPA
python benchmark.py

# with causal masking column (adds Flash Causal results + 4th plot line)
python benchmark.py --causal
```

Sweeps sequence lengths [512, 1024, 2048, 4096, 8192], measures wall-clock time,
peak memory, and TFLOPS. Saves `memory_usage.png` and `throughput.png`.

---

## Benchmark results

> Run `python benchmark.py` on a CUDA GPU and paste results here.

---

## Plots

> Run `python benchmark.py` to generate `memory_usage.png` and `throughput.png`,
> then add them here.
