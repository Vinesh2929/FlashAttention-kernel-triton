"""
benchmark.py

Wall-clock time, memory, and TFLOPS for flash_attention vs naive_attention
vs PyTorch's built-in SDPA (scaled dot-product attention from F.scaled_dot_product_attention).

The memory difference is the whole point of FlashAttention. Naive attention allocates
an O(N^2) score matrix on GPU, so memory grows quadratically with sequence length.
FlashAttention tiles, so memory stays roughly flat — the only allocations that grow
with N are the input/output tensors themselves, which are O(N*d).

How we measure memory: torch.cuda.max_memory_allocated() tracks peak allocation
since the last reset. We reset before each run, then subtract the baseline (input
tensors already allocated) to isolate just what each forward pass needs.

TFLOPS calculation: attention is roughly 4 * N^2 * d FLOPs (two matrix multiplies
of shape N x d and N x N). Dividing by wall-clock time gives TFLOPS achieved.
"""

import argparse
import torch
import torch.nn.functional as F
import triton.testing
import matplotlib
matplotlib.use("Agg")  # no display needed — we just want to save PNGs
import matplotlib.pyplot as plt
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from naive_attention import naive_attention
from flash_attention import flash_attention


# sequence lengths to sweep — 8192 might OOM with naive on small GPUs
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]

# fixed parameters — one head, head_dim=64 is standard
BATCH   = 1
HEADS   = 1
HEAD_DIM = 64
DTYPE   = torch.float16

# how many warmup + benchmark iterations triton.testing.do_bench runs
WARMUP  = 25
REP     = 100


def measure_memory_mb(fn, inputs):
    """
    Run fn(*inputs), return peak memory in MB allocated during the call.
    We reset the peak tracker before the call so we only see what fn uses,
    not whatever was already on GPU.
    """
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()

    fn(*inputs)

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    # subtract the baseline (input tensors themselves)
    return (peak - baseline) / (1024 ** 2)


def flops_attention(N: int, d: int) -> float:
    """
    Approximate FLOPs for one attention forward pass.
    Two matmuls: (N x d) @ (d x N) = 2*N^2*d, and (N x N) @ (N x d) = 2*N^2*d.
    Total ~= 4 * N^2 * d.
    """
    return 4.0 * N * N * d


def run_benchmark(causal: bool = False):
    if not torch.cuda.is_available():
        print("ERROR: need a CUDA GPU to benchmark")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Benchmarking on: {torch.cuda.get_device_name(0)}")
    print(f"dtype={DTYPE}, batch={BATCH}, heads={HEADS}, head_dim={HEAD_DIM}")
    if causal:
        print("mode: causal=True  (Flash Causal column included)")
    print()

    results = []

    for N in SEQ_LENGTHS:
        print(f"seq_len = {N} ...", end=" ", flush=True)

        Q = torch.randn(BATCH, HEADS, N, HEAD_DIM, device=device, dtype=DTYPE)
        K = torch.randn(BATCH, HEADS, N, HEAD_DIM, device=device, dtype=DTYPE)
        V = torch.randn(BATCH, HEADS, N, HEAD_DIM, device=device, dtype=DTYPE)

        # --- naive attention ---
        try:
            naive_time_ms = triton.testing.do_bench(
                lambda: naive_attention(Q, K, V),
                warmup=WARMUP, rep=REP,
            )
            naive_mem_mb = measure_memory_mb(naive_attention, (Q, K, V))
            naive_tflops = flops_attention(N, HEAD_DIM) / naive_time_ms / 1e9
        except torch.cuda.OutOfMemoryError:
            print(f"\n  naive OOM at N={N}, skipping")
            naive_time_ms = float("nan")
            naive_mem_mb  = float("nan")
            naive_tflops  = float("nan")

        # --- flash attention (non-causal) ---
        flash_time_ms = triton.testing.do_bench(
            lambda: flash_attention(Q, K, V),
            warmup=WARMUP, rep=REP,
        )
        flash_mem_mb = measure_memory_mb(flash_attention, (Q, K, V))
        flash_tflops = flops_attention(N, HEAD_DIM) / flash_time_ms / 1e9

        # --- torch SDPA ---
        sdpa_time_ms = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V),
            warmup=WARMUP, rep=REP,
        )
        sdpa_mem_mb = measure_memory_mb(
            lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
            (Q, K, V),
        )
        sdpa_tflops = flops_attention(N, HEAD_DIM) / sdpa_time_ms / 1e9

        row = {
            "N":           N,
            "naive_ms":    naive_time_ms,
            "naive_mb":    naive_mem_mb,
            "naive_tflops": naive_tflops,
            "flash_ms":    flash_time_ms,
            "flash_mb":    flash_mem_mb,
            "flash_tflops": flash_tflops,
            "sdpa_ms":     sdpa_time_ms,
            "sdpa_mb":     sdpa_mem_mb,
            "sdpa_tflops":  sdpa_tflops,
        }

        # --- flash attention (causal) — only when --causal flag is set ---
        # causal skips upper-triangle kv blocks entirely so actual compute is ~half
        # TFLOPS is computed using half the full flop count — fairer comparison
        if causal:
            flash_causal_time_ms = triton.testing.do_bench(
                lambda: flash_attention(Q, K, V, causal=True),
                warmup=WARMUP, rep=REP,
            )
            flash_causal_mem_mb = measure_memory_mb(
                lambda q, k, v: flash_attention(q, k, v, causal=True),
                (Q, K, V),
            )
            flash_causal_tflops = flops_attention(N, HEAD_DIM) / 2 / flash_causal_time_ms / 1e9
            row["flash_causal_ms"]     = flash_causal_time_ms
            row["flash_causal_mb"]     = flash_causal_mem_mb
            row["flash_causal_tflops"] = flash_causal_tflops

        results.append(row)
        print("done")

    return results


def print_table(results, causal: bool = False):
    """Print the benchmark results as a clean markdown table."""
    print()
    print("## Benchmark Results")
    print()

    def fmt_ms(v):
        return f"{v:10.3f}" if not math.isnan(v) else "       OOM"

    def fmt_mb(v):
        return f"{v:10.2f}" if not math.isnan(v) else "       OOM"

    def fmt_tf(v):
        return f"{v:12.3f}" if not math.isnan(v) else "         OOM"

    causal_header = "| Causal (ms) | Causal (MB) | Causal TFLOPS " if causal else ""
    causal_sep    = "-------------|-------------|---------------|" if causal else ""

    header = (
        "| Seq Len "
        "| Naive (ms) | Naive (MB) | Naive TFLOPS "
        "| Flash (ms) | Flash (MB) | Flash TFLOPS "
        + causal_header +
        "| SDPA (ms) | SDPA (MB) | SDPA TFLOPS |"
    )
    sep = (
        "|---------|"
        "------------|------------|--------------|"
        "------------|------------|--------------|"
        + causal_sep +
        "-----------|-----------|-------------|"
    )
    print(header)
    print(sep)

    for r in results:
        causal_cols = ""
        if causal:
            causal_cols = (
                f"| {fmt_ms(r['flash_causal_ms'])}  "
                f"| {fmt_mb(r['flash_causal_mb'])}  "
                f"| {fmt_tf(r['flash_causal_tflops'])}"
            )

        print(
            f"| {r['N']:7d} "
            f"| {fmt_ms(r['naive_ms'])} | {fmt_mb(r['naive_mb'])} | {fmt_tf(r['naive_tflops'])}"
            f"| {fmt_ms(r['flash_ms'])} | {fmt_mb(r['flash_mb'])} | {fmt_tf(r['flash_tflops'])}"
            + causal_cols +
            f"| {fmt_ms(r['sdpa_ms'])} | {fmt_mb(r['sdpa_mb'])} | {fmt_tf(r['sdpa_tflops'])} |"
        )

    print()


def save_plots(results, causal: bool = False, output_dir="."):
    """Generate and save the two benchmark plots."""

    seq_lens     = [r["N"]            for r in results]
    naive_mbs    = [r["naive_mb"]     for r in results]
    flash_mbs    = [r["flash_mb"]     for r in results]
    sdpa_mbs     = [r["sdpa_mb"]      for r in results]
    naive_tflops = [r["naive_tflops"] for r in results]
    flash_tflops = [r["flash_tflops"] for r in results]
    sdpa_tflops  = [r["sdpa_tflops"]  for r in results]

    # --- memory plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(seq_lens, naive_mbs, marker="o", label="Naive PyTorch",       linewidth=2)
    ax.plot(seq_lens, flash_mbs, marker="s", label="FlashAttention",      linewidth=2)
    ax.plot(seq_lens, sdpa_mbs,  marker="^", label="Torch SDPA",          linewidth=2)
    if causal:
        flash_causal_mbs = [r["flash_causal_mb"] for r in results]
        ax.plot(seq_lens, flash_causal_mbs, marker="D", label="FlashAttention (causal)", linewidth=2)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Peak Memory (MB)", fontsize=12)
    ax.set_title("Attention Memory Usage vs Sequence Length", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    # log scale so the quadratic blowup of naive is visually obvious
    ax.set_yscale("log")
    fig.tight_layout()
    mem_path = os.path.join(output_dir, "memory_usage.png")
    fig.savefig(mem_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {mem_path}")

    # --- throughput plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(seq_lens, naive_tflops, marker="o", label="Naive PyTorch",  linewidth=2)
    ax.plot(seq_lens, flash_tflops, marker="s", label="FlashAttention", linewidth=2)
    ax.plot(seq_lens, sdpa_tflops,  marker="^", label="Torch SDPA",     linewidth=2)
    if causal:
        # causal TFLOPS uses half-flops since upper triangle is genuinely skipped
        flash_causal_tflops = [r["flash_causal_tflops"] for r in results]
        ax.plot(seq_lens, flash_causal_tflops, marker="D",
                label="FlashAttention (causal, eff. TFLOPS)", linewidth=2)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("TFLOPS", fontsize=12)
    ax.set_title("Attention Throughput vs Sequence Length", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    thr_path = os.path.join(output_dir, "throughput.png")
    fig.savefig(thr_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {thr_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark FlashAttention vs naive vs SDPA")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="include Flash Causal column — runs flash_attention(causal=True) and adds a 4th line to plots",
    )
    args = parser.parse_args()

    results = run_benchmark(causal=args.causal)
    print_table(results, causal=args.causal)
    save_plots(results, causal=args.causal)
