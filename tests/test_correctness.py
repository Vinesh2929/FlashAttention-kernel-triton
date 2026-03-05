"""
tests/test_correctness.py

Correctness tests — does our Triton kernel actually match vanilla PyTorch attention?

The strategy: run naive_attention (our ground truth reference) and flash_attention
on the same inputs, then check that the outputs are close enough. We use a loose
tolerance (atol=1e-2, rtol=1e-2) because float16 accumulation introduces small
rounding errors, and the online softmax accumulation compounds them slightly.

If these tests pass, the kernel is computing the right thing. Memory savings and
speed come for free from the tiling — correctness is what we actually need to verify.
"""

import sys
import os
import torch

# add parent directory to path so we can import without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from naive_attention import naive_attention
from flash_attention import flash_attention


# test matrix — all combinations of these get run
SEQ_LENGTHS = [128, 256, 512, 1024]
HEAD_DIMS   = [32, 64, 128]
BATCH_SIZES = [1, 2, 4]
DTYPES      = [torch.float16, torch.float32]

# tolerance — float16 accumulation means we can't be too strict here
# the paper uses similar tolerances in their correctness evaluation
ATOL = 1e-2
RTOL = 1e-2


def run_one_test(batch: int, seq_len: int, head_dim: int, dtype: torch.dtype) -> bool:
    """
    Run a single correctness test. Returns True if passed, False if failed.
    Prints results either way.
    """
    heads = 4  # fixed at 4 — testing across more head counts would just be redundant

    device = "cuda"
    torch.manual_seed(0)  # deterministic inputs

    # generate random inputs in the given dtype on GPU
    Q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    K = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    V = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    # run both implementations
    with torch.no_grad():
        # naive reference runs in float32 for precision, then convert to target dtype
        # so we're comparing apples to apples
        if dtype == torch.float16:
            Q32 = Q.float()
            K32 = K.float()
            V32 = V.float()
            ref_out = naive_attention(Q32, K32, V32).half()
        else:
            ref_out = naive_attention(Q, K, V)

        flash_out = flash_attention(Q, K, V)

    # compute the max absolute difference across all elements
    max_diff = (ref_out.float() - flash_out.float()).abs().max().item()

    # check if they're close enough
    try:
        torch.testing.assert_close(
            flash_out.float(),
            ref_out.float(),
            atol=ATOL,
            rtol=RTOL,
        )
        passed = True
    except AssertionError:
        passed = False

    dtype_str = "fp16" if dtype == torch.float16 else "fp32"
    status = "PASSED" if passed else "FAILED"

    print(
        f"  [{status}] "
        f"batch={batch:2d}  seq_len={seq_len:5d}  head_dim={head_dim:4d}  dtype={dtype_str}"
        f"  |  max_diff={max_diff:.2e}"
    )

    return passed


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This kernel requires a GPU.")
        sys.exit(1)

    print("=" * 70)
    print("FlashAttention correctness tests")
    print(f"  comparing flash_attention vs naive_attention")
    print(f"  tolerance: atol={ATOL}, rtol={RTOL}")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)
    print()

    total  = 0
    passed = 0

    # sweep all combinations
    for dtype in DTYPES:
        dtype_str = "float16" if dtype == torch.float16 else "float32"
        print(f"--- dtype = {dtype_str} ---")

        for batch in BATCH_SIZES:
            for seq_len in SEQ_LENGTHS:
                for head_dim in HEAD_DIMS:
                    ok = run_one_test(batch, seq_len, head_dim, dtype)
                    total  += 1
                    passed += int(ok)

        print()

    # summary
    print("=" * 70)
    failed = total - passed
    print(f"Results: {passed}/{total} passed, {failed} failed")

    if failed == 0:
        print("All tests passed.")
    else:
        print(f"{failed} test(s) FAILED — check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
