"""Tiny target for Nsight Compute: one compute-bound GEMM + one memory-bound add."""
import torch
dev = "cuda"
N = 2048
A = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
B = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
sz = 1 << 24
x = torch.randn(sz, device=dev); y = torch.randn(sz, device=dev)
for _ in range(3):           # warmup so cuBLAS picks its kernel
    _ = A @ B; _ = x + y
torch.cuda.synchronize()
_ = A @ B                    # compute-bound kernel
_ = x + y                    # memory-bound kernel
torch.cuda.synchronize()
