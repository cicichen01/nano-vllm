"""
Show compute-bound vs memory-bound in two ways:
  (1) ROOFLINE self-check: measure achieved TFLOP/s and GB/s; whichever is near
      its hardware peak is the bottleneck.
  (2) torch.profiler trace: per-kernel CUDA time + a Chrome trace you can open in
      chrome://tracing or https://ui.perfetto.dev

Run:
  ENV=/home/cicichen/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib $ENV/bin/python h100_setup/profile_bottleneck.py
"""
import torch
from torch.profiler import profile, ProfilerActivity

dev = "cuda"
# H100 SXM nominal peaks (bf16 dense tensor cores, HBM3)
PEAK_TFLOPS = 989.0      # bf16, no sparsity
PEAK_GBs    = 3350.0     # ~3.35 TB/s

def bench(fn, flops, bytes_, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters): fn()
    end.record(); torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    tflops = flops / (ms*1e-3) / 1e12
    gbs    = bytes_ / (ms*1e-3) / 1e9
    return ms, tflops, gbs

def classify(tflops, gbs):
    comp = 100*tflops/PEAK_TFLOPS
    mem  = 100*gbs/PEAK_GBs
    bound = "COMPUTE-bound" if comp > mem else "MEMORY-bound"
    return comp, mem, bound

print(f"H100 nominal peaks: {PEAK_TFLOPS:.0f} TFLOP/s (bf16),  {PEAK_GBs:.0f} GB/s\n")
print(f"{'workload':<34}{'ms':>8}{'TFLOP/s':>10}{'GB/s':>9}{'%compute':>10}{'%mem':>8}  verdict")
print("-"*100)

# ---------- (A) COMPUTE-bound: big square GEMM (prefill-like) ----------
N = 8192
A = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
Bm = torch.randn(N, N, device=dev, dtype=torch.bfloat16)
flops = 2*N*N*N
bytes_ = 3*N*N*2                      # read A,B write C (bf16)
ms, tf, gb = bench(lambda: A @ Bm, flops, bytes_)
c, m, v = classify(tf, gb)
print(f"{'A) GEMM 8192³ (prefill-like)':<34}{ms:>8.3f}{tf:>10.1f}{gb:>9.0f}{c:>9.1f}%{m:>7.1f}%  {v}")

# ---------- (B) MEMORY-bound: large elementwise add ----------
sz = 1 << 26                          # 67M elements
x = torch.randn(sz, device=dev, dtype=torch.float32)
y = torch.randn(sz, device=dev, dtype=torch.float32)
flops = sz                            # one add per element
bytes_ = 3*sz*4                       # read x,y write z (fp32)
ms, tf, gb = bench(lambda: x + y, flops, bytes_)
c, m, v = classify(tf, gb)
print(f"{'B) elementwise add 64M (copy-like)':<34}{ms:>8.3f}{tf:>10.2f}{gb:>9.0f}{c:>9.2f}%{m:>7.1f}%  {v}")

# ---------- (C) MEMORY-bound: decode-like GEMV (1 token vs big weight) ----------
H = 8192
q = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
W = torch.randn(H, 4*H, device=dev, dtype=torch.bfloat16)
flops = 2*1*H*(4*H)
bytes_ = H*(4*H)*2                    # reading the weight dominates
ms, tf, gb = bench(lambda: q @ W, flops, bytes_)
c, m, v = classify(tf, gb)
print(f"{'C) GEMV [1×H]@[H×4H] (decode-like)':<34}{ms:>8.3f}{tf:>10.2f}{gb:>9.0f}{c:>9.2f}%{m:>7.1f}%  {v}")

# ---------- (D) batched GEMV becomes more compute-bound as batch grows ----------
for Bsz in [1, 8, 64, 256]:
    qb = torch.randn(Bsz, H, device=dev, dtype=torch.bfloat16)
    flops = 2*Bsz*H*(4*H)
    bytes_ = H*(4*H)*2 + Bsz*H*2      # weight (shared) + activations
    ms, tf, gb = bench(lambda: qb @ W, flops, bytes_)
    c, m, v = classify(tf, gb)
    print(f"{f'D) GEMM [B={Bsz:>3}×H]@[H×4H]':<34}{ms:>8.3f}{tf:>10.1f}{gb:>9.0f}{c:>9.1f}%{m:>7.1f}%  {v}")

# ---------- torch.profiler trace (compute-bound vs memory-bound kernels) ----------
print("\n=== torch.profiler: per-kernel CUDA time (compute GEMM vs memory add) ===")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
    for _ in range(20):
        _ = A @ Bm        # compute-bound
        _ = x + y         # memory-bound
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
trace = "/home/cicichen/nano-vllm/h100_setup/bottleneck_trace.json"
prof.export_chrome_trace(trace)
print(f"\nChrome trace written: {trace}")
print("Open in chrome://tracing or https://ui.perfetto.dev to see the kernel timeline.")
