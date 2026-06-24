"""
Launch-bound demo: many TINY kernels per step.
- EAGER: every kernel needs a CPU launch (aten:: op). Kernels finish faster than
  the CPU can launch the next -> GPU sits IDLE between them (gaps) -> launch-bound.
- CUDA GRAPH: capture the whole sequence once, replay as ONE launch -> no per-kernel
  CPU dispatch -> GPU kernels run back-to-back -> big speedup.

Run:
  ENV=/home/cicichen/.conda/envs/nanovllm
  LD_LIBRARY_PATH=$ENV/targets/x86_64-linux/lib:$ENV/lib $ENV/bin/python h100_setup/profile_cudagraph.py
"""
import torch, time
from torch.profiler import profile, ProfilerActivity

dev = "cuda"
N_OPS = 50       # tiny kernels per "step" (like the many small ops in a decode step)
STEPS = 300
x = torch.zeros(256, device=dev)    # TINY tensor -> GPU work per op is negligible

def run_ops(t):
    # a chain of tiny, data-dependent in-place ops (so they serialize, like a layer stack)
    for _ in range(N_OPS):
        t.add_(1.0)
        t.mul_(0.9999)
    return t

def timeit(fn, iters):
    torch.cuda.synchronize(); s = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/iters*1e3   # ms/step

# ---- EAGER ----
for _ in range(5): run_ops(x)          # warmup
eager_ms = timeit(lambda: run_ops(x), STEPS)

# ---- CUDA GRAPH capture (must run on a side stream + use STATIC tensors) ----
g = torch.cuda.CUDAGraph()
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3): run_ops(x)       # warmup the capture
torch.cuda.current_stream().wait_stream(side)
with torch.cuda.graph(g):
    run_ops(x)                          # capture ONE step's worth of ops
for _ in range(5): g.replay()           # warmup
graph_ms = timeit(lambda: g.replay(), STEPS)

kernels_per_step = N_OPS*2
print(f"kernels per step      : {kernels_per_step}  (tiny ops on a 256-elem tensor)")
print(f"EAGER   : {eager_ms:.3f} ms/step   ({eager_ms*1e3/kernels_per_step:.1f} us per kernel-launch)")
print(f"GRAPH   : {graph_ms:.3f} ms/step   (1 replay launch for all {kernels_per_step} kernels)")
print(f"SPEEDUP : {eager_ms/graph_ms:.1f}x   <- this gap is the CPU LAUNCH OVERHEAD eager pays, graph removes")

# ---- profiler traces ----
def prof_to(fn, iters, path, tag):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(iters): fn()
        torch.cuda.synchronize()
    p.export_chrome_trace(path)
    print(f"\n[{tag}] top ops by CUDA time:")
    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=4))
    print(f"[{tag}] trace -> {path}")

prof_to(lambda: run_ops(x), 20, "/home/cicichen/nano-vllm/h100_setup/trace_eager.json", "EAGER")
prof_to(lambda: g.replay(), 20, "/home/cicichen/nano-vllm/h100_setup/trace_graph.json", "GRAPH")
print("\nOpen the two traces in https://ui.perfetto.dev :")
print("  EAGER -> many aten:: launches + GPU GAPS between tiny kernels (launch-bound)")
print("  GRAPH -> one cudaGraphLaunch + kernels BACK-TO-BACK, no gaps")
