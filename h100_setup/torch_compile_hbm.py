"""Isolate fusion's HBM benefit vs CUDA-graph's launch-only benefit.
Chain of N pointwise ops, run 3 ways: eager | graph-only (capture eager) | fused (torch.compile).
Two sizes: SMALL (launch-bound) and LARGE (memory-bound)."""
import torch, time, json
from torch.profiler import profile, ProfilerActivity
dev="cuda"; N=20
def chain(x):
    for _ in range(N): x = torch.relu(x)*1.001 + 0.001   # N fuseable pointwise ops
    return x

def cuda_time(fn,it=50,warm=20):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e)/it

def kcount(fn,path,it=5):
    for _ in range(3): fn()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
        for _ in range(it): fn()
        torch.cuda.synchronize()
    p.export_chrome_trace(path)
    ev=json.load(open(path))["traceEvents"]
    return sum(1 for e in ev if e.get("cat")=="kernel")/it

def run_size(S, tag):
    x=torch.randn(S,device=dev)
    # eager
    eager=lambda: chain(x)
    # graph-only (capture the eager chain)
    st=torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): chain(x)
    torch.cuda.current_stream().wait_stream(st)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=chain(x)
    graph=lambda: g.replay()
    # fused
    cchain=torch.compile(chain); fused=lambda: cchain(x)
    bytes_chain = 2*S*4*N    # unfused HBM: each op reads+writes S floats, x N
    print(f"\n[{tag}]  S={S} elems  (unfused HBM≈{bytes_chain/1e6:.0f} MB, fused≈{2*S*4/1e6:.1f} MB)")
    print(f"   {'mode':<14}{'ms/iter':>9}{'GPU kernels':>13}")
    for name,fn in [("eager",eager),("graph-only",graph),("fused",fused)]:
        ms=cuda_time(fn); k=kcount(fn,f"/tmp/hbm_{tag}_{name}.json")
        print(f"   {name:<14}{ms:>9.4f}{k:>13.0f}")

run_size(1<<12,  "SMALL (launch-bound)")   # 4K elems -> tiny -> launch-bound
run_size(1<<24,  "LARGE (memory-bound)")   # 16M elems -> 64MB -> memory-bound
