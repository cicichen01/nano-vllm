"""Does `flag != 0` itself sync, or only the `if`? And does bool(tensor) sync? Verify directly."""
import torch
dev = "cuda"
flag = torch.ones(1, dtype=torch.int32, device=dev)
def cap(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    return g, out

print("V2 — the COMPARISON only (no Python if):  cond = (flag != 0)  [pure GPU op]")
def v2():
    cond = (flag != 0)              # elementwise on GPU — no host readback
    return cond.to(torch.int32)
try:
    g,o = cap(v2); print("   capture SUCCEEDED → `flag != 0` alone does NOT sync (it's a GPU op)")
except Exception as e:
    print("   capture FAILED:", type(e).__name__, "-", str(e).splitlines()[0][:150])
