"""Test the USER's exact suggestion: pick the shape with `if flag != 0` (no explicit .item()).
Each variant in its OWN try; a failed capture is isolated so it can't contaminate the next."""
import torch
dev = "cuda"

def cap(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    return g, out

flag = torch.ones(1, dtype=torch.int32, device=dev)

print("V1 — your suggestion:  return ones(2,4) if flag != 0 else ones(4,2)")
def v1(): return torch.ones(2,4,device=dev) if (flag != 0) else torch.ones(4,2,device=dev)
try:
    g,o = cap(v1); print("   capture SUCCEEDED, shape", tuple(o.shape))
except Exception as e:
    print("   capture FAILED:", type(e).__name__, "-", str(e).splitlines()[0][:150])
