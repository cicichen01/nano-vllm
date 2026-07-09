"""Is it 'pre-allocated buffer' or 'fixed shape' that makes it graphable?
F : FRESH torch.ones(2,4) (constant shape) inside capture           -> works (graph pool).
G1: shape CHOSEN by flag.item() (ones(2,4) vs ones(4,2))            -> sync -> capture FAILS.
G2: same idea but flag read ON-DEVICE (tl.load) in a Triton kernel  -> captures, BUT shape is FIXED
     (the kernel fills a host-allocated buffer; it cannot pick (2,4) vs (4,2))."""
import torch, triton
import triton.language as tl
dev = "cuda"

def cap(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    return g, out

print("CASE F — FRESH torch.ones(2,4) (constant shape) * a GPU value")
scale = torch.ones(1, device=dev)
def fF():
    y = torch.ones(2, 4, device=dev)      # fresh, CONSTANT shape
    return y * scale
g, out = cap(fF); scale.fill_(5.0); g.replay(); torch.cuda.synchronize()
print("  replay scale=5 → shape", tuple(out.shape), "  ✅ works (fresh alloc, fixed shape)")

print("\nCASE G1 — shape chosen by flag.item() (host readback)")
flag = torch.ones(1, dtype=torch.int32, device=dev)
def fG1():
    return torch.ones(2, 4, device=dev) if flag.item() else torch.ones(4, 2, device=dev)
try:
    g, out = cap(fG1); print("  capture SUCCEEDED (unexpected):", tuple(out.shape))
except Exception as e:
    print("  capture FAILED:", type(e).__name__, "-", str(e)[:110])

print("\nCASE G2 — flag read ON-DEVICE (tl.load), trying to make the shape depend on it")
@triton.jit
def fill(out_ptr, flag_ptr, N: tl.constexpr):
    flag = tl.load(flag_ptr)                       # on-device, no sync
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, offs.to(tl.float32) + flag * 100.0)   # CONTENT depends on flag

out = torch.empty(2, 4, device=dev)               # host MUST allocate a shape BEFORE the kernel
def fG2():
    fill[(1,)](out, flag, N=8)
    return out
g, o = cap(fG2)                                    # captures fine — no .item(), no sync
flag.fill_(1); g.replay(); torch.cuda.synchronize()
print("  flag=1 → shape", tuple(o.shape), " content", o.flatten().tolist())
flag.fill_(0); g.replay(); torch.cuda.synchronize()
print("  flag=0 → shape", tuple(o.shape), " content", o.flatten().tolist())
print("  → capture SUCCEEDS, but shape is (2,4) BOTH times. tl.load changed CONTENT, not SHAPE.")
print("    There is no way to write fG2 so it returns (2,4) OR (4,2) from the on-device flag —")
print("    the output must be host-allocated with a shape BEFORE the kernel runs.")
