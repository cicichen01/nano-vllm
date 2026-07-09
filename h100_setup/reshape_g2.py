"""Isolated: does flag = tl.load(flag_ptr) let a kernel produce a value-dependent SHAPE?
(no prior failed capture to corrupt state)"""
import torch, triton
import triton.language as tl
dev = "cuda"

@triton.jit
def fill(out_ptr, flag_ptr, N: tl.constexpr):
    flag = tl.load(flag_ptr)                          # on-device, NO sync
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, offs.to(tl.float32) + flag * 100.0)   # CONTENT depends on flag

flag = torch.ones(1, dtype=torch.int32, device=dev)
out  = torch.empty(2, 4, device=dev)                 # host allocates the shape BEFORE the kernel
def fG2():
    fill[(1,)](out, flag, N=8)
    return out

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): fG2()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): o = fG2()
print("capture with tl.load(flag): SUCCEEDED  (no .item(), no sync)")
flag.fill_(1); g.replay(); torch.cuda.synchronize()
print("  flag=1 → shape", tuple(o.shape), " content", o.flatten().tolist())
flag.fill_(0); g.replay(); torch.cuda.synchronize()
print("  flag=0 → shape", tuple(o.shape), " content", o.flatten().tolist())
print("→ capture SUCCEEDS, but shape (2,4) BOTH times: tl.load changed CONTENT, not SHAPE.")
print("  The output was host-allocated torch.empty(2,4) BEFORE the kernel — the kernel can't pick (2,4) vs (4,2).")
