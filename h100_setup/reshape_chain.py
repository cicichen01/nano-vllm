"""Two chained kernels in one CUDA graph. kernel1's LAYOUT depends on a GPU flag (content varies),
but its output is a FIXED [8] buffer. kernel2 consumes that [8] buffer → its shape is UNCHANGED
regardless of the flag. So chaining is fine: the value changed CONTENT, not the downstream shape."""
import torch, triton
import triton.language as tl
dev = "cuda"

@triton.jit
def reshuffle(out_ptr, in_ptr, flag_ptr, N: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    flag = tl.load(flag_ptr)
    offs = tl.arange(0, N)
    pr = offs // R; pc = offs % R
    src = tl.where(flag != 0, offs, pc * C + pr)   # identity vs transpose (device-side)
    tl.store(out_ptr + offs, tl.load(in_ptr + src))

@triton.jit
def times10(out_ptr, in_ptr, N: tl.constexpr):     # kernel2: always sees N=8, fixed grid
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.load(in_ptr + offs) * 10.0)

inp  = torch.arange(8., device=dev)
mid  = torch.empty(8, device=dev)     # FIXED-shape intermediate
out  = torch.empty(8, device=dev)
flag = torch.ones(1, dtype=torch.int32, device=dev)

def f():
    reshuffle[(1,)](mid, inp, flag, N=8, R=2, C=4)   # kernel1: content depends on flag
    times10[(1,)](out, mid, N=8)                       # kernel2: consumes fixed [8]
    return out

s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): f()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): f()

flag.fill_(1); g.replay(); torch.cuda.synchronize()
print("flag=1 : mid =", mid.tolist(), " out(=mid*10) =", out.tolist())
flag.fill_(0); g.replay(); torch.cuda.synchronize()
print("flag=0 : mid =", mid.tolist(), " out(=mid*10) =", out.tolist())
print("\nkernel2 saw shape [8] in BOTH cases — only the CONTENT of `mid` changed, not its shape.")
