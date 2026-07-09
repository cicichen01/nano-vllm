"""Value-driven reshape/transpose done ENTIRELY on the GPU, inside a CUDA graph.
The flag is read on-device (tl.load), the layout branch is device-side (tl.where) — no .item(),
no host sync in the decision. Replaying with a different flag gives a different layout, from ONE
captured graph. But the output buffer stays FIXED shape [8] — the value picks CONTENT, not a new host shape."""
import torch, triton
import triton.language as tl
dev = "cuda"

@triton.jit
def reshuffle(out_ptr, in_ptr, flag_ptr, N: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    flag = tl.load(flag_ptr)                 # read flag ON THE GPU (no host, no sync)
    offs = tl.arange(0, N)
    pr = offs // R; pc = offs % R
    src_t = pc * C + pr                       # transpose [R,C] -> [C,R] index mapping
    src = tl.where(flag != 0, offs, src_t)    # DEVICE-side branch: identity vs transpose
    tl.store(out_ptr + offs, tl.load(in_ptr + src))

inp  = torch.arange(8., device=dev)          # [0..7] = matrix [2,4]
out  = torch.empty(8, device=dev)
flag = torch.ones(1, dtype=torch.int32, device=dev)

def f():
    reshuffle[(1,)](out, inp, flag, N=8, R=2, C=4)
    return out

# warmup + capture (flag=1 at capture)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): f()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): f()

# change layout by changing ONLY the GPU flag — no .item() in the decision
flag.fill_(1); g.replay(); torch.cuda.synchronize()
print("flag=1  → identity layout :", out.tolist())
flag.fill_(0); g.replay(); torch.cuda.synchronize()
print("flag=0  → transpose layout:", out.tolist())
print("expected transpose of [2,4]=[[0,1,2,3],[4,5,6,7]] → [0,4,1,5,2,6,3,7]")
