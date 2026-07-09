"""Can `flag = tl.load(flag_ptr)` (on-device) make the OUTPUT SHAPE value-dependent? No.
A Triton kernel fills a buffer the HOST already allocated with a fixed shape. The on-device flag
changes CONTENT, but the output shape is whatever the host passed in — the kernel can't change it."""
import torch, triton
import triton.language as tl
dev = "cuda"

@triton.jit
def reshuffle(out_ptr, in_ptr, flag_ptr, N: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    flag = tl.load(flag_ptr)                          # flag read ON-DEVICE (no .item, no sync)
    offs = tl.arange(0, N)
    pr = offs // R; pc = offs % R
    src = tl.where(flag != 0, offs, pc * C + pr)      # identity vs transpose CONTENT
    tl.store(out_ptr + offs, tl.load(in_ptr + src))

inp  = torch.arange(8., device=dev)
flag = torch.ones(1, dtype=torch.int32, device=dev)

# The HOST decides the output shape by allocation. The kernel just fills it.
out = torch.empty(2, 4, device=dev)                  # host chose shape (2,4)
flag.fill_(1); reshuffle[(1,)](out, inp, flag, N=8, R=2, C=4); torch.cuda.synchronize()
print("flag=1 (on-device): out.shape =", tuple(out.shape), " content =", out.flatten().tolist())
flag.fill_(0); reshuffle[(1,)](out, inp, flag, N=8, R=2, C=4); torch.cuda.synchronize()
print("flag=0 (on-device): out.shape =", tuple(out.shape), " content =", out.flatten().tolist())
print("→ output.shape is (2,4) BOTH times. The on-device flag changed CONTENT, not SHAPE.")
print()

# To get a DIFFERENT shape you must allocate it on the HOST — the shape comes from torch.empty, not the kernel:
outA = torch.empty(2, 4, device=dev)
outB = torch.empty(4, 2, device=dev)
print("host torch.empty(2,4).shape =", tuple(outA.shape), " ; torch.empty(4,2).shape =", tuple(outB.shape))
print("→ choosing (2,4) vs (4,2) is a HOST decision; to base it on `flag` the host must read flag → .item() → sync.")
