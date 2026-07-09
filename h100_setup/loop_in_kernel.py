"""The KV-block loop (Q·Kᵀ per block) lives INSIDE one kernel. The graph captures the KERNEL LAUNCH,
not the loop iterations. Capture with count=0 (loop runs 0x → output 0); replay with real count →
the SAME launched kernel loops the real number of times → real result. One launch node, runtime loop."""
import torch, triton
import triton.language as tl
dev = "cuda"

@triton.jit
def loopsum(out_ptr, in_ptr, n_ptr):
    n = tl.load(n_ptr)              # loop count = a VALUE read on-device (like context_lens)
    acc = 0.0
    i = 0
    while i < n:                    # data-dependent loop INSIDE the kernel (like looping KV blocks)
        acc += tl.load(in_ptr + i) # the per-iteration work (analog of Q·Kᵀ → softmax → ·V)
        i += 1
    tl.store(out_ptr, acc)

inp = torch.arange(1, 9, dtype=torch.float32, device=dev)   # [1..8]
n   = torch.zeros(1, dtype=torch.int32, device=dev)
out = torch.empty(1, dtype=torch.float32, device=dev)
def f():
    loopsum[(1,)](out, inp, n)
    return out

# capture with count = 0  (loop body never runs during capture)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): f()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): f()
torch.cuda.synchronize()
print("capture (n=0): out =", out.item(), " (loop ran 0x → no 'Q·Kᵀ' done at capture)")

n.fill_(5); g.replay(); torch.cuda.synchronize()
print("replay  (n=5): out =", out.item(), " expected", inp[:5].sum().item(), " (loop ran 5x now)")
n.fill_(8); g.replay(); torch.cuda.synchronize()
print("replay  (n=8): out =", out.item(), " expected", inp.sum().item(), " (loop ran 8x now)")
print("→ ONE kernel launch was captured; the LOOP COUNT is a runtime value read on-device.")
print("  The loop body (the 'Q·Kᵀ' work) is compiled INTO the kernel — it runs at REPLAY, not capture.")
