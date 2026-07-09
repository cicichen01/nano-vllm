"""Does capturing flash_attn_with_kvcache with context_lens=0 / block_table=0 miss graph parts?
Capture with ZEROS, replay with REAL KV, compare to an eager call with the same real inputs."""
import torch
from flash_attn import flash_attn_with_kvcache
dev = "cuda"; torch.manual_seed(0)
bs, nh, nkv, hd = 2, 8, 8, 64
psz, nblocks, maxb = 256, 16, 4
kc = torch.randn(nblocks, psz, nkv, hd, dtype=torch.bfloat16, device=dev)
vc = torch.randn(nblocks, psz, nkv, hd, dtype=torch.bfloat16, device=dev)
q  = torch.randn(bs, 1, nh, hd, dtype=torch.bfloat16, device=dev)
scale = hd ** -0.5
block_table   = torch.zeros(bs, maxb, dtype=torch.int32, device=dev)   # ZERO at capture
cache_seqlens = torch.zeros(bs, dtype=torch.int32, device=dev)          # ZERO at capture

def f():
    return flash_attn_with_kvcache(q, kc, vc, cache_seqlens=cache_seqlens,
                                   block_table=block_table, softmax_scale=scale, causal=True)

# capture with zeros (no KV read during capture)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3): f()
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): out = f()
print("captured with zeros: out during capture (garbage) abs-sum =", out.float().abs().sum().item())

# set REAL block table + lengths, replay
real_bt = torch.tensor([[0,1,2,3],[4,5,6,7]], dtype=torch.int32, device=dev)
real_sl = torch.tensor([300, 500], dtype=torch.int32, device=dev)      # < 4*256
block_table.copy_(real_bt); cache_seqlens.copy_(real_sl)
g.replay(); torch.cuda.synchronize()

# eager reference with the SAME real inputs
ref = flash_attn_with_kvcache(q, kc, vc, cache_seqlens=real_sl, block_table=real_bt,
                              softmax_scale=scale, causal=True)
same = torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
print("graph-replay (captured on zeros) vs eager (real): allclose =", same,
      " max|diff| =", (out.float()-ref.float()).abs().max().item())
print("→ if True: capturing with zeros recorded the FULL attention launch; nothing missed.")
