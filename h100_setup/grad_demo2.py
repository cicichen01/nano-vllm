import torch, torch.nn as nn
d = 128

class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = 1e-6
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.w

def run(mode, L, d, gain, seed=0):
    torch.manual_seed(seed)
    lins  = [nn.Linear(d, d, bias=False) for _ in range(L)]
    for lin in lins: lin.weight.data *= gain          # scale sublayer contribution
    norms = [RMSNorm(d) for _ in range(L)]
    x0 = torch.randn(1, d, requires_grad=True); x = x0
    for i in range(L):
        x = x + lins[i](norms[i](x)) if mode == "pre" else norms[i](x + lins[i](x))
    v = torch.randn(1, d); v = v / v.norm()
    x.backward(v)
    return x0.grad.norm().item()

print("=== REAL networks: |grad @ input| for a UNIT gradient at the output (L=64) ===")
print(f"{'sublayer gain':>14} | {'PRE-norm':>14} | {'POST-norm':>14} | ratio pre/post")
print("-"*66)
for gain in [1.0, 2.0, 3.0, 4.0]:
    gp = run("pre", 64, d, gain); gq = run("post", 64, d, gain)
    print(f"{gain:>14.1f} | {gp:>14.3e} | {gq:>14.3e} | {gp/max(gq,1e-30):>10.2e}")

print("\n=== Controlled toy: the two Jacobian PRODUCTS with the SAME small S and a mild")
print("    contracting norm N=0.9*I (models rms>1) — isolates 'bare I' vs 'prod N' ===")
torch.manual_seed(1)
d2 = 32
print(f"{'depth L':>8} | {'pre:  ||prod(I+J)||':>20} | {'post: ||prod 0.9(I+S)||':>24}")
print("-"*60)
for L in [10, 20, 40, 80, 160]:
    Ppre = torch.eye(d2); Ppost = torch.eye(d2)
    for _ in range(L):
        J = torch.randn(d2, d2) * (0.3 / d2**0.5)     # small sublayer Jacobian
        Ppre  = (torch.eye(d2) + J) @ Ppre            # pre-norm factor:  (I + J)
        Ppost = (0.9 * (torch.eye(d2) + J)) @ Ppost   # post-norm factor: N(I+S), N=0.9 I
    print(f"{L:>8} | {Ppre.norm().item():>20.3e} | {Ppost.norm().item():>24.3e}")
