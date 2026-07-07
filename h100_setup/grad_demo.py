import torch, torch.nn as nn
torch.manual_seed(0)
d = 128

class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = 1e-6
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.w

def run(mode, L, d, seed=0):
    torch.manual_seed(seed)                                  # SAME weights for pre & post -> fair
    lins  = [nn.Linear(d, d, bias=False) for _ in range(L)]
    norms = [RMSNorm(d) for _ in range(L)]
    x0 = torch.randn(1, d, requires_grad=True)
    x = x0
    for i in range(L):
        x = x + lins[i](norms[i](x)) if mode == "pre" else norms[i](x + lins[i](x))
    v = torch.randn(1, d); v = v / v.norm()                  # unit upstream gradient at the OUTPUT
    x.backward(v)                                            # seed dL/dx_L = v  -> measures ||(dx_L/dx0)^T v||
    return x0.grad.norm().item()

print(f"{'depth L':>8} | {'PRE-norm  |grad@input|':>24} | {'POST-norm |grad@input|':>24} | ratio pre/post")
print("-"*84)
for L in [2, 5, 10, 20, 40, 80]:
    gp = run("pre", L, d); gq = run("post", L, d)
    print(f"{L:>8} | {gp:>24.4e} | {gq:>24.4e} | {gp/max(gq,1e-30):>12.2e}")
