"""Log-scale: gradient magnitude vs depth, pre-norm vs post-norm.
Left: controlled toy prod(I+J) vs prod 0.9(I+S) — post VANISHES.
Right: real L-layer nets, |grad@input| for a unit output gradient, gain=3."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn

GREEN, RED = "#2E7D32", "#B00020"

# ---- toy: explicit Jacobian products ----
torch.manual_seed(1); d2 = 32
Ls = [2, 5, 10, 20, 40, 80, 160]
pre_toy, post_toy = [], []
Ppre = torch.eye(d2); Ppost = torch.eye(d2); done = 0
for L in Ls:
    for _ in range(L - done):
        J = torch.randn(d2, d2) * (0.3 / d2**0.5)
        Ppre  = (torch.eye(d2) + J) @ Ppre
        Ppost = (0.9 * (torch.eye(d2) + J)) @ Ppost
    done = L
    pre_toy.append(Ppre.norm().item()); post_toy.append(Ppost.norm().item())

# ---- real nets ----
class RMSNorm(nn.Module):
    def __init__(s, d): super().__init__(); s.w = nn.Parameter(torch.ones(d)); s.eps = 1e-6
    def forward(s, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * s.w
def run(mode, L, d, gain, seed=0):
    torch.manual_seed(seed)
    lins = [nn.Linear(d, d, bias=False) for _ in range(L)]
    for lin in lins: lin.weight.data *= gain
    norms = [RMSNorm(d) for _ in range(L)]
    x0 = torch.randn(1, d, requires_grad=True); x = x0
    for i in range(L):
        x = x + lins[i](norms[i](x)) if mode == "pre" else norms[i](x + lins[i](x))
    v = torch.randn(1, d); v = v / v.norm(); x.backward(v)
    return x0.grad.norm().item()
Ls2 = [2, 5, 10, 20, 40, 64, 96, 128]
pre_r = [run("pre", L, 128, 3.0) for L in Ls2]
post_r = [run("post", L, 128, 3.0) for L in Ls2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.semilogy(Ls, pre_toy, "o-", color=GREEN, lw=2, label=r"PRE-norm  $\|\prod(I+J)\|$")
ax1.semilogy(Ls, post_toy, "s-", color=RED, lw=2, label=r"POST-norm  $\|\prod 0.9(I+S)\|$")
ax1.axhline(1.0, color="#888", ls=":", lw=1)
ax1.set_title("Controlled toy: isolates 'bare I' vs '∏N'\n(N=0.9·I models a contracting norm)", fontsize=12, fontweight="bold")
ax1.set_xlabel("depth  L  (number of layers)"); ax1.set_ylabel("gradient magnitude  (log scale)")
ax1.legend(fontsize=11); ax1.grid(alpha=0.3, which="both")
ax1.annotate("post-norm VANISHES\n0.9^160 ≈ 5e-8", xy=(160, post_toy[-1]), xytext=(70, 1e-3),
             fontsize=10, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
ax1.annotate("pre-norm grows,\nnever vanishes", xy=(160, pre_toy[-1]), xytext=(60, 1e3),
             fontsize=10, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))

ax2.semilogy(Ls2, pre_r, "o-", color=GREEN, lw=2, label="PRE-norm  |grad@input|")
ax2.semilogy(Ls2, post_r, "s-", color=RED, lw=2, label="POST-norm  |grad@input|")
ax2.axhline(1.0, color="#888", ls=":", lw=1)
ax2.set_title("Real L-layer nets (d=128, sublayer gain=3):\nunit gradient at output → |grad| at input", fontsize=12, fontweight="bold")
ax2.set_xlabel("depth  L  (number of layers)"); ax2.set_ylabel("|grad @ input|  (log scale)")
ax2.legend(fontsize=11); ax2.grid(alpha=0.3, which="both")
ax2.annotate("pre grows with depth\n(identity highway accumulates)", xy=(128, pre_r[-1]), xytext=(20, 40),
             fontsize=9.5, color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN))
ax2.annotate("post throttled ~O(1) / below\n(re-norm every layer)", xy=(128, post_r[-1]), xytext=(30, 0.12),
             fontsize=9.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED))

fig.suptitle("Gradient vs depth: PRE-norm keeps a bare-I highway (∏(I+J)) — POST-norm compounds norm Jacobians (∏N(I+S))",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = "/home/cicichen/nano-vllm/h100_setup/grad_decay.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
