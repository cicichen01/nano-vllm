"""Why pre-norm's gradient is stable: the backprop through L layers.
Both are PRODUCTS of L per-layer Jacobians; the difference is each factor's shape:
  pre-norm  (I + J_i)      -> product keeps a BARE identity  -> clean gradient highway
  post-norm N_i (I + S_i)  -> product = ∏N_i (no bare I)      -> L norm-Jacobians compound
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

GREEN, RED, GREY = "#2E7D32", "#B00020", "#F4F4F4"
fig, ax = plt.subplots(figsize=(16, 11.5))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def txt(x, y, s, fs=12, color="black", bold=False, ha="left"):
    ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center", fontweight="bold" if bold else "normal")

def card(x, y, w, h, fc="white", ec="#999", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                lw=1.4, edgecolor=ec, facecolor=fc, linestyle=ls))

# ---- title ----
txt(50, 97.5, "Backprop through L layers:  why pre-norm is stable at depth", fs=15, bold=True, ha="center")

# ---- definitions box ----
card(2, 76.5, 96, 17.5, fc=GREY, ec="#888")
txt(4, 91.5, "Symbols", fs=12.5, bold=True)
txt(4, 88.0, r"$r_i$ = residual stream entering sub-block $i$        $I$ = identity matrix  =  the raw skip connection ('$+\,r$')", fs=11.5)
txt(4, 84.3, r"$\mathrm{Norm}$ = RMSNorm/LayerNorm;   $N_i = \partial\,\mathrm{Norm}/\partial(\cdot)$  = its Jacobian  —  NOT the identity (rescales; rank-deficient)", fs=11.5)
txt(4, 80.6, r"$J_i = \partial\,[\,\mathrm{Sub}(\mathrm{Norm}(r_i))\,]\,/\,\partial r_i$  = PRE-norm sublayer-path Jacobian (includes the norm; usually small)", fs=11.5)
txt(4, 77.0, r"$S_i = \partial\,\mathrm{Sub}(r_i)\,/\,\partial r_i$  = POST-norm sublayer Jacobian          $L$ = number of layers (depth)", fs=11.5)

# ---- column headers ----
txt(26, 72.5, "PRE-NORM   (Qwen3 / Llama)", fs=13.5, color=GREEN, bold=True, ha="center")
txt(74, 72.5, "POST-NORM   (original Transformer)", fs=13.5, color=RED, bold=True, ha="center")
ax.plot([50, 50], [12, 71], color="#CCC", lw=1.2)

# ================= LEFT: PRE-NORM =================
LX = 4
txt(LX, 68.0, "1.  Recurrence:", fs=11.5, bold=True)
txt(LX+2, 64.6, r"$r_{i+1} \;=\; r_i \;+\; \mathrm{Sub}(\mathrm{Norm}(r_i))$", fs=14)
txt(LX, 59.5, "2.  Per-layer Jacobian:", fs=11.5, bold=True)
txt(LX+2, 56.0, r"$\dfrac{\partial r_{i+1}}{\partial r_i} \;=\; \mathbf{I} \,+\, J_i$".replace("dfrac","frac"), fs=15, color=GREEN)
txt(LX+2, 52.0, "identity  PLUS  a small term", fs=10.5, color=GREEN)
txt(LX, 47.5, "3.  Chain rule over L layers  (a PRODUCT):", fs=11.5, bold=True)
txt(LX+2, 43.6, r"$\dfrac{\partial r_L}{\partial r_0} \;=\; \prod_{i=0}^{L-1}\, (\,\mathbf{I} + J_i\,)$".replace("dfrac","frac"), fs=15, color=GREEN)
txt(LX, 38.0, "4.  Expand the product:", fs=11.5, bold=True)
txt(LX+2, 34.3, r"$=\; \mathbf{I} \;+\; \sum_i J_i \;+\; \sum_{i<j} J_j J_i \;+\; \cdots$", fs=13.5)
txt(LX+2, 30.6, r"$\Rightarrow$ a BARE $\mathbf{I}$ term: a route through ZERO norms", fs=10.8, color=GREEN, bold=True)
txt(LX, 26.0, r"5.  If $J_i$ small: $\;\approx\; \mathbf{I} + \sum_i J_i\;$  (stays near identity)", fs=11.5)
card(LX-1, 13.5, 44, 9.5, fc="#EAF3EA", ec=GREEN)
txt(LX+1, 20.0, "clean gradient HIGHWAY", fs=12, color=GREEN, bold=True)
txt(LX+1, 16.3, "the bare I is never multiplied by a norm Jacobian", fs=10.3, color="#333")

# ================= RIGHT: POST-NORM =================
RX = 52
txt(RX, 68.0, "1.  Recurrence:", fs=11.5, bold=True)
txt(RX+2, 64.6, r"$r_{i+1} \;=\; \mathrm{Norm}(\,r_i \,+\, \mathrm{Sub}(r_i)\,)$", fs=14)
txt(RX, 59.5, "2.  Per-layer Jacobian:", fs=11.5, bold=True)
txt(RX+2, 56.0, r"$\dfrac{\partial r_{i+1}}{\partial r_i} \;=\; N_i\,(\,\mathbf{I} + S_i\,)$".replace("dfrac","frac"), fs=15, color=RED)
txt(RX+2, 52.0, "norm Jacobian MULTIPLIES everything", fs=10.5, color=RED)
txt(RX, 47.5, "3.  Chain rule over L layers:", fs=11.5, bold=True)
txt(RX+2, 43.6, r"$\dfrac{\partial r_L}{\partial r_0} \;=\; \prod_{i=0}^{L-1}\, N_i\,(\,\mathbf{I} + S_i\,)$".replace("dfrac","frac"), fs=15, color=RED)
txt(RX, 38.0, "4.  Smallest term (take I from each factor):", fs=11.5, bold=True)
txt(RX+2, 34.3, r"$=\; N_{L-1} N_{L-2} \cdots N_0 \;=\; \prod_i N_i$", fs=13.5)
txt(RX+2, 30.6, r"$\Rightarrow$ NO bare $\mathbf{I}$: a product of $L$ norm Jacobians", fs=10.8, color=RED, bold=True)
txt(RX, 26.0, r"5.  $\prod_i N_i$ compounds multiplicatively $\Rightarrow$ vanish / explode", fs=11.5)
card(RX-1, 13.5, 45, 9.5, fc="#FBEAEA", ec=RED)
txt(RX+1, 20.0, "gradients cross L norm Jacobians", fs=12, color=RED, bold=True)
txt(RX+1, 16.3, "no bare I  =>  needs LR warmup + careful init", fs=10.3, color="#333")

# ---- bottom takeaway ----
card(2, 1.5, 96, 9.5, fc="#FFFDF0", ec="#C9A227")
txt(4, 8.4, "Both are PRODUCTS of L factors (chain rule — each $r_{i+1}$ depends on $r_i$).  The difference is each factor's SHAPE:", fs=11.3, bold=True)
txt(4, 4.6, r"pre-norm $(\mathbf{I}+J_i)$ = 'identity + small' -> product keeps a bare $\mathbf{I}$ (clean path).   The residual '$+r$' is what puts that additive $\mathbf{I}$ inside every factor;", fs=10.8)
txt(4, 1.9, r"post-norm $N_i(\mathbf{I}+S_i)$ = norm prefixes every factor -> smallest term is $\prod N_i$, no bare $\mathbf{I}$ -> the L norm-Jacobians compound.", fs=10.8)

plt.tight_layout()
out = "/home/cicichen/nano-vllm/h100_setup/grad_prenorm_postnorm.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
