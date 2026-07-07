"""Pre-norm (Qwen3/Llama) vs Post-norm (original Transformer): WHERE the norm sits.
Pre-norm: norm on the side branch → residual stream is a CLEAN identity line.
Post-norm: norm ON the residual path (after each +) → stream is interrupted by norms."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

CATTN, CMLP, CNORM, CNORM2, RED, GREEN = "#4C86C6", "#4FA06B", "#E8E8E8", "#E8A33D", "#C0392B", "#2E7D32"

def box(ax, x, y, w, h, t, fc, ec="#555", fs=10, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                lw=1.4, edgecolor=ec, facecolor=fc, zorder=4))
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal", zorder=5)

def arr(ax, x1, y1, x2, y2, c="#555", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=lw, color=c))

def plus(ax, x, y):
    ax.add_patch(Circle((x, y), 0.33, facecolor="white", edgecolor="#222", lw=1.8, zorder=6))
    ax.text(x, y, "+", ha="center", va="center", fontsize=15, fontweight="bold", zorder=7)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
for ax in (ax1, ax2):
    ax.set_xlim(0, 23); ax.set_ylim(0, 9); ax.axis("off")

# ===================== PRE-NORM =====================
SY = 6.4
ax1.text(0.3, 8.4, "PRE-NORM  (Qwen3 / Llama / GPT-NeoX):   x = x + Attn(Norm(x));   x = x + MLP(Norm(x))",
         fontsize=12.5, fontweight="bold", color="#111")
# clean identity stream
ax1.add_patch(FancyArrowPatch((0.3, SY), (21.5, SY), arrowstyle="-|>", mutation_scale=18, lw=3.0, color=GREEN, zorder=2))
ax1.text(21.7, SY, "→ next\n   layer", fontsize=8.5, va="center", color=GREEN)
for name, c, base in [("Attention", CATTN, 1.0), ("MLP", CMLP, 11.5)]:
    nx, sx, addx = base+1.8, base+4.6, base+7.3
    arr(ax1, nx, SY-0.35, nx, 4.35, c="#888")                 # tap raw copy down to norm
    box(ax1, nx-1.0, 3.0, 2.0, 1.3, "RMSNorm", CNORM, fs=9)
    box(ax1, sx-1.3, 3.0, 2.6, 1.3, name, c, fs=9.5, tc="white", bold=True)
    arr(ax1, nx+1.0, 3.65, sx-1.35, 3.65)                     # norm -> sublayer
    arr(ax1, sx+1.35, 3.9, addx-0.28, SY-0.28, c=c)           # sublayer out -> +
    plus(ax1, addx, SY)
ax1.text(0.3, 1.4, "Norm is on the SIDE BRANCH (a normalized COPY feeds each sub-block).  The residual stream (green line) has ONLY the '+' on it",
         fontsize=10, color=GREEN, va="center", fontweight="bold")
ax1.text(0.3, 0.6, "→ a CLEAN identity path: gradients flow straight through → stable to hundreds of layers.  (This is what enables the fused add+norm.)",
         fontsize=9.6, color="#333", va="center")

# ===================== POST-NORM =====================
ax2.text(0.3, 8.4, "POST-NORM  (original 2017 Transformer):   x = Norm(x + Attn(x));   x = Norm(x + MLP(x))",
         fontsize=12.5, fontweight="bold", color="#111")
# stream drawn as background line; norm boxes sit ON it (interrupting it)
ax2.add_patch(FancyArrowPatch((0.3, SY), (21.5, SY), arrowstyle="-|>", mutation_scale=18, lw=3.0, color="#B00020", zorder=2))
ax2.text(21.7, SY, "→ next\n   layer", fontsize=8.5, va="center", color="#B00020")
for name, c, base in [("Attention", CATTN, 1.0), ("MLP", CMLP, 11.5)]:
    sx, addx, normx = base+3.3, base+6.0, base+8.4
    arr(ax2, base+1.2, SY-0.35, sx-1.3, 3.9, c="#888")         # tap RAW x down to sublayer (NO pre-norm)
    box(ax2, sx-1.3, 3.0, 2.6, 1.3, name, c, fs=9.5, tc="white", bold=True)
    arr(ax2, sx+1.35, 3.9, addx-0.28, SY-0.28, c=c)           # sublayer out -> +
    plus(ax2, addx, SY)
    box(ax2, normx-1.15, SY-0.6, 2.3, 1.2, "RMSNorm", CNORM2, ec="#B00020", fs=9, bold=True)  # ON the stream
ax2.text(0.3, 1.4, "Norm sits ON the residual path, AFTER each '+' (the stream passes THROUGH each RMSNorm).  Sub-blocks read the RAW (un-normed) x",
         fontsize=10, color="#B00020", va="center", fontweight="bold")
ax2.text(0.3, 0.6, "→ the identity path is INTERRUPTED by a norm every sub-block → deep nets harder to train (needs warmup); no clean deferred-add fusion.",
         fontsize=9.6, color="#333", va="center")

fig.suptitle("Pre-norm vs Post-norm — where the normalization sits (a MODEL-ARCHITECTURE choice)",
             fontsize=13.5, fontweight="bold", y=0.99)
plt.tight_layout(rect=[0, 0, 1, 0.97])
out = "/home/cicichen/nano-vllm/h100_setup/prenorm_vs_postnorm.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
