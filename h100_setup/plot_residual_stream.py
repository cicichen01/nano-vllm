"""Residual stream: ONE tensor, with TWO residual connections (+) per layer (attn + MLP).
Plus nano-vllm's twist: each '+' is fused into the FOLLOWING RMSNorm (add_rms) and deferred."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

CATTN, CMLP, CNORM, RED = "#4C86C6", "#4FA06B", "#E8E8E8", "#C0392B"
fig, ax = plt.subplots(figsize=(16.5, 8.8))
ax.set_xlim(0, 31); ax.set_ylim(0, 12.8); ax.axis("off")

def box(x, y, w, h, t, fc, ec="#555", fs=10, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                lw=1.3, edgecolor=ec, facecolor=fc))
    ax.text(x+w/2, y+h/2, t, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold" if bold else "normal")

def arr(x1, y1, x2, y2, c="#555", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=lw, color=c))

STREAM_Y = 9.6
# the residual stream (one thick arrow)
ax.add_patch(FancyArrowPatch((0.2, STREAM_Y), (30.2, STREAM_Y), arrowstyle="-|>",
                             mutation_scale=20, lw=3.2, color="#222"))
ax.text(0.2, 11.9, "residual stream  —  ONE tensor, threaded through layers as  `residual`",
        fontsize=12.5, fontweight="bold", color="#111")

# 4 sub-blocks: (label, color, plus-label)
subs = [("Attention", CATTN, "+ attn skip"), ("MLP", CMLP, "+ MLP skip"),
        ("Attention", CATTN, "+ attn skip"), ("MLP", CMLP, "+ MLP skip")]
bases = [1, 8, 15, 22]
addxs = []
for (name, c, plab), base in zip(subs, bases):
    nx, sx, addx = base+1.7, base+4.5, base+6.6
    addxs.append(addx)
    # tap from stream down into norm (pre-norm: normalize a COPY, stream stays un-normed)
    arr(nx, STREAM_Y-0.35, nx, 7.35, c="#888")
    box(nx-1.05, 6.0, 2.1, 1.35, "RMSNorm", CNORM, fs=9)
    box(sx-1.25, 6.0, 2.6, 1.35, name, c, fs=9.5, tc="white", bold=True)
    arr(nx+1.05, 6.67, sx-1.3, 6.67)                       # norm -> sublayer
    arr(sx+1.35, 6.9, addx-0.28, STREAM_Y-0.28, c=c)        # sublayer output -> the '+'
    # the '+' node on the stream
    ax.add_patch(Circle((addx, STREAM_Y), 0.33, facecolor="white", edgecolor="#222", lw=1.8, zorder=5))
    ax.text(addx, STREAM_Y, "+", ha="center", va="center", fontsize=15, fontweight="bold", zorder=6)
    ax.text(addx+0.55, STREAM_Y-0.02, plab, ha="left", va="center", fontsize=8.3, color=c, fontweight="bold")

# layer brackets
def bracket(x0, x1, y, label):
    ax.plot([x0, x0, x1, x1], [y+0.25, y, y, y+0.25], color="#333", lw=1.4)
    ax.text((x0+x1)/2, y-0.5, label, ha="center", va="center", fontsize=11, fontweight="bold", color="#333")
bracket(0.6, 14.6, 5.0, "Layer 0   (2 residual connections)")
bracket(14.9, 28.9, 5.0, "Layer 1   (2 residual connections)")

# fusion callout: the attn '+' is computed INSIDE the next (MLP) RMSNorm
ax.add_patch(FancyBboxPatch((6.9, 5.6), 4.3, 4.55, boxstyle="round,pad=0.05,rounding_size=0.1",
                            lw=1.6, edgecolor=RED, facecolor="none", linestyle="--"))
ax.text(9.05, 10.75, "fused (add_rms): '+' runs inside the next RMSNorm", ha="center", va="center",
        fontsize=8.8, color=RED, fontweight="bold")

# bottom explainer
ax.add_patch(FancyBboxPatch((0.6, 1.1), 29.8, 3.0, boxstyle="round,pad=0.05,rounding_size=0.15",
                            lw=1.1, edgecolor="#999", facecolor="#F6F6F6"))
ax.text(1.1, 3.5, "One stream, TWO residual connections per layer (around attention, around MLP) — both add into the SAME running tensor.",
        fontsize=10.3, color="#111", va="center", fontweight="bold")
ax.text(1.1, 2.5, "nano-vllm twist: each '+' is DEFERRED and fused into the FOLLOWING RMSNorm — `add_rms_forward` = (residual += sublayer_out) then normalize, in ONE kernel.",
        fontsize=9.8, color="#333", va="center")
ax.text(1.1, 1.7, "So `residual` is passed in/out of each layer to carry the stream across the deferred boundary; the MLP add of layer i is completed by layer i+1's input_layernorm.",
        fontsize=9.8, color="#333", va="center")

plt.tight_layout()
out = "/home/cicichen/nano-vllm/h100_setup/residual_stream.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
