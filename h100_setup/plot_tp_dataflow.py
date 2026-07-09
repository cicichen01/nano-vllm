"""TP data-flow through ONE transformer layer (tp=2): where activations are
REPLICATED (full, identical on both ranks) vs SHARDED (per-rank slice), and where
the all_reduce collectives merge partials. Norms/act/residual = replicated (cheap,
recomputed on both, no comm); the big matmuls are split."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FULL, S0, S1, RED = "#4C86C6", "#4FA06B", "#E0803B", "#C0392B"
C0, C1 = 39.5, 64.5           # rank-0 / rank-1 lane centers
BW = 19                        # box width

fig, ax = plt.subplots(figsize=(13.5, 16))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

# lanes
for c in (C0, C1):
    ax.plot([c, c], [4, 92], color="#DDD", lw=1.2, zorder=0)
ax.text(C0, 95, "RANK 0", ha="center", fontsize=12, fontweight="bold")
ax.text(C1, 95, "RANK 1", ha="center", fontsize=12, fontweight="bold")

def box(cx, y, text, fc, tc="white"):
    ax.add_patch(FancyBboxPatch((cx-BW/2, y-1.9), BW, 3.8, boxstyle="round,pad=0.1,rounding_size=0.4",
                                lw=1.2, edgecolor="#555", facecolor=fc, zorder=3))
    ax.text(cx, y, text, ha="center", va="center", fontsize=8.6, color=tc, fontweight="bold", zorder=4)

def stage(y, label, kind, note):
    ax.text(1, y, label, ha="left", va="center", fontsize=9, fontweight="bold", color="#111")
    if kind == "full":
        box(C0, y, "FULL  [T, H]", FULL); box(C1, y, "FULL  [T, H]", FULL)
    elif kind == "shard":
        box(C0, y, "rank0 half", S0); box(C1, y, "rank1 half", S1)
    elif kind == "partial":
        box(C0, y, "partial₀", S0); box(C1, y, "partial₁", S1)
    elif kind == "reduce":
        ax.add_patch(FancyBboxPatch((C0-BW/2, y-1.9), (C1-C0)+BW, 3.8,
                     boxstyle="round,pad=0.1,rounding_size=0.4", lw=1.6, edgecolor=RED, facecolor="#FBEAEA", zorder=3))
        ax.text((C0+C1)/2, y, "↔  all_reduce (sum)  ↔", ha="center", va="center",
                fontsize=10, color=RED, fontweight="bold", zorder=4)
    ax.text(84, y, note, ha="left", va="center", fontsize=8, color="#555")

stages = [
    (90.0, "hidden (input)",            "full",    "replicated: identical on both ranks"),
    (83.0, "input_layernorm [repl]",    "full",    "both compute norm locally · NO comm"),
    (76.0, "qkv_proj  [COLUMN]",        "shard",   "input FULL → output = this rank's HEADS"),
    (69.0, "attention (per-head)",      "shard",   "each rank attends ITS heads · no comm"),
    (62.0, "o_proj  [ROW]",             "partial", "sharded input → PARTIAL sum"),
    (55.0, "all_reduce (sum)",          "reduce",  "combine partials → FULL on both"),
    (48.0, "+ residual  → hidden",  "full",    "replicated again"),
    (41.0, "post_attn_LN [repl]",       "full",    "both compute locally · NO comm"),
    (34.0, "gate_up_proj  [COLUMN]",    "shard",   "output sharded (intermediate dim)"),
    (27.0, "SiluAndMul (per-rank)",     "shard",   "elementwise, sharded · no comm"),
    (20.0, "down_proj  [ROW]",          "partial", "sharded input → PARTIAL sum"),
    (13.0, "all_reduce (sum)",          "reduce",  "combine partials → FULL on both"),
    (6.0,  "hidden → next layer",   "full",    "replicated"),
]
for y, lab, kind, note in stages:
    stage(y, lab, kind, note)

# down arrows between consecutive stages (per lane)
ys = [s[0] for s in stages]
for a, b in zip(ys[:-1], ys[1:]):
    for c in (C0, C1):
        ax.add_patch(FancyArrowPatch((c, a-2.0), (c, b+2.0), arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#999", zorder=1))

# block markers on the right
ax.text(99, 72, "ATTENTION block", ha="center", va="center", rotation=90, fontsize=11, color="#4C86C6", fontweight="bold")
ax.text(99, 27, "MLP block", ha="center", va="center", rotation=90, fontsize=11, color="#4FA06B", fontweight="bold")

# legend
ax.add_patch(FancyBboxPatch((1, 97.5), 60, 2.2, boxstyle="round,pad=0.1", lw=0, facecolor="white"))
lx = 1
for c, t in [(FULL, "REPLICATED (full, identical)"), (S0, "sharded rank0"), (S1, "sharded rank1"), ("#FBEAEA", "all_reduce (collective)")]:
    ax.add_patch(FancyBboxPatch((lx, 98.0), 1.6, 1.4, boxstyle="round,pad=0.03", lw=0.8, edgecolor="#888", facecolor=c))
    ax.text(lx+2.0, 98.7, t, ha="left", va="center", fontsize=8.2, color="#333"); lx += len(t)*0.62 + 6

ax.set_title("Tensor-parallel data-flow (tp=2) through one transformer layer:  replicate the cheap ops, shard the big matmuls, reconcile with all_reduce",
             fontsize=11.5, fontweight="bold", pad=16)
plt.tight_layout()
out = "/home/cicichen/nano-vllm/h100_setup/tp_dataflow.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
