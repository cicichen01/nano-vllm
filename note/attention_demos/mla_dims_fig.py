import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT="/home/cicichen/nano-vllm/h100_setup"
GREY="#7f8c8d"; GREEN="#3a9d5d"; BLUE="#2c6fbb"; RED="#c0392b"; ORANGE="#e08b25"; DARK="#2c3e50"

d_model, n_heads, head_dim, d_c = 4096, 32, 128, 512
S = 2.0/d_model                       # scale: 4096 -> 2.0 plot units

def mat(ax, x, ybase, cols, rows, name, dims, color, note=None):
    """draw weight matrix; width∝cols(in), height∝rows(out), bottom-aligned at ybase."""
    w, h = cols*S, rows*S
    ax.add_patch(Rectangle((x, ybase), w, h, fc=color, ec="#222", lw=1.3))
    ax.text(x+w/2, ybase+h+0.12, name, ha="center", va="bottom", fontsize=9.5, weight="bold")
    ax.text(x+w/2, ybase+h/2, dims, ha="center", va="center", fontsize=7.5,
            color="white", rotation=0 if w>0.5 else 90)
    if note:
        ax.text(x+w/2, ybase-0.18, note, ha="center", va="top", fontsize=7, color="#444")
    return x+w

fig, ax = plt.subplots(figsize=(15, 9.5)); ax.axis("off")
ax.set_xlim(0, 15); ax.set_ylim(0, 10)

# ---------------- Band 1: MHA stored weights ----------------
y1=7.2
ax.text(0.2, y1+1.0, "MHA\nstored weights", fontsize=11, weight="bold", color=DARK, va="center")
x=2.6
for nm in ["W_Q","W_K","W_V","W_O"]:
    c = RED if nm in ("W_K","W_V") else GREY
    x = mat(ax, x, y1, d_model, d_model, nm, "4096×4096", c) + 0.55
ax.text(12.0, y1+1.0, "KV cache / token\n= K+V = 2×4096\n= 8192 numbers",
        fontsize=8.5, color=RED, ha="center", va="center",
        bbox=dict(boxstyle="round", fc="#fdecea", ec=RED))

# ---------------- Band 2: MLA stored weights ----------------
y2=3.9
ax.text(0.2, y2+1.0, "MLA\nstored weights", fontsize=11, weight="bold", color=DARK, va="center")
x=2.6
x = mat(ax, x, y2, d_model, d_model, "W_Q", "4096×4096", GREY) + 0.5
x = mat(ax, x, y2, d_model, d_c, "W_DKV", "512×4096", GREEN,
        note="SHARED (common\nfactor, cached)") + 0.5
x = mat(ax, x, y2, d_c, d_model, "W_UK", "4096×512", BLUE, note="per-head") + 0.4
x = mat(ax, x, y2, d_c, d_model, "W_UV", "4096×512", BLUE, note="per-head") + 0.6
x = mat(ax, x, y2, d_model, d_model, "W_O", "4096×4096", GREY) + 0.5
ax.text(12.9, y2+1.0, "KV cache / token\n= c = 512\n(+64 rope)\n≈16× smaller",
        fontsize=8.5, color=GREEN, ha="center", va="center",
        bbox=dict(boxstyle="round", fc="#eafaf1", ec=GREEN))
# factorization annotation
ax.text(7.5, y2-0.75, "factorization:  W_K(4096×4096)  =  W_UK(4096×512) · W_DKV(512×4096)   "
        "→ cache the shared middle  c = W_DKV·h", fontsize=8.5, color=DARK, ha="center")

# ---------------- Band 3: equivalent (absorbed) compute ----------------
y3=0.7
ax.text(0.2, y3+1.0, "MLA equivalent\ncomputation\n(absorbed)", fontsize=11, weight="bold",
        color=DARK, va="center")
x=2.6
x = mat(ax, x, y3, d_model, d_c, "W_Q' = W_UKᵀW_Q", "512×4096", ORANGE,
        note="×n_heads (rank≤128)") + 0.7
x = mat(ax, x, y3, d_model, d_c, "W_DKV  (=K/V)", "512×4096", GREEN,
        note="shared latent c") + 0.7
x = mat(ax, x, y3, d_model, d_model, "W_O' = W_O·W_UV", "4096×(n_h·512)", ORANGE,
        note="absorbs W_UV") + 0.5
ax.text(12.5, y3+1.0, "dot-products &\nvalue-mix happen\nin d_c=512 latent,\neach head via its\nOWN absorbed query",
        fontsize=8, color=ORANGE, ha="center", va="center",
        bbox=dict(boxstyle="round", fc="#fef5e7", ec=ORANGE))

ax.set_title("Weight matrix dimensions: MHA vs MLA (stored) and MLA's equivalent absorbed compute\n"
             "d_model=4096, n_heads=32, head_dim=128, d_c=512   (all-head/stacked shapes; box size ∝ dims)",
             fontsize=12)
fig.tight_layout()
fig.savefig(f"{OUT}/mla_weight_dims.png", dpi=130); print("saved")
