import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = "/home/cicichen/nano-vllm/h100_setup"

BLUE = "#2c6fbb"
GREEN = "#3a9d5d"
RED = "#c0392b"
ORANGE = "#e08b25"
GREY = "#888888"
DARK = "#222222"

def box(ax, x, y, w, h, text, fc, ec="#333333", fs=10, tc="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                       fc=fc, ec=ec, lw=1.3, mutation_aspect=1)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)

def arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw)
    ax.add_patch(a)

# ----------------------------------------------------------------------------
# FIG 1: KV cache growth — MHA vs MLA vs GDN  (concrete numbers)
# ----------------------------------------------------------------------------
# Concrete toy model: d_model=4096, n_heads=32, head_dim=128, n_layers=60
# MHA cache per token per layer = 2 * n_heads * head_dim = 2*32*128 = 8192 floats
# MLA cache per token per layer = latent dim c ~ 512 floats (+small rope) -> ~576
# GDN: no per-token cache; fixed state = head_dim*head_dim per head (constant in N)
seq = np.arange(0, 32769, 256)
n_layers = 60
mha_per_tok = 2*32*128            # 8192
mla_per_tok = 512 + 64            # 576  (compressed latent + decoupled rope key)
mha_MB = seq * n_layers * mha_per_tok * 2 / 1e6   # bf16 = 2 bytes
mla_MB = seq * n_layers * mla_per_tok * 2 / 1e6
# GDN fixed state: per layer n_heads*head_dim*head_dim ; constant regardless of seq
gdn_state_MB = np.full_like(seq, n_layers*32*128*128*2/1e6, dtype=float)

fig, ax = plt.subplots(figsize=(8.4, 5.2))
ax.plot(seq, mha_MB, color=RED, lw=2.4, label="MHA (full KV cache)  8192 floats/tok/layer")
ax.plot(seq, mla_MB, color=BLUE, lw=2.4, label="MLA (latent KV cache)  ~576 floats/tok/layer")
ax.plot(seq, gdn_state_MB, color=GREEN, lw=2.4, ls="--",
        label="GDN (fixed recurrent state) — flat, no per-token cache")
ax.set_xlabel("sequence length  N  (tokens)")
ax.set_ylabel("memory for attention state  (MB, bf16)")
ax.set_title("KV-cache / state memory vs context length\n"
             "toy 32-layer-ish model: d=4096, 32 heads, head_dim=128, 60 layers",
             fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.grid(alpha=0.3)
ax.annotate("MHA grows LINEARLY\n→ dominates long-context serving",
            xy=(32768, mha_MB[-1]), xytext=(17000, mha_MB[-1]*0.9),
            fontsize=8.5, color=RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("MLA: same shape, ~14x smaller slope",
            xy=(32768, mla_MB[-1]), xytext=(20000, mla_MB[-1]+900),
            fontsize=8.5, color=BLUE, ha="center",
            arrowprops=dict(arrowstyle="->", color=BLUE))
ax.annotate("GDN: FLAT (O(1) in N)",
            xy=(24000, gdn_state_MB[0]), xytext=(20000, gdn_state_MB[0]+1400),
            fontsize=8.5, color=GREEN, ha="center",
            arrowprops=dict(arrowstyle="->", color=GREEN))
fig.tight_layout()
fig.savefig(f"{OUT}/attn_kvcache_growth.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# FIG 2: MLA — low-rank compression schematic
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

# --- left: standard MHA cache ---
ax = axes[0]
ax.set_title("Standard MHA:  cache full K and V for every head", fontsize=11)
ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
box(ax, 0.3, 8.2, 2.2, 1.1, "token\nhidden h_t\n(4096)", GREY, fs=9)
# per-head K,V stacks
for i,(lab,col,yoff) in enumerate([("K heads",BLUE,5.6),("V heads",GREEN,3.0)]):
    for hcol in range(6):
        box(ax, 3.4+hcol*0.75, yoff, 0.6, 1.9, "", col, fs=7, ec="#333")
    ax.text(3.4+6*0.75/2*1.0+1.5, yoff+2.2, f"{lab}: 32 x 128 = 4096 floats",
            ha="center", fontsize=8.5, color=col)
arrow(ax, 2.5, 8.7, 3.4, 6.6)
arrow(ax, 2.5, 8.4, 3.4, 4.0)
box(ax, 3.2, 0.6, 5.4, 1.3, "CACHED PER TOKEN = 8192 floats/layer",
    RED, fs=10)
ax.text(6.0, 2.15, "→ cache size grows with n_heads", ha="center", fontsize=8.5, color=RED)

# --- right: MLA cache ---
ax = axes[1]
ax.set_title("MLA:  cache ONE small latent, reconstruct K/V on the fly", fontsize=11)
ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
box(ax, 0.3, 8.2, 2.2, 1.1, "token\nhidden h_t\n(4096)", GREY, fs=9)
box(ax, 3.6, 8.1, 2.9, 1.3, "down-proj  W_DKV\n(4096 → 512)", ORANGE, fs=9)
arrow(ax, 2.5, 8.75, 3.6, 8.75)
box(ax, 4.0, 5.7, 2.1, 1.4, "latent c_t\n512 floats", BLUE, fs=10)
arrow(ax, 5.0, 8.1, 5.0, 7.1)
box(ax, 3.4, 0.6, 5.4, 1.3, "CACHED PER TOKEN = ~512 floats/layer",
    GREEN, fs=10)
arrow(ax, 5.0, 5.7, 5.0, 1.9, color=GREEN)
# reconstruct arrows
box(ax, 7.0, 6.4, 2.6, 1.0, "up-proj W_UK → K", BLUE, fs=8.5)
box(ax, 7.0, 4.9, 2.6, 1.0, "up-proj W_UV → V", GREEN, fs=8.5)
arrow(ax, 6.1, 6.4, 7.0, 6.9, color=BLUE)
arrow(ax, 6.1, 6.0, 7.0, 5.4, color=GREEN)
ax.text(8.3, 4.3, "reconstructed only\nwhen needed\n(folded into weights)",
        ha="center", fontsize=7.8, color="#444")
ax.text(5.0, 3.6, "~14x smaller\nSAME softmax attention", ha="center",
        fontsize=8.5, color=GREEN)

fig.tight_layout()
fig.savefig(f"{OUT}/mla_compression.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# FIG 3: GDN recurrent state update (delta rule + gate)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 4.8))
ax.axis("off"); ax.set_xlim(0,14); ax.set_ylim(0,10)
ax.set_title("GDN: one fixed-size state matrix S, updated per token (no cache, O(1) memory)",
             fontsize=12)

# timeline of states
xs = [1.0, 5.0, 9.0]
for i,x in enumerate(xs):
    box(ax, x, 4.2, 2.4, 2.4, f"State S_{i}\n(d x d matrix)\nFIXED size", "#34495e", fs=9.5)
# arrows between
for i in range(2):
    arrow(ax, xs[i]+2.4, 5.4, xs[i+1], 5.4, lw=2.2, color=DARK)
arrow(ax, xs[2]+2.4, 5.4, xs[2]+1.3, 5.4, lw=2.2, color=DARK)
ax.text(12.0, 5.7, "read out:\ny_t = q_t · S_t", fontsize=9, color=BLUE)

# token inputs
for i,x in enumerate(xs):
    box(ax, x, 7.6, 2.4, 1.2, f"token {i}\n(k_t, v_t, q_t)", GREY, fs=8.5)
    arrow(ax, x+1.2, 7.6, x+1.2, 6.6, color=GREY)

# update rule box
box(ax, 0.6, 0.5, 12.8, 2.5,
    "Gated Delta update:   S_t  =   α_t · S_{t-1} · (I − β_t k_t k_tᵀ)   +   β_t v_t k_tᵀ",
    "#f4f4f4", ec="#333", fs=13, tc=DARK)
ax.text(3.2, 0.85, "α_t = forget GATE\n(decay old memory)", fontsize=8.5, color=RED, ha="center")
ax.text(6.9, 0.85, "(I − β k kᵀ) = DELTA rule\nerase this key's old value", fontsize=8.5,
        color=ORANGE, ha="center")
ax.text(10.9, 0.85, "+ β v kᵀ = write\nnew value for key", fontsize=8.5, color=GREEN, ha="center")

fig.tight_layout()
fig.savefig(f"{OUT}/gdn_state_update.png", dpi=130)
plt.close(fig)

# ----------------------------------------------------------------------------
# FIG 4: Hybrid stack — how GDN + MLA/full-attn interleave (Kimi-Linear style)
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 7.2))
ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,13)
ax.set_title("Hybrid stack: mostly GDN, a few full-attention layers\n(e.g. Kimi Linear ≈ 3 GDN : 1 MLA)",
             fontsize=11)
# pattern bottom->top
pattern = ["GDN","GDN","GDN","MLA (full attn)"]*3
labels = pattern
y = 0.6
for i,lab in enumerate(labels):
    is_full = "MLA" in lab
    col = BLUE if is_full else GREEN
    box(ax, 2.2, y, 5.6, 0.82, f"L{i}:  {lab}", col, fs=9)
    y += 0.95
arrow(ax, 5.0, 0.4, 5.0, y-0.1, color=GREY, lw=1.2, style="-|>")
ax.text(8.4, 6.5, "GDN layers:\nO(N), no cache,\ncheap majority",
        fontsize=8.5, color=GREEN, ha="center", rotation=0)
ax.text(8.4, 9.6, "MLA layers:\nexact attention,\nglobal recall",
        fontsize=8.5, color=BLUE, ha="center")
ax.text(1.0, 6.5, "tokens\nflow up", fontsize=8.5, color=GREY, ha="center", rotation=90)
fig.tight_layout()
fig.savefig(f"{OUT}/hybrid_stack.png", dpi=130)
plt.close(fig)

print("saved 4 figures to", OUT)
