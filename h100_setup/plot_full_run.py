"""A full DECODE step in nano-vllm: what's INSIDE the CUDA graph (embed + 28 layers)
vs OUTSIDE it (CPU prep + copy-in above; LM head + sampler + .tolist below)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# colors by kernel kind
CMP = "#4FA06B"   # @torch.compile (Triton): norms, RoPE, SiLU*Mul, Sampler
CUS = "#E0803B"   # hand-written / library custom: store_kvcache (triton), flash-attn
GEM = "#4C86C6"   # cuBLAS GEMM (nvjet): qkv/o/gate_up/down/lm_head
OTH = "#9E9E9E"   # other: embed gather, copies, .tolist
RED = "#C0392B"

fig, ax = plt.subplots(figsize=(14.5, 16))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

def chip(x, y, w, text, color, h=3.3, fs=7.2, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y-h/2), w, h, boxstyle="round,pad=0.05,rounding_size=0.3",
                                lw=1.0, edgecolor="#555", facecolor=color, zorder=4))
    ax.text(x+w/2, y, text, ha="center", va="center", fontsize=fs, color=tc, fontweight="bold", zorder=5)

def arr(x1, y1, x2, y2, c="#666", lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=11, lw=lw, color=c, zorder=2))

ax.text(50, 98, "A full DECODE step: inside the CUDA graph (embed + 28 layers) vs outside (CPU prep, LM head, sampler)",
        ha="center", fontsize=12.5, fontweight="bold")

# ---------- (1) OUTSIDE: CPU prep + copy-in ----------
ax.text(4, 94, "① OUTSIDE GRAPH — CPU prep + copy-in", fontsize=10.5, fontweight="bold", color="#333")
chip(8, 90, 38, "prepare_decode (CPU): build input_ids / positions /\nslot_mapping / context_lens / block_tables  (pinned)", OTH, h=5, fs=7)
arr(46, 90, 54, 90)
chip(54, 90, 38, "copy into graph_vars  (H2D pinned→HBM  +  DtoD\ninto the static buffers the graph was captured against)", OTH, h=5, fs=7)
arr(73, 87.3, 73, 84.5)

# ---------- (2) CUDA GRAPH ----------
ax.add_patch(FancyBboxPatch((4, 25), 92, 59, boxstyle="round,pad=0.3,rounding_size=1.2",
                            lw=2.0, edgecolor=RED, facecolor="#FDF3F2", linestyle="--", zorder=1))
ax.text(6, 81.5, "② CUDA GRAPH  —  replayed as ONE  cudaGraphLaunch  (decode only; captured per batch-size bucket [1,2,4,…,512])",
        fontsize=10, fontweight="bold", color=RED)
ax.text(93.5, 53, "~432 kernels\n(15/layer × 28\n+ embed + norm)", ha="right", va="center", fontsize=7.5, color=RED, style="italic")

chip(40, 77, 20, "embed_tokens (gather)", OTH, fs=7.5)
arr(50, 75.3, 50, 73.2)

# decoder-layer box
ax.add_patch(FancyBboxPatch((6, 40), 82, 32, boxstyle="round,pad=0.2,rounding_size=0.8",
                            lw=1.4, edgecolor="#888", facecolor="white", zorder=2))
ax.text(8, 70, "Decoder layer", fontsize=9.5, fontweight="bold")
ax.add_patch(FancyBboxPatch((77, 68.5), 9, 3, boxstyle="round,pad=0.1,rounding_size=0.5", lw=1.2, edgecolor=RED, facecolor="#FBEAEA", zorder=3))
ax.text(81.5, 70, "× 28", ha="center", va="center", fontsize=9, color=RED, fontweight="bold", zorder=4)

# attention row
ax.text(8, 63.5, "ATTENTION", fontsize=8, fontweight="bold", color="#4C86C6")
aw, ax0, ay = 10.6, 8, 60
attn = [("input_LN", CMP), ("qkv_proj", GEM), ("q/k_norm", CMP), ("RoPE", CMP),
        ("store_kv", CUS), ("flash-attn", CUS), ("o_proj", GEM)]
for i, (t, c) in enumerate(attn):
    x = ax0 + i*(aw+0.7)
    chip(x, ay, aw, t, c, fs=6.8)
    if i: arr(ax0+i*(aw+0.7)-0.7, ay, x, ay, lw=1.0)
# down to mlp
arr(ax0+6*(aw+0.7)+aw/2, 58.3, 20, 52.7, c="#888")
ax.text(48, 55, "residual threaded; add fused into next norm", fontsize=6.8, color="#999", style="italic", ha="center")

# mlp row
ax.text(8, 53.5, "MLP", fontsize=8, fontweight="bold", color="#4FA06B")
mw, mx0, my = 15, 14, 50
mlp = [("post_attn_LN", CMP), ("gate_up_proj", GEM), ("SiLU·Mul", CMP), ("down_proj", GEM)]
for i, (t, c) in enumerate(mlp):
    x = mx0 + i*(mw+1.2)
    chip(x, my, mw, t, c, fs=6.8)
    if i: arr(mx0+i*(mw+1.2)-1.2, my, x, my, lw=1.0)
ax.text(46, 44.5, "output → next layer's input_layernorm  (fused add_rms)", fontsize=6.8, color="#999", style="italic", ha="center")

arr(50, 39.7, 50, 37.2)
chip(38, 34, 24, "final norm (model.norm)", CMP, fs=7.5)
arr(50, 32.3, 50, 29.8)
chip(35, 27, 30, "outputs = hidden states  [bs, hidden]", OTH, fs=7.5)
arr(50, 24.5, 50, 21.5, c=RED, lw=1.6)

# ---------- (3) OUTSIDE: eager tail ----------
ax.text(4, 19, "③ OUTSIDE GRAPH — eager, after replay", fontsize=10.5, fontweight="bold", color="#333")
chip(6, 14, 26, "compute_logits — LM head\n(ParallelLMHead: GEMM + gather)", GEM, h=5, fs=7)
arr(32, 14, 37, 14)
chip(37, 14, 26, "Sampler (@torch.compile)\nsoftmax / exponential / argmax", CMP, h=5, fs=7)
arr(63, 14, 68, 14)
chip(68, 14, 26, ".tolist()  — D2H copy\n(tokens → CPU; implicit SYNC)", OTH, h=5, fs=7)
ax.text(81, 9.3, "← this D2H is the per-step sync point", fontsize=7, color=RED, ha="center", style="italic")

# ---------- legend ----------
ax.text(4, 5.2, "kernel kind:", fontsize=8, fontweight="bold")
lx = 16
for c, t in [(CMP, "@torch.compile (Triton): norms, RoPE, SiLU·Mul, Sampler"),
             (GEM, "cuBLAS GEMM (nvjet)"), (CUS, "custom: store_kvcache / flash-attn"), (OTH, "other / copies")]:
    ax.add_patch(FancyBboxPatch((lx, 4.4), 1.6, 1.5, boxstyle="round,pad=0.03", lw=0.8, edgecolor="#777", facecolor=c))
    ax.text(lx+2.1, 5.15, t, fontsize=6.8, va="center", color="#333"); lx += len(t)*0.5 + 6

plt.tight_layout()
out = "/home/cicichen/nano-vllm/h100_setup/full_run.png"
plt.savefig(out, dpi=135, bbox_inches="tight", facecolor="white")
print("saved", out)
