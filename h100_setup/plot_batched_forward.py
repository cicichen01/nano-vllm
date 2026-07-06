"""Figure: how input tensors are constructed for a batched forward pass,
PREFILL (flat varlen packing) vs DECODE (1 token/seq + paged KV via block_tables)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

CA, CB, CC = "#4C86C6", "#E0803B", "#4FA06B"   # seq A / B / C colors
GREY, LGREY, RED = "#555555", "#E8E8E8", "#C0392B"

def box(ax, x, y, w, h, text, fc="white", ec=GREY, fs=10, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.03",
                                linewidth=1.2, edgecolor=ec, facecolor=fc))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal")

def label(ax, x, y, text, fs=11, color="black", ha="left", bold=True):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=14, linewidth=lw, color=color))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15.5, 22))
for ax in (ax1, ax2, ax3):
    ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis("off")

# =================== PREFILL ===================
ax1.text(1, 50, "PREFILL  —  flat 'varlen' packing:  all scheduled tokens of all seqs concatenated into ONE 1-D buffer",
         fontsize=13.5, fontweight="bold", color="#1A1A1A")

# --- separate sequences ---
label(ax1, 1, 45, "3 sequences being prefilled (different lengths):", fs=11)
seqs = [("Seq A (len 4)", CA, ["A0","A1","A2","A3"]),
        ("Seq B (len 3)", CB, ["B0","B1","B2"]),
        ("Seq C (len 2)", CC, ["C0","C1"])]
W = 4.6
y = 40
for name, c, toks in seqs:
    label(ax1, 1, y+1.4, name, fs=10, color=c)
    for i, t in enumerate(toks):
        box(ax1, 22 + i*W, y, W-0.4, 2.8, t, fc=c, tc="white", fs=10, bold=True)
    y -= 4.0

arrow(ax1, 30, 28.5, 30, 25.5, lw=2, color="#333")
ax1.text(31.5, 27, "pack / concat (no padding)", fontsize=10, color="#333", va="center")

# --- flat input_ids ---
flat = [("A0",CA),("A1",CA),("A2",CA),("A3",CA),("B0",CB),("B1",CB),("B2",CB),("C0",CC),("C1",CC)]
label(ax1, 1, 23.2, "input_ids  [total_tokens = 9]", fs=10.5)
x0 = 22
for i,(t,c) in enumerate(flat):
    box(ax1, x0 + i*W, 22, W-0.4, 2.6, t, fc=c, tc="white", fs=9.5, bold=True)
# seq boundary separators
for bx in [4,7]:
    ax1.plot([x0+bx*W-0.2]*2, [21.6, 25.0], color="black", lw=2.2)

# cu_seqlens markers
label(ax1, 1, 19.2, "cu_seqlens_q = [0, 4, 7, 9]", fs=10.5, color=RED)
for val in [0,4,7,9]:
    xx = x0 + val*W - 0.2
    arrow(ax1, xx, 20.6, xx, 21.5, color=RED, lw=1.4)
    ax1.text(xx, 20.1, str(val), ha="center", va="center", fontsize=9, color=RED, fontweight="bold")
ax1.text(x0+9*W+1.5, 19.2, "← marks where each seq starts/ends in the packed buffer",
         fontsize=9.5, color=RED, va="center")

# positions
pos = [0,1,2,3,0,1,2,0,1]
label(ax1, 1, 16.2, "positions", fs=10.5)
for i,p in enumerate(pos):
    box(ax1, x0 + i*W, 15, W-0.4, 2.4, str(p), fc=LGREY, fs=9.5)

# right: what consumes what
label(ax1, 1, 11.5, "Who uses the boundaries?", fs=11)
box(ax1, 1, 6.5, 30, 3.6, "Linear / RMSNorm / RoPE / MLP\n= PER-TOKEN → one big GEMM over all 9 tokens\n(shared weights, batching is FREE)",
    fc="#EAF3FB", ec=CA, fs=9.5)
box(ax1, 34, 6.5, 34, 3.6, "flash_attn_varlen_func(cu_seqlens_q, cu_seqlens_k)\n= ONLY op that needs boundaries →\nblock-diagonal mask (no cross-seq attention)",
    fc="#FBF0E7", ec=CB, fs=9.5)
# mini block-diagonal mask
mx, my, cell = 72, 5.6, 1.15
ax1.text(mx+4.5*cell, my+9.9, "attention mask (9×9)", fontsize=9, ha="center", color="#333")
blocks = [(0,4,CA),(4,7,CB),(7,9,CC)]
for i in range(9):
    for j in range(9):
        fc = "white"
        for s,e,c in blocks:
            if s<=i<e and s<=j<e and j<=i:   # within same seq + causal
                fc = c
        ax1.add_patch(Rectangle((mx+j*cell, my+(8-i)*cell), cell, cell,
                                facecolor=fc, edgecolor="#CCC", linewidth=0.4))

# =================== DECODE ===================
ax2.text(1, 50, "DECODE  —  exactly 1 new token per seq;  history lives in the paged KV cache, gathered via block_tables",
         fontsize=13.5, fontweight="bold", color="#1A1A1A")

decode = [("Seq A", CA, 8, [0,1]), ("Seq B", CB, 5, [2,3]), ("Seq C", CC, 6, [4,5])]  # (name,color,ctx_len,blocks)
label(ax2, 1, 45.5, "each seq: cached KV in blocks (block_size=4)  +  1 NEW token to process", fs=11)
y = 41
for name, c, ctx, blks in decode:
    label(ax2, 1, y+1.1, name, fs=10, color=c)
    # cached blocks
    for k,b in enumerate(blks):
        box(ax2, 12 + k*9, y, 8.4, 2.4, f"block {b}\n(KV)", fc="white", ec=c, fs=8.5)
    ax2.text(12+len(blks)*9+1.0, y+1.2, "+", fontsize=14, va="center", color="#333")
    box(ax2, 12+len(blks)*9+3, y, 6, 2.4, "NEW tok", fc=c, tc="white", fs=9, bold=True)
    y -= 4.0

arrow(ax2, 20, 30.0, 20, 27.5, lw=2, color="#333")
ax2.text(21.5, 28.7, "build 1 entry per seq", fontsize=10, color="#333", va="center")

x0 = 22; W2 = 7
names = [("Seq A",CA),("Seq B",CB),("Seq C",CC)]
# input_ids (one token per seq)
label(ax2, 1, 25.5, "input_ids  [B = 3]", fs=10.5)
for i,(n,c) in enumerate(names):
    box(ax2, x0+i*W2, 24.3, W2-0.5, 2.4, "tokᴬ".replace("ᴬ",n[-1]), fc=c, tc="white", fs=9.5, bold=True)
# positions
label(ax2, 1, 21.5, "positions = [7, 4, 5]", fs=10.5)
for i,p in enumerate([7,4,5]):
    box(ax2, x0+i*W2, 20.3, W2-0.5, 2.2, str(p), fc=LGREY, fs=9.5)
ax2.text(x0+3*W2+1.5, 21.5, "(= len(seq)-1, absolute position of the new token)", fontsize=9, color="#666", va="center")
# context_lens
label(ax2, 1, 17.7, "context_lens = [8, 5, 6]", fs=10.5, color=RED)
for i,cl in enumerate([8,5,6]):
    box(ax2, x0+i*W2, 16.5, W2-0.5, 2.2, str(cl), fc="#FBEAEA", ec=RED, fs=9.5)
ax2.text(x0+3*W2+1.5, 17.7, "← total KV length per seq (how far attention looks back)", fontsize=9, color=RED, va="center")
# block_tables
label(ax2, 1, 13.5, "block_tables  [B × max_blocks]", fs=10.5)
bt = [[0,1],[2,3],[4,5]]
for i,row in enumerate(bt):
    for j,b in enumerate(row):
        box(ax2, x0+i*W2 + j*3.2, 12.3, 3.0, 2.2, str(b), fc="white", ec=names[i][1], fs=9)
ax2.text(x0+3*W2+1.5, 13.4, "← which physical KV blocks hold each seq's history", fontsize=9, color="#666", va="center")
# slot_mapping
label(ax2, 1, 9.3, "slot_mapping = [31, 22, 25]", fs=10.5)
ax2.text(20, 9.3, "← 1 slot/seq: where THIS step's new K,V is written (block[-1]*block_size + offset)",
         fontsize=9, color="#666", va="center")

box(ax2, 1, 3.2, 67, 3.6,
    "flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=context_lens, block_table=block_tables)\n"
    "→ for each of the B query tokens, GATHER its own KV blocks and attend over its full context (Flash-Decoding).",
    fc="#FBF0E7", ec=CB, fs=9.5)

# =================== WHICH OPS MIX ACROSS TOKENS ===================
ax3.text(1, 50, "WHICH OPS MIX ACROSS TOKENS?  —  only attention.  Everything else is per-token (so it batches for free)",
         fontsize=13.5, fontweight="bold", color="#1A1A1A")

# tensor grid: tokens (rows) x features (cols)
gx, gy, cw, ch = 18, 33, 3.2, 2.6
nT, nF = 4, 6
for i in range(nT):
    for j in range(nF):
        ax3.add_patch(Rectangle((gx+j*cw, gy+(nT-1-i)*ch), cw, ch, facecolor="#EEF3F8", edgecolor="#BBB", lw=0.6))
for i in range(nT):
    ax3.text(gx+nF*cw+1, gy+(nT-1-i)*ch+ch/2, f"tok{i}", ha="left", va="center", fontsize=8.5, color="#555")
# feature axis (horizontal)
arrow(ax3, gx, gy-1.8, gx+nF*cw, gy-1.8, color=CA, lw=2)
ax3.text(gx+nF*cw/2, gy-3.3, "feature axis  →  per-token ops act/reduce HERE (within a row)", ha="center", va="center", fontsize=9, color=CA)
# token axis (vertical)
arrow(ax3, gx-3.5, gy, gx-3.5, gy+nT*ch, color=CB, lw=2)
ax3.text(gx-4.6, gy+nT*ch/2, "token axis\n↑ ATTENTION\nmixes HERE", ha="right", va="center", fontsize=9, color=CB)
ax3.text(gx+nF*cw+7.5, gy+nT*ch/2,
         "each row = one token, processed independently.\nReductions run ALONG a row (features) — never\nDOWN a column (tokens).  Only attention goes down.",
         ha="left", va="center", fontsize=9, color="#333")

# per-token column
ax3.add_patch(FancyBboxPatch((1, 11), 49, 15, boxstyle="round,pad=0.1,rounding_size=0.3", fc="#EAF3FB", ec=CA, lw=1.4))
ax3.text(3, 24.6, "PER-TOKEN  ·  one GEMM / elementwise over all tokens  ·  batching is FREE",
         fontsize=10.5, fontweight="bold", color=CA, va="center")
pt = ["• embedding lookup",
      "• RMSNorm / LayerNorm   (reduces over FEATURES, not tokens)",
      "• all Linears / GEMM  (qkv, o, gate_up, down, lm_head)",
      "• RoPE   (uses each token's own position index)",
      "• activations: SiLU / GELU / sigmoid / SwiGLU  (elementwise)",
      "• residual add",
      "• sampling softmax   (over VOCAB → per token)"]
for k, t in enumerate(pt):
    ax3.text(3, 22.3 - k*1.7, t, fontsize=9.2, color="#222", va="center")

# cross-token column
ax3.add_patch(FancyBboxPatch((52, 11), 46, 15, boxstyle="round,pad=0.1,rounding_size=0.3", fc="#FBF0E7", ec=CB, lw=1.4))
ax3.text(54, 24.6, "CROSS-TOKEN  ·  needs seq boundaries (cu_seqlens / block_tables)",
         fontsize=10.5, fontweight="bold", color=CB, va="center")
ax3.text(54, 21.2, "• ATTENTION  (the ONLY one):", fontsize=9.8, color="#222", va="center", fontweight="bold")
ax3.text(56, 18.4, "Q Kᵀ   →   softmax over KEYS   →   · V", fontsize=11, color=RED, va="center", fontweight="bold")
ax3.text(54, 15.3, "the sole op that combines DIFFERENT token\npositions in a standard transformer LLM.\n(within a seq for prefill; over cached KV for decode)",
         fontsize=9.2, color="#222", va="center")

# bottom rule
ax3.add_patch(FancyBboxPatch((1, 2.3), 97, 6.2, boxstyle="round,pad=0.1,rounding_size=0.3", fc="#F4F4F4", ec="#999", lw=1.1))
ax3.text(3, 6.6, "Rule of thumb:  an internal reduction ≠ cross-token.  Softmax mixes ONLY along the axis it is applied:",
         fontsize=9.8, color="#111", va="center", fontweight="bold")
ax3.text(3, 4.1, "over KEYS ⇒ attention (cross-token)      |      over VOCAB ⇒ sampling (per-token)      |      RMSNorm reduces over FEATURES ⇒ still per-token",
         fontsize=9.4, color="#333", va="center")

plt.tight_layout(pad=1.5)
out = "/home/cicichen/nano-vllm/h100_setup/batched_forward.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
