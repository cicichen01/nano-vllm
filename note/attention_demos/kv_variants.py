import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/home/cicichen/nano-vllm/h100_setup"
RED="#c0392b"; ORANGE="#e08b25"; PURPLE="#8e44ad"; BLUE="#2c6fbb"; GREEN="#3a9d5d"

# toy model: 32 heads, head_dim=128, 60 layers, bf16
n_layers=60; hd=128
seq=np.arange(0,32769,256)
def MB(floats_per_tok_layer):
    return seq*n_layers*floats_per_tok_layer*2/1e6

mha = 2*32*hd      # 8192  -> 32 KV heads
gqa = 2*8*hd       # 2048  -> 8 KV groups
mqa = 2*1*hd       # 256   -> 1 KV head
mla = 512+64       # 576   -> latent + decoupled rope key

fig, ax = plt.subplots(figsize=(9.0,5.6))
ax.plot(seq, MB(mha), color=RED,    lw=2.6, label=f"MHA — 32 KV heads = {mha} floats/tok/layer")
ax.plot(seq, MB(gqa), color=ORANGE, lw=2.4, label=f"GQA — 8 KV groups  = {gqa} floats/tok/layer")
ax.plot(seq, MB(mla), color=BLUE,   lw=2.4, label=f"MLA — latent ~512  = {mla} floats/tok/layer")
ax.plot(seq, MB(mqa), color=PURPLE, lw=2.4, label=f"MQA — 1 KV head    = {mqa} floats/tok/layer")
ax.plot(seq, np.full_like(seq, n_layers*32*hd*hd*2/1e6, dtype=float),
        color=GREEN, lw=2.4, ls="--", label="GDN — fixed state, FLAT (O(1) in N)")

ax.set_xlabel("sequence length N (tokens)")
ax.set_ylabel("attention-state memory (MB, bf16)")
ax.set_title("All KV-cache tricks are LINEAR in N — they only change the slope\n"
             "(GDN is the odd one out: flat)", fontsize=11)
ax.legend(fontsize=8.4, loc="upper left")
ax.grid(alpha=0.3)

# zoom inset for the small ones
axin = ax.inset_axes([0.55,0.12,0.4,0.42])
for f,c in [(gqa,ORANGE),(mla,BLUE),(mqa,PURPLE)]:
    axin.plot(seq, MB(f), color=c, lw=2)
axin.plot(seq, np.full_like(seq, n_layers*32*hd*hd*2/1e6, dtype=float), color=GREEN, lw=2, ls="--")
axin.set_title("zoom: GQA / MLA / MQA / GDN", fontsize=7.5)
axin.tick_params(labelsize=6)
axin.grid(alpha=0.3)
ax.text(0.55,0.56,"MQA < MLA < GQA < MHA  (slope)", transform=ax.transAxes,
        fontsize=8.2, color="#333")

fig.tight_layout()
fig.savefig(f"{OUT}/kv_variants_slopes.png", dpi=130)
print("saved")
