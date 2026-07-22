import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT="/home/cicichen/nano-vllm/h100_setup"
RED="#c0392b"; GREEN="#3a9d5d"; BLUE="#2c6fbb"; ORANGE="#e08b25"; DARK="#333"
def rect(ax,x,y,w,h,t,fc,fs=9,tc="white"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.01,rounding_size=0.03",
        fc=fc,ec="#333",lw=1.3)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc)

fig,ax=plt.subplots(figsize=(12,4.6)); ax.axis("off"); ax.set_xlim(0,14); ax.set_ylim(0,8)
ax.set_title("MLA = low-rank factorization of the K/V projection; cache the small middle",fontsize=12)

# MHA side
rect(ax,0.4,3.0,3.0,2.6,"W_K\nd_model x (n_heads·head_dim)\nFULL RANK\n(cache = full K)",RED,fs=8.5)
ax.text(1.9,2.4,"MHA",ha="center",fontsize=10,color=RED,weight="bold")

ax.text(4.3,4.3,"≈",fontsize=26,ha="center")

# factored
rect(ax,5.2,3.0,2.4,2.6,"W_UK\nd_c → n_heads·head_dim\nPER-HEAD up-proj",BLUE,fs=8)
ax.text(8.0,4.3,"·",fontsize=26,ha="center")
rect(ax,8.6,3.4,2.2,1.8,"W_DKV\nd_model → d_c\ndown-proj",ORANGE,fs=8)
ax.text(8.0,2.4,"MLA (factored)",ha="center",fontsize=10,color=GREEN,weight="bold")

# what gets cached
rect(ax,11.2,3.8,2.4,1.0,"cache c = W_DKV·h\n(d_c ≈ 512)",GREEN,fs=8.5)
ax.annotate("",xy=(11.2,4.3),xytext=(10.8,4.3),arrowprops=dict(arrowstyle="-|>",color=GREEN,lw=1.5))

ax.text(7.0,1.3,"cache the SHARED low-rank middle (small)  →  each head decompresses via its OWN W_UK (distinct K, full head_dim)",
        ha="center",fontsize=8.8,color=DARK)
ax.text(7.0,0.5,"residual / Q / O / MLP all stay full-width  —  only the CACHED KV intermediate is low-rank",
        ha="center",fontsize=8.8,color=GREEN,weight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/mla_lowrank_factorization.png",dpi=130); print("saved")
