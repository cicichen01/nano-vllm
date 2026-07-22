import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT="/home/cicichen/nano-vllm/h100_setup"
RED="#c0392b"; GREEN="#3a9d5d"; BLUE="#2c6fbb"; GREY="#999"; DARK="#333"; ORANGE="#e08b25"
def box(ax,x,y,w,h,t,fc,fs=8.5,tc="white",ec="#333",lw=1.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.01,rounding_size=0.04",
        fc=fc,ec=ec,lw=lw)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc)

fig,axes=plt.subplots(1,3,figsize=(14,5.6))
for ax in axes: ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)

# --- 1. shrink d_model ---
ax=axes[0]
ax.set_title("(A) Shrink d_model\n= cut the WHOLE stream everywhere  ✗",fontsize=10.5,color=RED)
box(ax,3.2,8.3,3.6,1.0,"residual stream\nNARROW (2048)",RED,fs=9)
box(ax,3.4,6.6,3.2,0.9,"attention (narrow)",GREY,fs=8)
box(ax,3.4,5.3,3.2,0.9,"MLP (narrow)",GREY,fs=8)
box(ax,3.4,4.0,3.2,0.9,"attention (narrow)",GREY,fs=8)
box(ax,3.4,2.7,3.2,0.9,"MLP (narrow)",GREY,fs=8)
ax.text(5.0,1.4,"capacity cut in\nEVERY layer, every op",ha="center",fontsize=8.5,color=RED)

# --- 2. MQA ---
ax=axes[1]
ax.set_title("(B) MQA\n= 1 shared K/V, all heads IDENTICAL  ✗ quality",fontsize=10.5,color=ORANGE)
box(ax,3.5,8.3,3.0,1.0,"residual FULL (4096)",GREEN,fs=8.5)
box(ax,4.0,6.6,2.0,1.0,"ONE K/V head",ORANGE,fs=8.5)
for i,x in enumerate([0.6,2.5,4.4,6.3,8.0]):
    box(ax,x,4.4,1.5,0.9,f"Q head {i}",BLUE,fs=7.5)
    ax.annotate("",xy=(5.0,6.6),xytext=(x+0.75,5.3),
        arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=1))
ax.text(5.0,3.2,"every head sees the\nSAME key/value\n→ lost head diversity",ha="center",fontsize=8.3,color=ORANGE)

# --- 3. MLA ---
ax=axes[2]
ax.set_title("(C) MLA\n= shared latent, PER-HEAD decompress  ✓",fontsize=10.5,color=GREEN)
box(ax,3.5,8.3,3.0,1.0,"residual FULL (4096)",GREEN,fs=8.5)
box(ax,3.7,6.6,2.6,1.0,"latent c (512)\nlow-rank cache",BLUE,fs=8.3)
for i,(x,col) in enumerate(zip([0.5,2.6,4.7,6.8,8.4],["#1f6fb2","#2a7d3f","#8e44ad","#d35400","#16a085"])):
    box(ax,x,4.3,1.5,1.0,f"W_UK^{i}\n→ own K/V",col,fs=7)
    ax.annotate("",xy=(x+0.75,5.3),xytext=(5.0,6.6),
        arrowprops=dict(arrowstyle="-|>",color=col,lw=1))
ax.text(5.0,3.0,"one small cache, but each head\nreconstructs a DIFFERENT key/value\n(low-rank, not low-diversity)",
        ha="center",fontsize=8.0,color=GREEN)
ax.text(5.0,1.6,"d_model untouched +\nhead diversity kept",ha="center",fontsize=8.5,color=GREEN,weight="bold")

fig.tight_layout(); fig.savefig(f"{OUT}/mla_vs_shrink_capacity.png",dpi=130); print("saved")
