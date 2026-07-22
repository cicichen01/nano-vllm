import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT="/home/cicichen/nano-vllm/h100_setup"
RED="#c0392b"; GREEN="#3a9d5d"; BLUE="#2c6fbb"; GREY="#888"; DARK="#222"; ORANGE="#e08b25"
def box(ax,x,y,w,h,t,fc,fs=9,tc="white",ec="#333"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",
        fc=fc,ec=ec,lw=1.2)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc)
def arr(ax,x1,y1,x2,y2,c=DARK,lw=1.6):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,color=c,lw=lw))

fig,axes=plt.subplots(1,2,figsize=(12.5,5.4))

# LEFT: naive - reconstruct per cached token
ax=axes[0]; ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.set_title("NAIVE MLA: up-project every cached token\n= O(N) extra matmuls per step",fontsize=10.5,color=RED)
box(ax,0.3,8.4,2.3,1.1,"new query hₜ",GREY)
for i,y in enumerate([6.2,4.6,3.0,1.4]):
    lbl = "c_i (512)" if i<3 else "  ... N of them"
    box(ax,0.3,y,1.7,1.1,f"cache\n{lbl}",BLUE,fs=8)
    box(ax,2.6,y,2.2,1.1,"W_UK·c_i\n→ full K",ORANGE,fs=8)
    arr(ax,2.0,y+0.55,2.6,y+0.55,c=RED)
    box(ax,5.3,y,2.2,1.1,"dot with q",GREY,fs=8)
    arr(ax,4.8,y+0.55,5.3,y+0.55)
box(ax,3.0,0.1,4.6,0.9,"N up-projections  → work grows with context",RED,fs=9)

# RIGHT: absorbed
ax=axes[1]; ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,10)
ax.set_title("ABSORBED MLA: fold W_UK into W_Q (once/step)\n= O(1) extra, dot in latent space",fontsize=10.5,color=GREEN)
box(ax,0.3,8.4,2.3,1.1,"new query hₜ",GREY)
box(ax,3.2,8.4,3.4,1.1,"q'ₜ = (W_UKᵀW_Q)hₜ\nONCE per step",GREEN,fs=8.5)
arr(ax,2.6,8.95,3.2,8.95,c=GREEN)
for i,y in enumerate([6.2,4.6,3.0,1.4]):
    lbl = "c_i (512)" if i<3 else "  ... N of them"
    box(ax,0.3,y,1.7,1.1,f"cache\n{lbl}",BLUE,fs=8)
    box(ax,3.0,y,3.0,1.1,"q'ₜ · c_i  (512-dim dot)\nno reconstruction",DARK,fs=8)
    arr(ax,2.0,y+0.55,3.0,y+0.55,c=GREEN)
arr(ax,4.9,8.4,4.9,7.4,c=GREEN)
box(ax,2.8,0.1,4.6,0.9,"just N cheap dot-products (small dim)",GREEN,fs=9)

fig.tight_layout(); fig.savefig(f"{OUT}/mla_absorption.png",dpi=130); print("saved")
