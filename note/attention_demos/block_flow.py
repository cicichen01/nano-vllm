import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT="/home/cicichen/nano-vllm/h100_setup"
BLUE="#2c6fbb"; GREEN="#3a9d5d"; ORANGE="#e08b25"; PURPLE="#8e44ad"; GREY="#888"; DARK="#333"; RED="#c0392b"
def box(ax,x,y,w,h,t,fc,fs=9,tc="white",ec="#333"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",
        fc=fc,ec=ec,lw=1.3)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc)
def arr(ax,x1,y1,x2,y2,c=DARK,lw=1.8):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=14,color=c,lw=lw))

fig,ax=plt.subplots(figsize=(11.5,7.6)); ax.axis("off"); ax.set_xlim(0,14); ax.set_ylim(0,15)
ax.set_title("One transformer block: heads are MIXED by W_O, THEN a single shared MLP",fontsize=12)

# residual stream in
box(ax,0.4,13.4,3.0,1.0,"h_t  (d_model=4096)",GREY)
arr(ax,1.9,13.4,1.9,12.7)
ax.text(0.3,7.5,"RESIDUAL STREAM (d_model)  — full width the whole way",rotation=90,
        fontsize=8.5,color=RED,va="center")

# heads
head_cols=[BLUE,GREEN,ORANGE,PURPLE]
for i,x in enumerate([1.6,4.1,6.6,9.1]):
    box(ax,x,10.9,2.1,1.5,f"head {i}\nattn → out^{i}\n(head_dim=128)",head_cols[i],fs=7.8)
    arr(ax,1.9,12.7,x+1.05,12.4,c=GREY,lw=1.2)
ax.text(11.9,11.6,"4 heads run\nINDEPENDENTLY\n(here)",fontsize=8,color=DARK,ha="center")

# concat
box(ax,2.0,8.9,8.5,1.0,"concat all heads  →  (n_heads·head_dim = 4096)",DARK,fs=9)
for i,x in enumerate([1.6,4.1,6.6,9.1]):
    arr(ax,x+1.05,10.9,x+1.6,9.9,c=head_cols[i],lw=1.3)

# W_O  <-- the mixing
box(ax,3.0,6.9,6.5,1.1,"W_O   (mix heads → d_model)\n★ THIS is where heads combine",RED,fs=9.5)
arr(ax,6.25,8.9,6.25,8.0,c=RED)
ax.text(11.9,7.45,"heads are now\nFULLY MIXED\ninto one vector",fontsize=8,color=RED,ha="center")

# add residual
box(ax,4.7,5.2,3.0,0.9,"+ residual",GREY,fs=9)
arr(ax,6.25,6.9,6.25,6.1,c=DARK)
arr(ax,1.9,12.7,1.9,5.65,c=GREY,lw=1.0)  # skip line
arr(ax,1.9,5.65,4.7,5.65,c=GREY,lw=1.0)

# MLP  <-- shared, sees mixed vector
box(ax,3.0,3.1,6.5,1.3,"MLP  (one shared FFN)\nsees the MIXED d_model vector — NOT per-head",GREEN,fs=9.5)
arr(ax,6.25,5.2,6.25,4.4,c=DARK)
ax.text(11.9,3.75,"MLP is shared:\nno notion of heads\nby this point",fontsize=8,color=GREEN,ha="center")

# add + out
box(ax,4.7,1.3,3.0,0.9,"+ residual → out",GREY,fs=9)
arr(ax,6.25,3.1,6.25,2.2,c=DARK)

fig.tight_layout(); fig.savefig(f"{OUT}/transformer_block_heads_mlp.png",dpi=130); print("saved")
