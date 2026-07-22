import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT="/home/cicichen/nano-vllm/h100_setup"
DARK="#2c3e50"; RED="#c0392b"; GREEN="#3a9d5d"; BLUE="#2c6fbb"; GREY="#7f8c8d"; ORANGE="#e08b25"
PAIR={GREY:("#7f8c8d","#9aa7ab"), RED:("#c0392b","#d9695b"),
      BLUE:("#2c6fbb","#5a8fd0"), ORANGE:("#e08b25","#eeb056")}

d_model, n_heads, head_dim, d_c = 4096, 32, 128, 512
S = 2.0/d_model
NDISP = 8

def mat(ax, x, yb, out_dim, in_dim, name, dims, base, split=None, note=None):
    """width ∝ out (columns), height ∝ in (rows).
       split='col' -> heads are output columns (vertical strips)
       split='row' -> heads are input rows (horizontal strips)"""
    w, h = out_dim*S, in_dim*S
    if split is None:
        ax.add_patch(Rectangle((x,yb), w, h, fc=base, ec="#222", lw=1.4))
    elif split=="col":
        sw=w/NDISP
        for i in range(NDISP):
            ax.add_patch(Rectangle((x+i*sw,yb), sw, h, fc=PAIR[base][i%2], ec="#222", lw=0.5))
        ax.add_patch(Rectangle((x,yb), w, h, fill=False, ec="#222", lw=1.4))
    elif split=="row":
        sh=h/NDISP
        for i in range(NDISP):
            ax.add_patch(Rectangle((x,yb+i*sh), w, sh, fc=PAIR[base][i%2], ec="#222", lw=0.5))
        ax.add_patch(Rectangle((x,yb), w, h, fill=False, ec="#222", lw=1.4))
    ax.text(x+w/2, yb+h+0.12, name, ha="center", va="bottom", fontsize=9.3, weight="bold")
    ax.text(x+w/2, yb+h/2, dims, ha="center", va="center", fontsize=7.0,
            color="white", rotation=0 if w>0.6 else 90)
    if note: ax.text(x+w/2, yb-0.16, note, ha="center", va="top", fontsize=6.8, color="#444")
    return x+w

fig, ax = plt.subplots(figsize=(15.5,10)); ax.axis("off")
ax.set_xlim(0,15.5); ax.set_ylim(0,10.3)
ax.set_title("Per-head structure  (convention: width ∝ output, height ∝ input;  h·W → output columns)\n"
             "heads = output COLUMNS for Q/K/V/UK/UV (vertical split);  heads = input ROWS for W_O "
             "(horizontal).  W_DKV is ONE shared block.", fontsize=11)

# ---- Band 1: MHA ----
y1=7.0
ax.text(0.15,y1+1.0,"MHA\nstored", fontsize=11, weight="bold", color=DARK, va="center")
x=2.5
x=mat(ax,x,y1,d_model,d_model,"W_Q","4096→4096",GREY,split="col",note="heads = output columns")+0.5
x=mat(ax,x,y1,d_model,d_model,"W_K","4096→4096",RED,split="col",note="per head (vertical)")+0.5
x=mat(ax,x,y1,d_model,d_model,"W_V","4096→4096",RED,split="col",note="per head (vertical)")+0.5
x=mat(ax,x,y1,d_model,d_model,"W_O","4096→4096",GREY,split="row",note="heads = input ROWS")+0.4
ax.text(13.6,y1+1.0,"cache/tok\nK+V=8192",fontsize=8,color=RED,ha="center",va="center",
        bbox=dict(boxstyle="round",fc="#fdecea",ec=RED))

# ---- Band 2: MLA ----
y2=3.8
ax.text(0.15,y2+1.0,"MLA\nstored", fontsize=11, weight="bold", color=DARK, va="center")
x=2.5
x=mat(ax,x,y2,d_model,d_model,"W_Q","4096→4096",GREY,split="col",note="per head (vertical)")+0.5
x=mat(ax,x,y2,d_c,d_model,"W_DKV","4096→512",GREEN,split=None,note="SHARED\n(down-proj, cached)")+0.6
x=mat(ax,x,y2,d_model,d_c,"W_UK","512→4096",BLUE,split="col",note="per head (vertical)")+0.5
x=mat(ax,x,y2,d_model,d_c,"W_UV","512→4096",BLUE,split="col",note="per head (vertical)")+0.6
x=mat(ax,x,y2,d_model,d_model,"W_O","4096→4096",GREY,split="row",note="per head (rows)")+0.4
ax.text(13.7,y2+1.0,"cache/tok\nc=512\n≈16× less",fontsize=8,color=GREEN,ha="center",va="center",
        bbox=dict(boxstyle="round",fc="#eafaf1",ec=GREEN))
ax.text(7.3,y2-0.75,"per head:  W_K^(h) = W_DKV then W_UK^(h)   "
        "(SAME shared down-proj W_DKV, per-head up-proj column-slice W_UK^(h))",
        fontsize=8.2,color=DARK,ha="center")

# ---- Band 3: absorbed ----
y3=0.6
ax.text(0.15,y3+1.1,"MLA absorbed\n(equivalent)", fontsize=10.5, weight="bold", color=DARK, va="center")
# n_heads separate tall-thin W_Q'^(h): map d_model->d_c  (out=512 width, in=4096 height)
x0=2.6; bw=d_c*S; bh=d_model*S
for i in range(5):
    ax.add_patch(Rectangle((x0+i*(bw+0.12), y3), bw, bh, fc=PAIR[ORANGE][i%2], ec="#222", lw=0.7))
grp_c=x0+2.5*(bw+0.12)-0.06
ax.text(grp_c, y3+bh+0.28, "W_Q'^(h) = W_UKᵀW_Q", ha="center", fontsize=8.8, weight="bold")
ax.text(grp_c, y3+bh+0.06, "4096→512, ×n_heads, rank≤128", ha="center", fontsize=7, color="#444")
ax.text(grp_c, y3-0.16, "n_heads SEPARATE tall-thin matrices\n(concat = 4096×16384, NOT a square)",
        ha="center", va="top", fontsize=6.8, color="#444")
x=x0+5*(bw+0.12)+1.1
x=mat(ax,x,y3,d_c,d_model,"W_DKV (=K/V)","4096→512",GREEN,split=None,note="shared latent c")+0.9

# W_O' = W_O · W_UV : in = n_h·d_c = 16384 (4× W_O's input!), out = d_model.
# true height would be 8.0 -> cap it and annotate. heads = input ROWS (horizontal split).
wo_w=d_model*S; wo_h=1.9   # capped (true ~8.0)
sh=wo_h/NDISP
for i in range(NDISP):
    ax.add_patch(Rectangle((x,y3+i*sh), wo_w, sh, fc=PAIR[ORANGE][i%2], ec="#222", lw=0.5))
ax.add_patch(Rectangle((x,y3), wo_w, wo_h, fill=False, ec="#222", lw=1.4))
ax.text(x+wo_w/2, y3+wo_h+0.12, "W_O' = W_O·W_UV", ha="center", va="bottom", fontsize=9, weight="bold")
ax.text(x+wo_w/2, y3+wo_h/2, "16384→4096", ha="center", va="center", fontsize=7.2, color="white")
ax.text(x+wo_w/2, y3-0.16, "in=16384 (=n_h·d_c, 4× W_O)\nNOT to scale;  heads = input rows",
        ha="center", va="top", fontsize=6.8, color="#444")
ax.text(x+wo_w+0.5,y3+1.2,"→ each head dots its own tall-thin\n   W_Q'^(h) against the ONE shared c\n   (d_c=512);  W_UV is absorbed into\n   W_O → W_O' (input side inflates 4×)",
        fontsize=8,color=ORANGE,ha="left",va="center",
        bbox=dict(boxstyle="round",fc="#fef5e7",ec=ORANGE))

fig.tight_layout()
fig.savefig(f"{OUT}/mla_weight_dims_heads.png", dpi=130); print("saved")
