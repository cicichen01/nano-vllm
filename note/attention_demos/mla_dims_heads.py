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
NDISP = 8   # visible slices (represents n_heads=32)

def mat(ax, x, yb, cols, rows, name, dims, base, split=None, note=None):
    w, h = cols*S, rows*S
    if split is None:
        ax.add_patch(Rectangle((x,yb), w, h, fc=base, ec="#222", lw=1.4))
    elif split=="row":
        sh=h/NDISP
        for i in range(NDISP):
            c=PAIR[base][i%2]
            ax.add_patch(Rectangle((x,yb+i*sh), w, sh, fc=c, ec="#222", lw=0.5))
        ax.add_patch(Rectangle((x,yb), w, h, fill=False, ec="#222", lw=1.4))
    elif split=="col":
        sw=w/NDISP
        for i in range(NDISP):
            c=PAIR[base][i%2]
            ax.add_patch(Rectangle((x+i*sw,yb), sw, h, fc=c, ec="#222", lw=0.5))
        ax.add_patch(Rectangle((x,yb), w, h, fill=False, ec="#222", lw=1.4))
    ax.text(x+w/2, yb+h+0.12, name, ha="center", va="bottom", fontsize=9.5, weight="bold")
    ax.text(x+w/2, yb+h/2, dims, ha="center", va="center", fontsize=7.3,
            color="white", rotation=0 if w>0.5 else 90)
    if note: ax.text(x+w/2, yb-0.16, note, ha="center", va="top", fontsize=6.8, color="#444")
    return x+w

fig, ax = plt.subplots(figsize=(15.5, 10)); ax.axis("off")
ax.set_xlim(0,15.5); ax.set_ylim(0,10.3)
ax.set_title("Per-head structure of the weight matrices  (rows = heads for Q/K/V/UK/UV, "
             "columns = heads for W_O)\nd_model=4096, n_heads=32, head_dim=128, d_c=512   "
             "(8 slices drawn = 32 heads; W_DKV is ONE shared block)", fontsize=11.5)

# ---- Band 1: MHA ----
y1=7.3
ax.text(0.15,y1+1.0,"MHA\nstored", fontsize=11, weight="bold", color=DARK, va="center")
x=2.5
x=mat(ax,x,y1,d_model,d_model,"W_Q","4096×4096",GREY,split="row",note="32 head row-slices")+0.5
x=mat(ax,x,y1,d_model,d_model,"W_K","4096×4096",RED,split="row",note="per head")+0.5
x=mat(ax,x,y1,d_model,d_model,"W_V","4096×4096",RED,split="row",note="per head")+0.5
x=mat(ax,x,y1,d_model,d_model,"W_O","4096×4096",GREY,split="col",note="head col-slices")+0.4
ax.text(13.6,y1+1.0,"cache/tok\nK+V=8192",fontsize=8,color=RED,ha="center",va="center",
        bbox=dict(boxstyle="round",fc="#fdecea",ec=RED))

# ---- Band 2: MLA ----
y2=3.95
ax.text(0.15,y2+1.0,"MLA\nstored", fontsize=11, weight="bold", color=DARK, va="center")
x=2.5
x=mat(ax,x,y2,d_model,d_model,"W_Q","4096×4096",GREY,split="row",note="per head")+0.5
x=mat(ax,x,y2,d_model,d_c,"W_DKV","512×4096",GREEN,split=None,note="SHARED\n(one block, cached)")+0.5
x=mat(ax,x,y2,d_c,d_model,"W_UK","4096×512",BLUE,split="row",note="per head")+0.35
x=mat(ax,x,y2,d_c,d_model,"W_UV","4096×512",BLUE,split="row",note="per head")+0.55
x=mat(ax,x,y2,d_model,d_model,"W_O","4096×4096",GREY,split="col",note="per head")+0.4
ax.text(13.7,y2+1.0,"cache/tok\nc=512\n≈16× less",fontsize=8,color=GREEN,ha="center",va="center",
        bbox=dict(boxstyle="round",fc="#eafaf1",ec=GREEN))
ax.text(7.3,y2-0.7,"per head:  W_K^(h) = W_UK^(h) · W_DKV   "
        "(each head's own W_UK row-slice, but the SAME shared W_DKV)",fontsize=8.3,color=DARK,ha="center")

# ---- Band 3: absorbed equivalent ----
y3=0.7
ax.text(0.15,y3+1.05,"MLA absorbed\n(equivalent\ncompute)", fontsize=10.5, weight="bold",
        color=DARK, va="center")
# n_heads separate W_Q' bars stacked (NOT one square)
x0=2.5; barw=d_model*S; barh=d_c*S*0.5
for i in range(5):
    c=PAIR[ORANGE][i%2]
    ax.add_patch(Rectangle((x0, y3+0.05+i*(barh+0.05)), barw, barh, fc=c, ec="#222", lw=0.6))
ax.text(x0+barw/2, y3+0.05+5*(barh+0.05)+0.28, "W_Q'^(h) = W_UKᵀW_Q", ha="center", fontsize=9, weight="bold")
ax.text(x0+barw/2, y3+0.05+5*(barh+0.05)+0.02, "512×4096  ×n_heads (rank≤128)", ha="center", fontsize=7, color="#444")
ax.text(x0+barw/2, y3-0.15, "n_heads SEPARATE matrices\n(don't merge into a square)", ha="center", va="top", fontsize=6.8, color="#444")
x=x0+barw+0.7
x=mat(ax,x,y3+0.6,d_model,d_c,"W_DKV (=K/V)","512×4096",GREEN,split=None,note="shared latent c")+0.7
ax.text(x+0.1,y3+1.4,"→ each head dots its own\n   W_Q'^(h) against the ONE\n   shared c in d_c=512 space;\n   W_UV folds into W_O",
        fontsize=8,color=ORANGE,ha="left",va="center",
        bbox=dict(boxstyle="round",fc="#fef5e7",ec=ORANGE))

fig.tight_layout()
fig.savefig(f"{OUT}/mla_weight_dims_heads.png", dpi=130); print("saved")
