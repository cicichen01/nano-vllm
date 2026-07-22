import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT="/home/cicichen/nano-vllm/h100_setup"
GREY="#7f8c8d"; BLUE="#2c6fbb"; GREEN="#3a9d5d"; DARK="#2c3e50"

fig,ax=plt.subplots(figsize=(10,11)); ax.axis("off"); ax.set_xlim(0,10); ax.set_ylim(0,21)
def panel(y,h,color): ax.add_patch(FancyBboxPatch((0.4,y),9.2,h,
    boxstyle="round,pad=0.1,rounding_size=0.15",fc="#f7f9fb",ec=color,lw=1.6))

ax.text(5,20.4,"Per-token weight  $w_j$  on value  $v_j$",fontsize=16,weight="bold",ha="center")

panel(18.2,1.7,DARK)
ax.text(0.8,19.35,"common form",fontsize=12,weight="bold",color=DARK,va="center")
ax.text(5.4,19.45,r"$y_i \;=\; \sum_j\, w_j\, v_{j,i}$",fontsize=17,va="center",ha="center")
ax.text(5.4,18.6,r"(one scalar $w_j$ per token, shared over all dims $i$;   $s_j \equiv q\cdot k_j$)",
        fontsize=10,va="center",ha="center",color="#555")

panel(16.5,1.2,GREY)
ax.text(0.8,17.1,"linear",fontsize=12,weight="bold",color=GREY,va="center")
ax.text(5.4,17.1,r"$w_j \;=\; q\cdot k_j$",fontsize=17,va="center",ha="center")

panel(13.9,2.2,BLUE)
ax.text(0.8,15.0,"softmax",fontsize=12,weight="bold",color=BLUE,va="center")
ax.text(5.4,15.0,r"$w_j \;=\; \frac{e^{\,q\cdot k_j/\sqrt{d_k}}}{\sum_l e^{\,q\cdot k_l/\sqrt{d_k}}}$",
        fontsize=17,va="center",ha="center")

# delta
panel(6.0,7.1,GREEN)
ax.text(0.8,12.55,"DeltaNet",fontsize=12,weight="bold",color=GREEN,va="center")
ax.text(5.4,12.65,r"$w \;=\; [\,q\cdot k_1,\ \dots,\ q\cdot k_M\,]\;(I+L)^{-1}$  (vector form)",
        fontsize=15,va="center",ha="center")
ax.text(5.4,11.75,r"per element:   $w_j \;=\; \sum_l (q\cdot k_l)\,[(I+L)^{-1}]_{lj}"
        r"\;=\; (q\cdot k_j) + \sum_{l>j}(q\cdot k_l)\,[(I+L)^{-1}]_{lj}$",
        fontsize=12.5,va="center",ha="center",color="#1e5631")
ax.text(5.4,10.9,r"$L_{ji} = k_j\cdot k_i\ \ \mathrm{if}\ i<j,\quad 0\ \ \mathrm{if}\ i\geq j$"
        r"$\quad$(strictly-lower-tri of $KK^{T}$)",fontsize=11.5,va="center",ha="center",color="#333")
rows=[["1","0","0","0"],
      [r"$k_2\!\cdot\!k_1$","1","0","0"],
      [r"$k_3\!\cdot\!k_1$",r"$k_3\!\cdot\!k_2$","1","0"],
      [r"$k_4\!\cdot\!k_1$",r"$k_4\!\cdot\!k_2$",r"$k_4\!\cdot\!k_3$","1"]]
cx=[3.15,4.6,6.05,7.5]; ry=[9.85,9.3,8.75,8.2]
ax.text(1.6,9.0,r"$I+L\;=$",fontsize=14,va="center")
for r,yy in zip(rows,ry):
    for e,xx in zip(r,cx): ax.text(xx,yy,e,fontsize=11,va="center",ha="center")
x0,x1,ytop,ybot=2.65,8.05,10.15,7.9
for xb,d in [(x0,0.18),(x1,-0.18)]:
    ax.plot([xb,xb],[ybot,ytop],color=DARK,lw=1.6)
    ax.plot([xb,xb+d],[ytop,ytop],color=DARK,lw=1.6)
    ax.plot([xb,xb+d],[ybot,ybot],color=DARK,lw=1.6)
ax.text(5.35,7.45,"(shown for M=4; lower-tri entries are the key dot-products)",
        fontsize=9.5,ha="center",color="#555")
ax.text(5.35,6.7,r"$(I+L)^{-1}$ is lower-tri, unit diagonal  →  $w_j$ = own score minus "
        "corrections from LATER overlapping keys",fontsize=10,ha="center",color="#1e5631")

panel(3.1,2.5,GREEN)
ax.text(0.8,5.2,"equivalent",fontsize=11,weight="bold",color=GREEN,va="center")
ax.text(5.4,5.15,r"$y=\sum_j (q\cdot k_j)\,e_j,\qquad e_j = v_j - \sum_{i<j}(k_j\cdot k_i)\,e_i$",
        fontsize=14,va="center",ha="center")
ax.text(5.4,4.0,r"$(I+L)^{-1}$ decorrelates the scores: subtracts the overlap with earlier keys.",
        fontsize=10.5,va="center",ha="center",color="#333")

ax.text(5,2.3,"linear: raw similarity   |   softmax: exponential, positive, normalized   |   "
        "delta: similarity decorrelated by $(I+L)^{-1}$",fontsize=10,ha="center",color=DARK,style="italic")
fig.savefig(f"{OUT}/attention_weight_formulas.png",dpi=130,bbox_inches="tight"); print("saved")
