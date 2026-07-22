import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT="/home/cicichen/nano-vllm/h100_setup"
DARK="#2c3e50"; GREEN="#1e5631"

fig,ax=plt.subplots(figsize=(13.5,6.2)); ax.axis("off"); ax.set_xlim(0,16); ax.set_ylim(0,10)
ax.text(8,9.4,r"$(I+L)^{-1}$  —  depends ONLY on the keys $k$  (shown for $M=4$)",
        fontsize=16,weight="bold",ha="center")
ax.text(8,8.55,r"$(I+L)^{-1} = I - L + L^2 - L^3 + \cdots = \sum_{n\geq0}(-L)^n$   "
        r"(finite: $L$ strictly-lower-tri $\Rightarrow$ nilpotent).   Each below-diagonal entry = "
        r"alternating sums of products of $k_j\!\cdot\!k_i$ along paths $i\to\cdots\to j$.",
        fontsize=10.5,ha="center",color="#444")

ax.text(1.35,4.7,r"$(I+L)^{-1}=$",fontsize=15,va="center")
# columns
c=[5.0,10.0,12.7,14.4]; r=[6.6,5.6,4.6,3.3]
short=11; long=9.3
def T(x,y,s,fs=short): ax.text(x,y,s,fontsize=fs,va="center",ha="center")
# row 1
T(c[0],r[0],"1"); T(c[1],r[0],"0"); T(c[2],r[0],"0"); T(c[3],r[0],"0")
# row 2
T(c[0],r[1],r"$-\,k_2\!\cdot\!k_1$"); T(c[1],r[1],"1"); T(c[2],r[1],"0"); T(c[3],r[1],"0")
# row 3
T(c[0],r[2],r"$-\,k_3\!\cdot\!k_1 + (k_3\!\cdot\!k_2)(k_2\!\cdot\!k_1)$",long)
T(c[1],r[2],r"$-\,k_3\!\cdot\!k_2$"); T(c[2],r[2],"1"); T(c[3],r[2],"0")
# row 4 (col1 spans two lines)
T(c[0],r[3]+0.32,r"$-\,k_4\!\cdot\!k_1 + (k_4\!\cdot\!k_2)(k_2\!\cdot\!k_1) + (k_4\!\cdot\!k_3)(k_3\!\cdot\!k_1)$",long)
T(c[0],r[3]-0.32,r"$-\,(k_4\!\cdot\!k_3)(k_3\!\cdot\!k_2)(k_2\!\cdot\!k_1)$",long)
T(c[1],r[3],r"$-\,k_4\!\cdot\!k_2 + (k_4\!\cdot\!k_3)(k_3\!\cdot\!k_2)$",long)
T(c[2],r[3],r"$-\,k_4\!\cdot\!k_3$"); T(c[3],r[3],"1")
# brackets
xl,xr,yt,yb=1.9,15.4,7.0,2.7
for xb,d in [(xl,0.22),(xr,-0.22)]:
    ax.plot([xb,xb],[yb,yt],color=DARK,lw=1.8)
    ax.plot([xb,xb+d],[yt,yt],color=DARK,lw=1.8)
    ax.plot([xb,xb+d],[yb,yb],color=DARK,lw=1.8)

ax.text(8,1.9,r"Diagonal = 1 (the linear term $I$).  Off-diagonal $(j,i)$ = $-\,k_j\!\cdot\!k_i$ "
        r"(first-order $-L$) plus higher-order chained overlaps ($+L^2-L^3\dots$).",
        fontsize=10.5,ha="center",color=GREEN)
ax.text(8,1.2,r"If all keys are orthogonal ($k_j\!\cdot\!k_i=0$): $(I+L)^{-1}=I$  $\Rightarrow$  "
        r"DeltaNet $=$ linear attention.",fontsize=10.5,ha="center",color=GREEN,weight="bold")
fig.savefig(f"{OUT}/deltanet_inverse_matrix.png",dpi=140,bbox_inches="tight"); print("saved")
