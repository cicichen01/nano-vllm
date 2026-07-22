import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT="/home/cicichen/nano-vllm/h100_setup"
GREY="#7f8c8d"; BLUE="#2c6fbb"; GREEN="#3a9d5d"; RED="#c0392b"

# 3-key example: k3 overlaps k1,k2; query q=k3
K=np.array([[1.,0,0],[0,1,0],[0.6,0.8,0]]); q=K[2]
s=K@q                                   # raw scores q·kj = [0.6,0.8,1.0]
lin=s.copy()
e=np.exp(s-s.max()); soft=e/e.sum()
L=np.tril(K@K.T,-1); dw=s@np.linalg.inv(np.eye(3)+L)   # delta weights

cases=[("Linear\n$w_j = q\\cdot k_j$", lin, GREY),
       ("Softmax\n$w_j = e^{q\\cdot k_j}/\\sum_l e^{q\\cdot k_l}$", soft, BLUE),
       ("DeltaNet\n$w = [q\\cdot k_j]\\,(I+L)^{-1}$", dw, GREEN)]
labels=["$v_1$\n(k1)","$v_2$\n(k2)","$v_3$\n(k3=q)"]

fig,axes=plt.subplots(1,3,figsize=(12,4.6),sharey=True)
fig.suptitle("Per-token weight $w_j$ on value $v_j$   (query $q=k_3$;  raw scores "
             "$q\\cdot k_j$ = [0.6, 0.8, 1.0];  $k_3$ overlaps $k_1,k_2$)", fontsize=11)
for ax,(title,w,c) in zip(axes,cases):
    bars=ax.bar(range(3), w, color=[c,c,RED], edgecolor="#222", width=0.6)
    bars[2].set_color(c); bars[2].set_edgecolor(RED); bars[2].set_linewidth(2.5)  # target v3
    ax.axhline(0,color="#333",lw=0.8)
    ax.set_title(title,fontsize=10.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels,fontsize=9)
    for i,val in enumerate(w):
        ax.text(i, val+(0.03 if val>=0 else -0.06), f"{val:.2f}", ha="center",
                va="bottom" if val>=0 else "top", fontsize=9, weight="bold")
    ax.grid(axis="y",alpha=0.3)
axes[0].set_ylabel("weight $w_j$")
axes[0].text(1,0.5,"leaks v1,v2\n(interference)",ha="center",fontsize=8,color=RED)
axes[1].text(1,0.5,"diffuse\n(all positive)",ha="center",fontsize=8,color=RED)
axes[2].text(1,0.5,"clean: only v3\n(overlap subtracted)",ha="center",fontsize=8,color=GREEN)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(f"{OUT}/attention_weights.png",dpi=130); print("saved")
