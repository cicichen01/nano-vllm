"""Why batching decode fills the memory bandwidth: warp-level latency hiding.
Timeline (Gantt) of warps: orange = waiting on an HBM KV-read (latency), green = computing.
Below each: how many KV reads are 'in flight' vs HBM capacity.
1 sequence (few warps) -> bandwidth idle.  8 sequences (many warps) -> bandwidth saturated."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

L, C, T, CAP = 7, 2, 40, 8          # load-latency, compute-time, timeline, HBM concurrent-request capacity
ORANGE, GREEN, RED = "#E4A04A", "#4FA06B", "#C0392B"

def segs(off):
    t, out = off, []
    while t < T:
        out.append((t, min(L, T-t), "load")); t += L
        if t >= T: break
        out.append((t, min(C, T-t), "compute")); t += C
    return out

def curve(all_segs, ts, kind):
    return np.array([sum(1 for s in all_segs for (t, d, k) in s if k == kind and t <= x < t+d) for x in ts])

def draw(ax, offsets, title, sub):
    ax.set_xlim(-7, T+9); ax.set_ylim(0, 13); ax.axis("off")
    ax.text(-7, 12.4, title, fontsize=13, fontweight="bold", color="#1A1A1A")
    ax.text(-7, 11.4, sub, fontsize=9.5, color="#444")
    all_segs = [segs(o) for o in offsets]
    lane_top = 11.0
    for i, s in enumerate(all_segs):
        y = lane_top - i*0.8
        ax.text(-1.2, y+0.28, f"warp{i}", ha="right", va="center", fontsize=7.5, color="#666")
        for (t, d, k) in s:
            ax.add_patch(Rectangle((t, y), d, 0.55, facecolor=(ORANGE if k == "load" else GREEN), edgecolor="white", lw=0.4))
    ts = np.linspace(0, T, 500)
    inf = curve(all_segs, ts, "load")
    base, scale = 0.5, 3.0/CAP
    ax.plot(ts, base+inf*scale, color=RED, lw=1.6)
    ax.fill_between(ts, base, base+inf*scale, color=RED, alpha=0.15)
    ax.plot([0, T], [base+CAP*scale]*2, ls="--", color="#999", lw=1)
    ax.text(T+0.5, base+CAP*scale, "HBM bandwidth\ncapacity", fontsize=7.5, color="#999", va="center")
    ax.text(-1.2, base+1.8, "KV reads\nin flight", ha="right", va="center", fontsize=8, color=RED)
    memutil = 100*inf.mean()/CAP
    ax.text(T+0.5, 8.6, f"memory\nbandwidth\nused ≈ {memutil:.0f}%", fontsize=11, color=RED, va="center", fontweight="bold")
    return memutil

fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9.5))
m1 = draw(a1, [0, 1.5],
          "1 sequence  →  few warps  →  memory bandwidth UNDER-utilized",
          "only 1-2 KV reads ever in flight; the SM mostly WAITS on HBM (orange) — HBM sits idle, work drips out slowly")
m2 = draw(a2, [i*(L+C)/8 for i in range(8)],
          "8 sequences batched  →  many warps  →  memory bandwidth SATURATED",
          "~6-8 KV reads always in flight; whenever one warp waits on HBM, another is ready to compute → HBM stays busy, same total work finishes far sooner")
fig.legend(handles=[Rectangle((0, 0), 1, 1, fc=ORANGE), Rectangle((0, 0), 1, 1, fc=GREEN)],
           labels=["warp waiting on an HBM KV-read (long latency)", "warp computing"],
           loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=2, fontsize=9.5, frameon=False)
fig.text(0.5, 0.012, "Same per-seq work either way (memory-bound) — batching doesn't make it compute-bound; it fills the memory pipeline so you reach PEAK bandwidth.",
         ha="center", fontsize=9.5, color="#333", style="italic")
plt.tight_layout(rect=[0, 0.09, 1, 1])
out = "/home/cicichen/nano-vllm/h100_setup/bandwidth_util.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out, "util:", round(m1), round(m2))
