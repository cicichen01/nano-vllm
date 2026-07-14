"""Generate a RoPE explainer diagram (frequency ladder + Goldilocks zones)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

base = 10000.0
d = 128                      # head dim
n_pairs = d // 2             # 64 pairs
i = np.arange(n_pairs)
theta = base ** (-2.0 * i / d)     # per-pair angular frequency (rad/token)
wavelength = 2 * np.pi / theta     # tokens per full rotation

fig = plt.figure(figsize=(13, 9))
fig.suptitle("RoPE: a geometric ladder of frequencies covers every scale of distance",
             fontsize=15, fontweight="bold")

# ---------------------------------------------------------------- panel 1
# The frequency ladder: wavelength per pair (log y), geometric spacing.
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(i, wavelength, c=i, cmap="viridis", s=28, zorder=3)
ax1.set_yscale("log")
ax1.set_xlabel("pair index  i   (0 = fast … 63 = slow)")
ax1.set_ylabel("wavelength  2π/θᵢ   (tokens/rotation)")
ax1.set_title("1. The ladder: θᵢ = base^(−2i/d)\n"
              "geometric → evenly spaced in log", fontsize=11)
ax1.grid(True, which="both", alpha=0.25)
ax1.axhline(1, color="grey", ls=":", lw=0.8)
ax1.annotate("fast hands\n(local, ~few tokens)", (2, wavelength[2]),
             xytext=(8, 40), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="tab:red"))
ax1.annotate("slow hands\n(global, ~10k tokens)", (61, wavelength[61]),
             xytext=(20, 8000), fontsize=8,
             arrowprops=dict(arrowstyle="->", color="tab:blue"))

# ---------------------------------------------------------------- panel 2
# Why geometric beats linear: coverage of scales.
ax2 = fig.add_subplot(2, 2, 2)
lin_wl = np.linspace(wavelength.min(), wavelength.max(), n_pairs)
ax2.scatter(np.log10(wavelength), np.ones(n_pairs), s=18,
            c="tab:green", label="geometric (RoPE)")
ax2.scatter(np.log10(lin_wl), np.zeros(n_pairs), s=18,
            c="tab:orange", label="linear (bad)")
for scale, name in [(1, "10"), (2, "100"), (3, "1k"), (4, "10k")]:
    ax2.axvline(scale, color="grey", ls=":", lw=0.7)
    ax2.text(scale, 1.35, name, ha="center", fontsize=8, color="grey")
ax2.set_yticks([0, 1]); ax2.set_yticklabels(["linear", "geometric"])
ax2.set_ylim(-0.6, 1.7)
ax2.set_xlabel("log₁₀(wavelength)  →  scale of distance")
ax2.set_title("2. Coverage of scales\n"
              "linear clumps at long range, leaves short-range gaps", fontsize=11)
ax2.legend(loc="lower right", fontsize=8)
ax2.grid(True, axis="x", alpha=0.25)

# ---------------------------------------------------------------- panel 3
# Goldilocks zone: phase difference vs distance for 3 hands.
ax3 = fig.add_subplot(2, 2, 3)
dist = np.arange(0, 400)
for idx, color, name in [(3, "tab:red", "fast  (λ≈%d)" % wavelength[3]),
                         (20, "tab:green", "mid   (λ≈%d)" % wavelength[20]),
                         (40, "tab:blue", "slow  (λ≈%d)" % wavelength[40])]:
    phase = np.degrees(dist * theta[idx]) % 360
    ax3.plot(dist, phase, color=color, lw=1.4, label=name)
ax3.set_xlabel("distance  Δ = m − n   (tokens)")
ax3.set_ylabel("phase difference  Δ·θᵢ  (deg, mod 360)")
ax3.set_title("3. Goldilocks zones\n"
              "fast wraps early (aliased); slow barely moves (invisible)", fontsize=11)
ax3.set_yticks([0, 90, 180, 270, 360])
ax3.legend(fontsize=8, loc="upper right")
ax3.grid(True, alpha=0.25)

# ---------------------------------------------------------------- panel 4
# Roles per distance: for a fixed Δ, which bands are content / signal / washed.
ax4 = fig.add_subplot(2, 2, 4)
Delta = 60
rot = np.degrees(Delta * theta) % 360          # rotation each pair sees at Δ=60
# classify: ~unrotated (content), resonant (signal), wrapped (washed)
role = np.where(wavelength > 6 * Delta, 0,      # slow: content-matching
        np.where(wavelength > 0.5 * Delta, 1,   # resonant-ish: distance signal
                 2))                            # fast: washed
colors = np.array(["tab:blue", "tab:green", "tab:red"])[role]
ax4.scatter(i, wavelength, c=colors, s=28, zorder=3)
ax4.axhline(Delta, color="black", ls="--", lw=1,
            label="Δ = %d tokens" % Delta)
ax4.set_yscale("log")
ax4.set_xlabel("pair index  i")
ax4.set_ylabel("wavelength (tokens)")
ax4.set_title("4. Roles at a fixed distance (Δ=60)\n"
              "every band does a job — none wasted", fontsize=11)
ax4.grid(True, which="both", alpha=0.25)
legend_el = [Patch(color="tab:blue", label="slow → pure content match"),
             Patch(color="tab:green", label="resonant → distance signal"),
             Patch(color="tab:red", label="fast → washed out")]
ax4.legend(handles=legend_el + [plt.Line2D([], [], color="black", ls="--",
           label="Δ = 60")], fontsize=7.5, loc="lower left")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = "note/rope_diagram.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
