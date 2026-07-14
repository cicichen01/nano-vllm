"""End-to-end: one generate() call flowing through the components (sequence diagram)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

lanes = [("LLMEngine\n(generate / step)", 8, "#607D8B"),
         ("Scheduler", 26, "#4C86C6"),
         ("BlockManager", 44, "#4FA06B"),
         ("ModelRunner", 62, "#E0803B"),
         ("GPU\n(model + graph)", 80, "#C0392B"),
         ("Sampler", 96, "#7E57C2")]
X = {n.split("\n")[0]: x for n, x, _ in lanes}

fig, ax = plt.subplots(figsize=(15.5, 13))
ax.set_xlim(0, 104); ax.set_ylim(0, 100); ax.axis("off")
ax.text(52, 98, "End-to-end: one generate() call through the components", ha="center", fontsize=14, fontweight="bold")

for name, x, c in lanes:
    ax.add_patch(FancyBboxPatch((x-8, 90), 16, 5, boxstyle="round,pad=0.1,rounding_size=0.4", lw=1.3, ec="#444", fc=c))
    ax.text(x, 92.5, name, ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
    ax.plot([x, x], [8, 90], color="#CCC", lw=1.1, zorder=0)

def msg(a, b, y, label, ret=False):
    x1, x2 = X[a], X[b]
    col = "#888" if ret else "#333"
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=12, lw=1.3,
                                 color=col, linestyle=("--" if ret else "-"), zorder=3))
    ax.text((x1+x2)/2, y+0.9, label, ha="center", va="bottom", fontsize=7.3,
            color=col, style=("italic" if ret else "normal"))

def selfbox(a, y, label):
    x = X[a]
    ax.add_patch(FancyBboxPatch((x-7.5, y-1.6), 15, 3.2, boxstyle="round,pad=0.05,rounding_size=0.3",
                                lw=1.0, ec="#777", fc="#F3F3F3", zorder=3))
    ax.text(x, y, label, ha="center", va="center", fontsize=6.9, color="#222")

def phase(y, label):
    ax.text(0.5, y, label, ha="left", va="center", fontsize=9, fontweight="bold", color="#B00020", rotation=0)

# SUBMIT
phase(86.5, "① SUBMIT")
msg("LLMEngine", "Scheduler", 85, "tokenize → add_request(seq) → waiting deque")

# STEP LOOP box
ax.add_patch(FancyBboxPatch((3.5, 24), 99, 56, boxstyle="round,pad=0.2,rounding_size=1.0",
                            lw=1.6, ec="#B00020", fc="none", linestyle="--", zorder=1))
ax.text(4.5, 78.5, "② STEP LOOP  — repeat until waiting & running are empty", fontsize=9.5, fontweight="bold", color="#B00020")

msg("LLMEngine", "Scheduler", 75, "schedule()")
msg("Scheduler", "BlockManager", 71.5, "prefill: can_allocate/allocate (prefix hit)  |  decode: can_append/may_append (+preempt)")
msg("BlockManager", "Scheduler", 68, "block_table / #cached blocks", ret=True)
msg("Scheduler", "LLMEngine", 64.5, "(seqs, is_prefill)", ret=True)
msg("LLMEngine", "ModelRunner", 61, "run(seqs, is_prefill)")
selfbox("ModelRunner", 57, "prepare_prefill/decode\nbuild tensors + H2D (pinned)")
msg("ModelRunner", "GPU", 52.5, "forward: embed → 28× layer → norm   (eager | graph.replay)")
selfbox("GPU", 48, "hidden → compute_logits\n(lm_head: last-tok slice / gather)")
msg("GPU", "Sampler", 43.5, "logits")
selfbox("Sampler", 39, "temperature → softmax\n→ Gumbel-argmax")
msg("Sampler", "ModelRunner", 34.5, "token_ids  (.tolist = D2H sync)", ret=True)
msg("ModelRunner", "LLMEngine", 31, "token_ids", ret=True)
msg("LLMEngine", "Scheduler", 27.5, "postprocess(seqs, token_ids)")
msg("Scheduler", "BlockManager", 24.5, "hash_blocks (register) / deallocate(finished)")

# RETURN
phase(18, "③ RETURN")
selfbox("LLMEngine", 14, "detokenize completion_token_ids\n→ [{text, token_ids}]")

# footer legend
ax.text(52, 6, "solid = call · dashed = return.   prefill: eager, flat varlen, batches many seqs · "
        "decode: CUDA-graph, 1 tok/seq · finished seqs' blocks freed & hashed for prefix reuse.",
        ha="center", fontsize=8, color="#555", style="italic")

plt.tight_layout()
out = "/home/cicichen/nano-vllm/h100_setup/end_to_end.png"
plt.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
print("saved", out)
