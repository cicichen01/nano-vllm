# Note 6 — Step 6: The model & layers (`models/qwen3.py` + `layers/`)

Where everything composes into an actual model. Most machinery (RMSNorm, RoPE, attention kernels,
TP column/row, `store_kvcache`, Context, Sampler) was dissected earlier (notes 5, gpu_attention,
torch_compile). This note is the **wiring** + the **new details**.

## Model hierarchy (top-down)
```
Qwen3ForCausalLM
 ├─ model: Qwen3Model
 │    ├─ embed_tokens: VocabParallelEmbedding
 │    ├─ layers: N × Qwen3DecoderLayer
 │    │      ├─ input_layernorm: RMSNorm
 │    │      ├─ self_attn: Qwen3Attention
 │    │      │     ├─ qkv_proj: QKVParallelLinear      (col-parallel, fused Q,K,V)
 │    │      │     ├─ q_norm / k_norm: RMSNorm         (Qwen3 QK-norm, over head_dim)
 │    │      │     ├─ rotary_emb: RotaryEmbedding
 │    │      │     ├─ attn: Attention                  (store_kvcache + flash-attn)
 │    │      │     └─ o_proj: RowParallelLinear        (all_reduce)
 │    │      ├─ post_attention_layernorm: RMSNorm
 │    │      └─ mlp: Qwen3MLP  (gate_up_proj → SiluAndMul → down_proj)
 │    └─ norm: RMSNorm (final)
 └─ lm_head: ParallelLMHead
```
Standard **pre-norm decoder-only transformer** + Qwen3 **QK-norm**. `forward` = `model(...)` → hidden;
`compute_logits` = `lm_head(hidden)` (called separately, OUTSIDE the CUDA graph).

## ★ The fused residual stream (`Qwen3DecoderLayer.forward`) — the subtle bit
Standard pre-norm is `x = x + attn(norm(x)); x = x + mlp(norm(x))`. nano-vllm **defers each residual add
and fuses it into the NEXT norm** via `RMSNorm.add_rms_forward`:
```python
if residual is None:                              # first layer
    hidden, residual = input_layernorm(hidden), hidden      # residual = un-normed input; hidden = norm(x)
else:
    hidden, residual = input_layernorm(hidden, residual)    # add_rms: residual += hidden (prev mlp's add); hidden = norm(residual)
hidden = self_attn(positions, hidden)                       # attn on normed
hidden, residual = post_attention_layernorm(hidden, residual)  # add_rms: residual += hidden (attn's add); hidden = norm(residual)
hidden = mlp(hidden)                                        # mlp on normed
return hidden, residual                                    # mlp's add DEFERRED to next layer's input_layernorm
```
- A **`residual` tensor is threaded through all layers** = the running residual stream. Each `add_rms_forward`
  does **"add previous sublayer output into residual, then normalize" in one fused kernel** (the
  `triton_..._add_mean_mul_pow_rsqrt` — the `add` fragment IS this residual add).
- `Qwen3Model.forward` finishes the last mlp's add with `self.norm(hidden, residual)`.
- **Payoffs:** (1) fewer kernels (add welded into norm); (2) **fp32 for the norm reduction** (variance/
  rsqrt computed on the fp32 sum), **residual accumulated in bf16** (`residual = x.to(orig_dtype)` — the
  stream is stored back in bf16 each layer; the fp32 is transient, its real benefit is the accurate
  variance, not the residual precision — quality choice, not required).

## Linear family + TP weight loading (recap + new part)
Concept (col = split output → shard/all_gather; row = split contraction → all_reduce) covered earlier.
**New: `weight_loader`** — every param carries a `weight_loader`; `load_model` (loader.py) reads each HF
tensor and calls it, which **narrows to this rank's TP slice** and copies in.
- `ColumnParallelLinear` (tp_dim=0): forward = `F.linear` (output is a shard, no comm).
- `RowParallelLinear` (tp_dim=1): forward = `F.linear` then **`all_reduce`** (partial sums); bias only on rank 0.
- **Fused projections** `QKVParallelLinear` / `MergedColumnParallelLinear`: HF stores the pieces
  **separately** (`q/k/v_proj`, `gate/up_proj`); the bridge is `packed_modules_mapping`:
  ```python
  "q_proj":("qkv_proj","q"), "k_proj":("qkv_proj","k"), "v_proj":("qkv_proj","v"),
  "gate_proj":("gate_up_proj",0), "up_proj":("gate_up_proj",1)
  ```
  So `...q_proj.weight` routes to `qkv_proj`'s loader with `shard_id="q"`, which writes it at the correct
  **offset** in the fused weight (and slices to this rank's heads). Three HF matrices → one `qkv_proj` GEMM.
  QKV offsets handle **GQA** (Q size ≠ K/V size): q at 0, k at `num_heads*head_dim`, v after that.

## embed_head — two tricks
- **`VocabParallelEmbedding`**: vocab split across ranks; each id belongs to one rank's slice → forward
  **masks** ids outside `[vocab_start,end)`, looks up, **zeros** non-owned rows, **all_reduce(sum)** → each
  token gets its embedding from the owning rank.
- **`ParallelLMHead`** (prefill optimization): for prefill computes logits **only for the last token of each
  seq** (`x[cu_seqlens_q[1:]-1]`) — next-token prediction needs only the last position → avoids a giant
  `[all_prefill_tokens × vocab]` matmul. Then `F.linear` + **gather** the vocab-sharded logits to rank 0.

## Smaller pieces (in context)
- `Qwen3Attention.forward`: `qkv_proj → split(q,k,v) → view per-head → q_norm/k_norm → rope → attn → o_proj(flatten)`.
  q_norm/k_norm applied per-head over `head_dim` (only when `not qkv_bias`).
- `Qwen3MLP` / `SiluAndMul`: `gate_up = gate_up_proj(x); chunk into (gate,up); silu(gate)*up; down_proj`. = SwiGLU.
- `Attention.forward`: `store_kvcache` (Triton, skips slot==-1) then prefill `flash_attn_varlen_func`
  (block_table only on prefix-cache hit) / decode `flash_attn_with_kvcache`, via global Context.
- `tie_word_embeddings`: Qwen3-0.6B ties `lm_head.weight = embed_tokens.weight`.

## The three things worth a close look
1. **Residual stream** (deferred-add fused into next norm; fp32 accumulation).
2. **`weight_loader` + `packed_modules_mapping`** (how separate HF q/k/v/gate/up land in fused, TP-sharded params).
3. **`ParallelLMHead` last-token trick** (prefill computes logits for last position only).
Everything else was already seen at the kernel level in earlier notes.

## ★★ Pre-norm vs post-norm (why 2 norms/layer, and why pre-norm)
Figures: `residual_stream.png`, `prenorm_vs_postnorm.png`, `grad_prenorm_postnorm.png`, `grad_decay.png`.

**Two norms per layer = one PRE-norm per sub-block, NOT "before+after attention":**
- `input_layernorm` = pre-norm for **attention**; `post_attention_layernorm` = pre-norm for the **MLP**
  (named by *position*, it's the MLP's input norm). Plus `q_norm`/`k_norm` = Qwen3 QK-norm *inside*
  attention (separate purpose), and a final `model.norm` before the LM head.
- Each sub-block needs a normalized input because the **residual stream grows with depth** — re-normalize
  before every consumer so attention/MLP always see well-scaled activations.

**Pre-norm vs post-norm (a MODEL-architecture choice, lives in `DecoderLayer.forward`, not the engine):**
- pre-norm (Qwen3/Llama): `x = x + Sub(Norm(x))` — norm on the **side branch**; residual is a **clean identity line**.
- post-norm (2017 Transformer): `x = Norm(x + Sub(x))` — norm **on the residual path** (wraps the add).
- Same op *count* (1 add + 1 norm/sub-block) — the difference is **placement relative to the skip**.

**The subtle equivalence + where they diverge:** at the attn→MLP boundary BOTH compute `Norm(r + attn_out)`
as the MLP input — identical there. They diverge at the **MLP's residual add** and what continues:
- pre-norm: `Norm(r+attn)` is a **throwaway branch** (feeds MLP only); the stream carries the **un-normed**
  `r1=r+attn`; MLP add = `r1 + mlp_out` (adds to un-normed). → a **raw un-normed copy of the stream survives**.
- post-norm: `Norm(r+attn)` **IS** the stream; MLP add = `Norm(x1 + mlp_out)` (adds to normed, re-norms). → no raw survivor.
- Count norms on the residual path itself: **pre-norm 0**, **post-norm 2/layer**.

**Why it matters — gradients (backprop through L layers; both are PRODUCTS of L factors):**
- Symbols: `J_i` = pre-norm sublayer-path Jacobian `∂[Sub(Norm(r_i))]/∂r_i` (includes norm, small);
  `S_i` = post-norm sublayer Jacobian; `N_i = ∂Norm/∂(·)` (NOT identity; rescales/rank-deficient); `I` = raw skip.
- **pre-norm:** `∂r_{i+1}/∂r_i = I + J_i` → `∂r_L/∂r_0 = ∏(I+J_i) = I + ΣJ_i + Σ J_jJ_i + …`
  → has a **bare I** (route through ZERO norms) → clean gradient highway; `≈ I+ΣJ_i` when `J_i` small.
- **post-norm:** `∂r_{i+1}/∂r_i = N_i(I+S_i)` → `∂r_L/∂r_0 = ∏ N_i(I+S_i)`; smallest term `= ∏N_i` (NO bare I)
  → **L norm-Jacobians compound** → vanish/explode → needs LR warmup + careful init.
- The residual `+r` is what puts the additive `I` **inside every factor**; post-norm's outer Norm strips it.

**Measured (`grad_demo.py`/`grad_demo2.py`):**
- Real L=64 nets, unit grad at output → |grad@input|, ratio pre/post: gain1 **6×**, gain2 9×, gain3 30×, gain4 **34×**
  (pre GROWS with depth/gain — accumulates on the identity highway; post throttled ~O(1) or below).
- Controlled toy `∏(I+J)` vs `∏0.9(I+S)`: at L=160 pre=**5088** vs post=**2.4e-4** (post vanishes as `0.9^L`).
  (Real RMSNorm resets rms≈1/layer so `N≈I` at benign init → post doesn't dramatically vanish there, just
  fails to accumulate; the toy's `N=0.9` exposes the mechanism when rms>1 makes the norm contract.)

**Engine impact:** swapping to a post-norm model = write a new `DecoderLayer.forward` in a **new model file**
(and drop the fused deferred-add — post-norm can still fuse its own add+norm, but there's no un-normed
residual stream to defer); the scheduler/paging/runner/kernels are untouched.

## ★★ Weight loading mechanics (`utils/loader.py` + `weight_loader`)
**`weight_loader` is a vLLM convention, NOT a PyTorch feature** — a function **stapled as an attribute onto each
`nn.Parameter`** (`self.weight.weight_loader = self.weight_loader` in `LinearBase`), invoked by a **custom load
loop** that replaces PyTorch's native `load_state_dict`.

**Module <-> Parameter <-> weight_loader (the reference loop):**
- Module **owns/registers** the Parameter (`module.weight` is the tensor in HBM).
- Parameter **carries** `weight_loader` as a plain attribute.
- `weight_loader` is the Module's **bound method** -> closes over `self` so it can read TP config
  (`tp_rank`, `tp_size`, `num_heads`, `output_sizes`, ...). So: Module->Param (owns), Param->loader (attr), loader->Module (method).

**Dotted names = the module-tree path.** A param's name (`model.layers.0.self_attn.q_proj.weight`) is the
concatenation of **attribute names** down the tree (`nn.ModuleList` -> numeric index `layers.0`). `get_parameter(name)`
splits on `.` and walks `getattr` -> the SAME object as attribute access. Checkpoint = `{name_string: tensor}` (pure
data); loading = get the right numbers into the right param buffer so `forward` computes the intended function.

**Two ways to load a checkpoint (the fundamental choice):**
- **Option 1 (HF):** define the Module to **match the checkpoint 1:1** (names+shapes) -> `model.load_state_dict()` just works.
- **Option 2 (nano-vllm):** define the model **fused + TP-sharded** for speed -> write a **translation** loader
  (rename/offset/slice/reshape) that must be **math-equivalent** (fuse q/k/v = concat; forward splits it back).
  (Variants: convert the ckpt offline; or PyTorch `_load_state_dict_pre_hook`.)

**The load loop (`load_model`) is CHECKPOINT-driven:**
```python
for weight_name in f.keys():                       # e.g. "...self_attn.q_proj.weight"
    for k in packed_modules_mapping:               # k = "q_proj","k_proj","v_proj","gate_proj","up_proj"
        if k in weight_name:                        # SUBSTRING test (fragment in full path) -> covers ALL layers
            v, shard_id = packed_modules_mapping[k] # ("qkv_proj","q")
            param = model.get_parameter(weight_name.replace(k, v))   # rename fragment -> fused param
            param.weight_loader(param, f.get_tensor(weight_name), shard_id)   # WITH shard_id
            break
    else:                                           # no fused match
        param = model.get_parameter(weight_name)
        getattr(param, "weight_loader", default_weight_loader)(param, f.get_tensor(weight_name))  # NO shard_id
```
- **Fused q/k/v hit the loop 3x separately** (checkpoint has 3 tensors); each renamed `*_proj->qkv_proj` and written
  into its **row-slice** of the ONE fused param (`q` 0:2048, `k` 2048:3072, `v` 3072:4096 for 0.6B). Then forward =
  one GEMM + `split`. Same for gate/up -> `gate_up_proj`.
- **`k in weight_name` = substring** (mapping keys are short fragments; checkpoint names are full dotted paths) -> one
  entry matches that fragment in **every** layer, index-agnostic. `.replace(k,v)` swaps just the fragment.

**Why the two branches never mismatch the loader signature (design invariant):**
- The **`shard_id` branch** is reached ONLY for `packed_modules_mapping` targets = the **fused** modules `qkv_proj`
  (`QKVParallelLinear`) / `gate_up_proj` (`MergedColumnParallelLinear`) — whose `weight_loader` **DOES** take `shard_id`.
- The **`else` branch** (no `shard_id`) handles everything else — `RowParallelLinear` (`o_proj`,`down_proj`),
  `VocabParallelEmbedding` (`embed_tokens`,`lm_head`), and `RMSNorm`/norms (no custom loader -> `default_weight_loader`)
  — whose loaders take **no** `shard_id`.
- Invariant: **`packed_modules_mapping` keys map only to fused modules whose loader accepts `shard_id`.** Add a mapping
  entry pointing at a non-sharded loader -> `TypeError` (extra positional arg). `RowParallelLinear` is never in the
  mapping, so it's never passed `shard_id`.

**Load = H2D transfer.** Model built with `set_default_device("cuda")` -> params in HBM; `safe_open(..., "cpu")` ->
`get_tensor` gives a **CPU** tensor (mmap'd from the file, disk->CPU lazily). `weight_loader` narrows to this rank's
shard (cheap CPU view) then `param.data.copy_(shard)` = **CPU->GPU H2D** of only that slice, in-place into the
pre-allocated HBM buffer.

**Generic vs fused param layout** (Qwen3-0.6B, one layer): generic = 7 separate params (`q/k/v/o_proj`,
`gate/up/down_proj`, matching ckpt); fused = 4 (`qkv_proj[4096,1024]`, `o_proj`, `gate_up_proj[6144,1024]`,
`down_proj`). TP=2 further halves the split dim (qkv/gate_up / dim0, o/down / dim1). Fusion = fewer/bigger GEMMs at
runtime; the custom loader is the load-time price.

## ★★ Tensor-parallel data-flow (per-module design; replicate cheap, shard big)
Figure: `tp_dataflow.png`. Demo: `h100_setup/tp_parallel_demo.py` (+ TP figures).

**TP is a PER-MODULE design, coordinated across neighbors.** Each layer type gets its own scheme, and the
schemes are chosen so **one module's output sharding = the next module's expected input sharding** → activations
stay sharded through a block and reconcile with the **minimum** collectives.

**The canonical transformer recipe (Megatron), as in nano-vllm:**
| module | scheme | input | output / comm |
|---|---|---|---|
| `input_layernorm`, `post_attn_LN`, residual add | **replicated** | full | full (both ranks recompute locally, NO comm) |
| `qkv_proj`, `gate_up_proj` | **column-parallel** | replicated | sharded slice (this rank's heads / inter dim) |
| attention (per-head), `q_norm`/`k_norm`, RoPE, SiluAndMul | **per-rank** | sharded | sharded (no comm) |
| `o_proj`, `down_proj` | **row-parallel** | sharded | partial sum → **`all_reduce`** |
| `embed_tokens` | **vocab-parallel** | replicated ids | masked → **`all_reduce`** |
| `lm_head` | **vocab-parallel** | full | logits slice → **`gather`** |

→ **2 `all_reduce`s per layer** (end of attention, end of MLP) + embedding all_reduce + LM-head gather = the minimum.

**Why column→row chains:** `qkv (col)` splits heads → each rank runs ITS heads' attention with **no comm** →
`o_proj (row)` takes that head-split as its sharded input → ONE `all_reduce`. (col→col would need an extra
`all_gather`.) Same for `gate_up (col)` → act → `down (row)`. Get the pairing wrong → extra collectives; forget the
row `all_reduce` → wrong result (only a partial — the `tp_parallel_demo.py` failure).

**Why norms/act/residual are REPLICATED (not sharded):**
- They're **tiny** (norm weight ≈ `[hidden]` ≈ 0.02% of params; elementwise+small reduction) and sit at a
  **replicated island** between two `all_reduce`s (hidden is already full there).
- Sharding a norm would need an **`all_reduce` per norm** (to reduce variance over `hidden`) → comm ≫ the trivial
  compute saved. So replicate.
- **Recompute-on-both beats compute-once-and-broadcast:** input already replicated (recompute moves **no data**),
  compute is trivial and runs **in parallel** (twice ≈ once wall-clock), while a broadcast adds a **network
  transfer + collective overhead + a serialization stall + an extra collective**. Classic *recompute-vs-communicate*:
  comm ≫ cheap compute → redo it locally.

**What tp=2 actually buys** (given norms are redundant): the win is in the **big matmuls / attention / vocab** which
**are** split → **~2× weight capacity** (each rank holds half) + **~2× matmul throughput**. The replicated norms
waste ~0.02% — negligible. Both ranks are always working (on the sharded big ops); norms are just a small redundant island.

**Data-flow shape (fig):** replicated (blue) → `qkv/attn/o` sharded island → **`all_reduce`** → replicated → `gate_up/
act/down` sharded island → **`all_reduce`** → replicated. **Sharded islands bracketed by 2 all_reduces.**

**Is there a generic way?** No fully-automatic "optimal TP" in production inference, but generic *mechanisms* exist:
**PyTorch DTensor + `parallelize_module(plan)`** (you give a per-module colwise/rowwise plan; it auto-shards weights +
inserts collectives), **JAX GSPMD** (sharding propagation), experimental **auto-parallel search** (Alpa). Engines
hand-roll (`ColumnParallelLinear`/`RowParallelLinear`/`VocabParallelEmbedding` classes) for control over fused
weights, custom kernels (flash-attn/paged-KV), CUDA graphs, and quantization — and you still need the per-module
knowledge (which scheme, column→row chaining) even to write a correct plan.

## ★ Parallelism strategies (TP vs DP/PP/EP/SP-CP); nano-vllm supports TP only
| strategy | what's split | what's replicated | comm | fits bigger model? | scales throughput? |
|---|---|---|---|---|---|
| **TP** (tensor) | each layer's **weights** | activations at replicated islands | heavy (2 all_reduce/layer, needs NVLink) | ✅ | ~✅ (also speeds one request) |
| **DP** (data) | the **requests** | the **whole model** (full copy/rank) | none at inference (no grads) | ❌ (each rank needs full model) | ✅✅ (independent replicas) |
| **PP** (pipeline) | model **by layers** (stages) | — | light (activations at stage boundaries); pipeline bubbles | ✅ | via micro-batching |
| **EP** (expert) | **MoE experts** across ranks | dense/attention | expert all-to-all | ✅ (MoE) | ✅ |
| **SP / CP** (sequence / context) | the **sequence/context** dim | weights | attention comm (ring, …) | fits long context | — |

**nano-vllm = TP only** (`tensor_parallel_size`, single node). No DP/PP/EP/SP/CP — it's one model instance, TP-sharded.

**TP vs DP (the crisp contrast):** TP = *one* model **split** across ranks, ranks **cooperate** on **one** forward
(heavy per-layer comm) → **fit a big model** + speed a request; needs fast interconnect. DP = *whole* model **replicated**
per rank, ranks run **different** requests **independently** (no comm at inference) → **more throughput**, does NOT fit a
bigger model. TP within a node × PP across nodes × DP for replicas (× EP for MoE) is how big deployments combine them.

**DP for inference:** no gradients ⇒ DP = **N independent replicas** (per-rank requests). For **dense** models you can
just **launch N separate instances behind a router** — no model-level DP needed; vLLM's `--data-parallel-size` for dense
is mainly a **serving convenience** (one endpoint, load balancing). nano-vllm: run multiple instances yourself.

**MoE: DP + EP are entangled (NOT independent replicas):**
- MoE FFN = many experts (e.g. 256), each token routed to top-k. **Total expert params are huge** → **can't replicate**
  → **shard experts across ranks (EP)**: the full expert set = **ONE copy, partitioned** (each rank owns a distinct
  subset), **not** duplicated per rank.
- Run mode: **dense/attention replicated + data-parallel** (each rank its own requests, no comm) — cheap, per-rank KV;
  **MoE layer = all-to-all**: dispatch each token to the rank owning its expert → compute → combine back. So ranks are
  **coupled at MoE layers**; must launch DP+EP together in lockstep (impossible with naive separate instances).
- Mental model: **dense = N replicas; experts = 1 copy split N ways, reached by cross-rank token routing (all-to-all).**

**Where the MoE bottleneck is (DP+EP):** attention is **cheap per rank** (small DP-split batch, parallel, no comm).
The **MoE layer is the coordination point** — it sees the **global token set** mixed via all-to-all. It's the bottleneck
not because one GPU processes the whole batch (work is spread ≈ `B×top_k`/rank), but because of **(1) all-to-all comm**
(2×/MoE layer, ∝ tokens×hidden), **(2) load imbalance** (hot experts → straggler rank → others idle at the sync), and
**(3) aggregate token×top_k compute**. Mitigations: expert load-balancing (aux loss / capacity factor / drop),
fast all-to-all (DeepEP) + comm/compute overlap, hot-expert replication, tuned EP degree.

## ★ Data movement: mmap / page cache / pinned / H2D (weight load path)
**Weights go disk → CPU RAM → GPU — never disk → GPU directly** (that would need GPUDirect Storage / cuFile, not
used here). Two distinct transfer types: **DMA** (hardware moves bytes, no CPU) vs **CPU copy** (CPU runs a memcpy).

- **Page cache** = the OS's RAM cache of file contents (kernel memory). Disk→page cache is a **DMA**, not a CPU copy.
- **Buffered `read()`**: user code can't touch the kernel page cache → `read()` **copies page cache → user buffer**
  (1 CPU copy), then you use/transfer that.
- **mmap** (`safe_open(..., "cpu")`): **maps the page cache into user address space** → the tensor **views it directly,
  zero-copy** (no read copy), and reads are **lazy** (pages fault in on touch). This is also how CPU-only use works
  (map file → point a tensor at it → compute; no GPU needed).
- **H2D copy accounting** (`param.copy_(cpu_tensor)`):
  - source **pageable** (incl. mmap'd page cache — it's *pageable* to CUDA) → CUDA **stages pageable→pinned (1 CPU
    copy)** → DMA to GPU.
  - source **pinned** (`pin_memory=True`) → **direct DMA** to GPU (no staging copy).
- **mmap and pinning remove DIFFERENT copies:** mmap removes the **read** copy (page cache→user buffer); pinning
  removes the **H2D staging** copy. mmap'd pages are **not** pinned, so mmap alone does **not** speed the H2D.

**TP=1 (whole tensor): mmap+H2D and pinned-read+H2D do the SAME CPU-copy work (1 copy each)** — mmap pays it at
H2D-staging time, pinned-read pays it at read time (just a different stage). **Why mmap is still preferred:**
(a) **no big pinned allocation** (pinned mem is scarce; mmap uses CUDA's small internal staging), (b) **lazy** →
for **TP>1**, `narrow`-before-`copy_` faults/reads **only the shard** (contiguous shards) → less disk I/O + less copy;
(c) map the whole file once, shareable.

**nano-vllm's split:** **weights** use **mmap** (`"cpu"`) — one-time, big, TP-shardable, avoids huge pinned buffers;
**hot per-step tensors** (`prepare_*`) use **`pin_memory=True` + `non_blocking=True`** — tiny, want fastest async H2D
every step. (`torch.set_default_device("cuda")` governs only device-less factory calls → params/`graph_vars` on GPU;
`safe_open(..., "cpu")` is an explicit device → checkpoint tensors on CPU; `copy_` bridges = H2D.)
