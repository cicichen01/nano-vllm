# Note — CUDA-graph capture: what can (and can't) be captured

Empirically verified deep-dive (scripts in `h100_setup/`: `capture_zeros_attn.py`, `reshape_no_cpu.py`,
`reshape_chain.py`, `reshape_alloc.py`, `reshape_g2.py`, `reshape_kernel_flag.py`, `reshape_user.py`,
`loop_in_kernel.py`).

## 1. What a graph freezes: LAUNCHES (shapes + addresses), NOT data
A CUDA graph records the **sequence of kernel launches** — each kernel's **grid/block config (= shapes)** and
**argument addresses**. It does **not** record data values or the arithmetic. **Replay** = re-run those recorded
launches against **whatever is in the (fixed-address) buffers now**. So **capture is a throwaway "dry run"** — the
data used at capture is discarded.

## 2. capture-with-zeros works (`capture_cudagraph`)
nano-vllm captures `self.model(...)` on **all-zero** inputs (input_ids/positions/slot_mapping/context_lens/
block_tables = 0). Works because capture records **launches, not values**; zeros are a dry run.
- Requires: **no value-dependent control flow** that changes *which* kernels launch or their shapes. The transformer
  forward qualifies (all intermediate shapes = f(input shape, config constants), never token values).
- A **warmup** eager run precedes capture so lazy init / cuBLAS heuristics / allocations happen **outside** the
  captured region (a graph can't contain `cudaMalloc`/init).
- **Verified:** capture `flash_attn_with_kvcache` with `context_lens=0`/`block_table=0` (output **0.0** at capture),
  replay with **real** KV → **bit-exact vs eager (`max|diff|=0`)**. Nothing missed. (`capture_zeros_attn.py`)

## 3. The fundamental rule: shapes FROZEN, values free WITHIN shapes
| a runtime value affects… | graphable? | evidence |
|---|---|---|
| **content / layout / loop-count** within a FIXED shape | ✅ no sync | Case A; on-GPU transpose `reshape_no_cpu.py`; `context_lens` |
| a **SHAPE** (new host-visible size) | ❌ capture FAILS (host sync) | boolean-index (B); `.item()` reshape (C); `if flag!=0` (G1) |
- A graph is **intrinsically shape-frozen**: feed a different size and it **silently does the captured size**
  (captured 4 elems, replay with 8 → only 4 processed, **no sync**) (`reshape_chain.py` D/E).
- **Fresh allocation with a CONSTANT shape is fine** (the graph pool pins its address). **Pre-allocation is NOT the
  requirement — a FIXED SHAPE is** (Case F, `reshape_alloc.py`). Pre-allocated **static buffers are needed only for
  INPUTS you copy new data into each step** (they need stable *addresses*).

## 4. Why value→shape forces a sync (root cause)
A **shape is host-side control**: the host uses it to **allocate** the output and set each kernel's **launch grid** —
both **before** the kernel runs. To derive a shape from a **GPU value**, the host must **learn the value** →
**device→host read → SYNC**. Syncs are **illegal during capture**, so capture fails.
- So *"capture can't finish with a shape change"* is a **consequence**; the root is **"value→shape needs a host sync."**
- Verified: `flag != 0` alone (pure GPU op) captures fine; `if flag != 0` fails — the `if`/`bool(tensor)` does the
  implicit `.item()` (`reshape_user.py`).
- **On-device control flow can branch on a value WITHOUT a sync** (in-kernel `if`/`while`, `tl.where`; CUDA-graph
  **conditional nodes** ≥12.3; **dynamic parallelism**) — but each path is still **FIXED shape** (select among fixed
  shapes / drive content within one). A kernel reading `tl.load(flag)` changes **CONTENT, not the output tensor's
  SHAPE** — the host allocated the output *before* the launch (`reshape_kernel_flag.py`, `reshape_g2.py`).

## 5. In-kernel loops: how a graph "captures" work that didn't run at capture
The KV-block loop (`load K → Q·Kᵀ → softmax → ·V` per block) is **INSIDE ONE kernel** (device-side loop), **not** a
host loop launching a kernel per block.
- The graph captures the **kernel LAUNCH** (one node); the loop + Q·Kᵀ code is **compiled into the kernel binary**,
  captured **by reference**.
- `context_lens=0` at capture → the in-kernel loop runs **0×** (no Q·Kᵀ executed, output 0) — but the **launch (with
  the code) is recorded**. At replay with real `context_lens`, the **same kernel** loops the real number of times →
  real Q·Kᵀ. (`loop_in_kernel.py`: capture n=0→0; replay n=5→15, n=8→36.)
- Contrast: a **host-side** loop `for b in range(context_lens): launch(...)` would capture **0 launches** at
  `context_lens=0` → miss everything. flash-attn puts the loop **in-kernel** precisely to avoid this.

## 6. Why decode attention is graphable (co-design with paged attention)
- `flash_attn_with_kvcache` is **written graph-compatibly**: launch (grid/num_splits) from **SHAPES** (bs, heads,
  `block_table` shape); variable KV length as a **VALUE** (`context_lens`) consumed by an **in-kernel loop**; no host
  sync/alloc inside.
- **Paged attention enables this**: KV is a **fixed-shape cache** addressed by `block_tables` (fixed shape) +
  `context_lens` (values). It **converts the variable seqlen from a SHAPE** (dense `[bs, seqlen]` would vary per step
  → not graphable) **into a VALUE + fixed cache**.
- So graphability is **engineered** (kernel design + data layout), not automatic. A **dense-KV / host-side-block-loop /
  grid-from-value / internal-sync** implementation would **lose** it.
- **Prefill isn't graphable**: its input *shapes* (`total_tokens`, `cu_seqlens`) vary per call; and it's compute-bound
  so graphs wouldn't help anyway.

## 7. nano-vllm `run_model` graph path
- **Bucketing**: one graph per bs bucket `[1,2,4,8,16,32,…,512]`; pick **smallest bucket ≥ bs**; pad extra rows
  (`slot_mapping=-1`, `context_lens=0`). A graph's kernels have **grids for its bucket** → it does the **bucket's worth
  of work**. Using a *bigger* bucket is **correct** (padding discarded via `outputs[:bs]`) but **wastes compute**
  (GEMMs process all bucket rows; attention padding rows are ~free since `context_lens=0`). Hence smallest-≥-bs.
  ~36 graphs balance padding-waste vs #graphs (a graph per exact bs = 512 graphs = too much).
  - Corollary: replaying the bucket-B graph costs ~the same for any bs ≤ B (it always does B rows) — that "sameness"
    *is* the wasted compute bucketing minimizes.
- **`graph_vars` copies**: `input_ids` is **already on GPU** (`prepare_decode` did the H2D), so
  `graph_vars[:bs] = input_ids` is a **D2D** copy → **two hops: H2D (prepare_decode) + D2D (graph_vars)**. Could be a
  single **H2D directly into `graph_vars`**; two hops exist because `prepare_decode` is **shared with the eager path**.
  The D2D is ~µs (negligible). Must land in the **same `graph_vars` buffers** (frozen addresses) before `replay()`.
- **`graph_vars` are HBM**: created by `torch.zeros(...)` inside `capture_cudagraph`, which runs while
  `torch.set_default_device("cuda")` is in effect (set in `__init__` **before** capture, reset to `"cpu"` **after**) —
  no explicit `.cuda()`.

## 8. `set_default_device` vs explicit device (ties to weight loading)
- `torch.set_default_device("cuda")` affects **only factory ops without an explicit device** (model params → GPU,
  `graph_vars` → GPU). An **explicit device overrides it**.
- `safe_open(file, "pt", "cpu")` **explicitly** → `f.get_tensor` is a **CPU** tensor even under default-cuda. So
  **params (GPU) + checkpoint (CPU)** → `weight_loader`'s `copy_` = **H2D**.
- CPU-first is a **choice**: mmap (lazy disk read) + **narrow-to-shard on CPU** (cheap views) + **copy only the shard**
  to GPU (efficient, TP-aware). Loading direct-to-GPU would eagerly transfer the whole tensor.

## One-line laws
- **Graph freezes shapes + addresses; values are free only where they change content/loops within a fixed shape.**
- **value→shape ⟹ host must learn the size ⟹ sync ⟹ capture fails.** (Consequence, not a special rule.)
- **Capture records kernel *launches* (code by reference), not executions** — so in-kernel value-driven loops (KV
  blocks) are captured even when they run 0× at capture.
- **Decode is graphable because flash-attn + paged-KV are co-designed** to keep launches shape-determined and push all
  variability into values; **prefill isn't** (its token count *is* a varying shape).
