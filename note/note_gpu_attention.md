# Note — GPU Execution Model & Batched Attention (prefill vs decode internals)

Deep-dive from Step 3/5 discussion: how a batched forward actually runs on the GPU,
why prefill and decode need different attention kernels, and how to find the bottleneck.

Companion figures (in `h100_setup/`): `batched_decode.png`, `attention_fused.png`,
`decode_gemm_vs_gemv.png`, `flash_decoding_split_kv.png`, `prefill_saturation.png`.

---

## 1. GPU execution model (software ↔ hardware)

```
SOFTWARE (you write)              HARDWARE (runs it)
grid (one kernel launch)    →     the whole GPU
 └─ thread block (CTA)      →     placed WHOLE onto ONE SM (many blocks per SM)
     └─ warp (32 threads)   →     the unit a warp scheduler issues each cycle
         └─ thread          →     a lane on a CUDA/Tensor core
```
- **SM (Streaming Multiprocessor)** = the GPU's parallel compute unit. **H100 has 132.** Each has
  Tensor Cores, CUDA cores, registers, and shared memory / L1 (on-chip SRAM).
- **tile** = a chunk of work/data (algorithm concept). **thread block** = the group of threads that
  processes a tile (programming-model concept); usually 1 tile ↔ 1 block. A block runs **entirely on
  one SM** (never split).
- **An SM runs MANY blocks concurrently** (occupancy), limited by registers / shared mem / max
  threads (~2048) / max blocks (~32). Warps from those blocks are **interleaved cycle-by-cycle** by
  the schedulers → both genuine parallelism AND **latency hiding** (warp A stalls on memory → warp B
  runs). More resident warps → more memory requests in flight → closer to saturating HBM bandwidth.
- To use the GPU fully you need **#tiles ≫ #SMs** (enough work units to fill all SMs + hide latency).

## 2. A batched forward = batched tensor ops, NOT seq-by-seq

All B sequences are **stacked** into one tensor `[B, hidden]`; every op processes all of them at once.
The only Python seq-loop is the cheap CPU-side tensor assembly in `prepare_decode/prepare_prefill`.
Two op types with **different batching behavior**:
- **Linear layers (QKV/O/MLP proj):** `[B,d] @ shared W` → **ONE batched GEMM**. Weight is shared
  across all B rows → loaded once, reused → **amortized** (the decode throughput win).
- **Attention (QKᵀ, AV):** each row attends to its **OWN** K,V (different content + length) → **no
  shared operand** → B **independent** small matmuls, not one GEMM.

## 3. Prefill vs decode attention: [N×N] GEMM vs [1×ctx] GEMV

| | Prefill | Decode |
|---|---|---|
| queries/seq | many (N) | one |
| per-seq attn | `[N×N]` causal **GEMM** | `[1×ctx]` **GEMV** |
| arithmetic intensity | high (each key reused by N queries) → **compute-bound** | ~1 FLOP/byte (each key used by 1 query) → **memory-bound** |
| parallelism source | the **N query rows** (× heads × seqs) → fills SMs | only **batch** `B×heads` (small) → often too few |
| within-seq causal mask | yes | none (newest token sees all) |
| kernel | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |

Key intuition: **decode = computing ONE new row of the ever-growing causal attention matrix per
step** (prefill = the first N rows in bulk). Same math as "the next row," but that row has **N× less
compute per byte** → memory-bound. The n²-vs-n arithmetic intensity is **context-independent** (decode
is always more memory-bound than prefill).

## 4. Why decode underutilizes the GPU + Flash-Decoding (split-KV)

Decode has **no per-seq query parallelism** (1 query/seq). The only parallelism is **batch**
`B × num_heads`, which is often too few to fill 132 SMs (small batch), and each work unit does a
**long sequential KV scan** → too few in-flight memory requests → HBM underutilized.

**Flash-Decoding** adds parallelism on the **context dimension**:
1. split the long KV/context into chunks; each chunk → a **different SM**, in parallel.
2. each SM computes a **partial softmax** over its chunk: `(Oₛ, mₛ, lₛ)` (local output, local max, local sum-exp).
3. **combine** via log-sum-exp: `m*=max mₛ; final = Σ exp(mₛ−m*)·lₛ·Oₛ / Σ exp(mₛ−m*)·lₛ` — **exact**, not approximate.
- Same **total work & bytes** as the unsplit version; it just **distributes the reads across more SMs
  in parallel** → saturates HBM → same bytes in less time. ("Saturate HBM" ≈ "enough concurrent reads
  via more active SMs/warps.")
- Decode-specific: prefill already has query parallelism, so it **doesn't** split KV (query-tile
  parallelism is enough; splitting KV would just add reduction overhead).

## 5. Why B (decode batch) is capped
- `max_num_seqs`: explicit cap (CUDA-graph capture sizes, per-seq tensors).
- **KV cache**: fixed pool sized at init; `Σ(blocks per running seq) ≤ total blocks`. Long contexts use
  more blocks/seq → fewer concurrent seqs. KV full → `can_allocate/can_append` fail → no more admitted.

## 6. Why cross-seq attention is "skipped"
It's **semantically zero** — independent requests must not attend to each other (a block-diagonal mask
on top of causal). The varlen kernel uses **`cu_seqlens`** loop bounds so cross-seq key tiles are
**never scheduled** (not "scheduled then skipped"). Same for causal: fully-future tiles aren't
scheduled; only diagonal/boundary tiles apply a runtime mask. Nothing to do with SM placement.

## 7. Finding the bottleneck (memory-bound vs compute-bound)
- **Roofline (analytical):** arithmetic intensity (FLOPs/byte) vs ridge = peak_FLOPs / peak_BW
  (H100 bf16 ≈ 989 TFLOP/s ÷ 3.35 TB/s ≈ ~295 FLOPs/byte). Below ridge → memory-bound; above → compute-bound.
- **Nsight Compute (`ncu`)** — definitive per-kernel: **Compute(SM) Throughput % vs Memory Throughput %**
  (the ~100% one is the bottleneck), roofline chart, occupancy, and **warp-stall reasons**
  (`Long Scoreboard` = stalled on memory loads → memory-bound).
- **Nsight Systems (`nsys`) / `torch.profiler` timeline** — kernel durations + **GPU-idle gaps**
  (launch overhead, H2D copies). Good for "GPU waiting" bubbles; not for within-kernel memory-vs-compute.
- LLM heuristic: **prefill** → check Tensor-Core/SM utilization (low → under-saturated → more tokens);
  **decode** → check **DRAM bandwidth %** (near peak → memory-limited; far below → parallelism problem →
  batch more / split-KV).
- Empirical self-check (no ncu): measure achieved **TFLOP/s** and **GB/s**; whichever is near its peak
  is the bound. (See `h100_setup/profile_bottleneck.py`.)

---

## 8. Reading traces: async launches, CUDA graphs, profiling mechanics

### Async launch model (what a trace shows)
- Kernel launches are **async**: the CPU **enqueues** a kernel and returns immediately. So in a trace
  the **CPU ops cluster at the start** (fast enqueues) while the **GPU executes behind** over a longer span.
- **GPU track has NO gaps ⇒ GPU-bound** (CPU keeps the queue full; GPU is the limit — the good state).
- **GPU track HAS gaps ⇒ GPU idle**, waiting on the CPU (launch overhead) or a transfer/dependency.

### `synchronize`
- A **barrier**: blocks the CPU until the GPU finishes all enqueued work. `torch.cuda.synchronize()` →
  `cudaDeviceSynchronize`.
- Needed for **timing/profiling** (so the measured window covers GPU execution), **not for correctness**
  (CUDA enforces stream order; reading results — `.item()/.cpu()/.tolist()` — auto-syncs). Omit it → that
  exact call won't appear, but implicit syncs (memcpy from result reads, profiler-exit) still occur.

### CUDA graphs (launch-bound fix) — measured 3.5x in `profile_cudagraph.py`
- Many **tiny** kernels: launch overhead (~5 µs CPU) ≫ kernel runtime (~1 µs GPU) → GPU idles between
  them → **launch-bound** (CPU time ≫ GPU time, gaps on GPU track).
- A **CUDA graph captures the whole launch sequence once and replays it as ONE launch** → no per-kernel
  CPU dispatch → kernels run back-to-back. **Same GPU work**, just no launch overhead.
- Helps **decode** (dozens of tiny kernels/step → nano-vllm captures decode graphs). **Doesn't help
  prefill** (big kernels, launch overhead negligible → runs eager).

### One-time / first-call costs (why first op is slow)
- **`cudaMalloc`** via PyTorch's **caching allocator**: it calls `cudaMalloc` (expensive, +`cudaStreamIsCapturing`)
  **only when no cached free block fits**; freed tensors return **to the cache** (on refcount→0) and are
  **reused** → later ops do no `cudaMalloc`. Keeping outputs alive (not freeing) forces fresh allocs.
- **cuBLAS init + per-shape kernel heuristic selection**: first matmul of **each new shape** is slower
  (heuristic/autotune); same shape repeated → cached → fast. (Why varying shapes hurt → nano-vllm buckets
  decode batch sizes + CUDA-graph fixed shapes.)
- **CUDA context creation** (very first CUDA call): big one-time GPU+CPU init.
- **Profiler/CUPTI warmup**: first profiled ops carry instrumentation overhead ("Activity Buffer Request").
- During CPU-side inits the **GPU idles** (can't start a kernel until enqueued).

### gap ≠ memory-bound (3 distinct states)
| state | GPU busy? | timeline | detect |
|---|---|---|---|
| compute-bound kernel | yes | no gap | high achieved TFLOP/s |
| memory-bound kernel | **yes** | **no gap** | low TFLOP/s + high GB/s |
| GPU gap (idle) | **no** | **gap** | gaps + CPU≫GPU time |
Memory-bound is **NOT** a gap — it's a busy kernel achieving low compute. Gaps = launch/transfer waits.

### Reading FLOPs from the profiler
- `with_flops=True` → the **printed `key_averages().table()`** gets a `Total GFLOPs/MFLOPs` column;
  programmatically `e.flops` + `e.device_time_total` (us; attr renamed across versions) → achieved TFLOP/s.
- **NOT exported to the chrome `.json`** (verified: `"with_flops":1` is only a config flag; events carry
  `Input Dims`/`Input type` but no `flops`). In Perfetto, read `Input Dims` and compute FLOPs yourself.
- Per-**kernel** FLOP-efficiency → Nsight Compute (`ncu`).

---

## 9. The optimization hierarchy (diagnose in this order)

1. **Keep the GPU busy (no gaps)** — Level 1. Causes of idle: launch overhead (tiny kernels), Python/host
   work, H2D/D2H transfers, syncs. Fixes: **CUDA graphs**, overlap copies (async streams + pinned mem),
   fewer syncs, batch more. Diagnose: **timeline** (nsys / torch profiler) — bubbles, CPU≫GPU.
2. **Make each kernel efficient** — Level 2 (must know the bound first via roofline/ncu):
   - **memory-bound** → raise arithmetic intensity: **fuse kernels** (FlashAttention/fused norm — avoid HBM
     round-trips), **batch more** (amortize weights), **lower precision** (bf16/fp8), **better layout** (coalesced).
   - **compute-bound below peak** → Tensor Cores (right dtype/shape), better tiling, higher occupancy, cuBLAS/CUTLASS.
   Diagnose: **roofline** (achieved TFLOP/s vs GB/s vs peak), **ncu** (Compute% vs Memory%, occupancy, warp stalls).
3. **Do less total work** — Level 3 (algorithmic): KV cache, prefix caching, chunked prefill, speculative
   decoding, quantization, MoE, attention sparsity.

**Rule:** profile and identify the bound *first* — effort on the wrong axis is wasted (adding compute to a
memory-bound kernel, or optimizing a kernel that only runs in the gaps, does nothing).

nano-vllm mapping: L1 = CUDA graphs (decode) + pinned async H2D; L2 = FlashAttention (fused) + paged-KV
layout + batched decode + bf16; L3 = KV cache + prefix caching + chunked prefill (backlog: spec decode, quant).

### How to diagnose with roofline + ncu — see `h100_setup/roofline_diagnose.py` and the ncu command:
```
# per-kernel compute% vs memory% + the bound, on a tiny script:
ncu --set basic --target-processes all -k regex:"gemm|elementwise" \
    <env>/bin/python h100_setup/ncu_target.py
# key rows: "Compute (SM) Throughput %", "Memory Throughput %", "Achieved Occupancy",
#           warp stall "Long Scoreboard" (=waiting on memory loads → memory-bound)
```

### Worked example (measured on this H100)
**Roofline** (`roofline_diagnose.py`; ridge ≈ 295 FLOP/byte):
| kernel | intensity (FLOP/B) | achieved TFLOP/s | verdict |
|---|---|---|---|
| GEMM 8192³ | 2730 | 646 | compute-bound (right of ridge) |
| elementwise add 64M | 0.1 | 0.2 | memory-bound |
| GEMV B=1 / 16 / 64 / 256 | 1 / 16 / 64 / 254 | 2 / 37 / 144 / 500 | memory-bound, **climbing the memory roof as B↑** (decode-batching) |

**Nsight Compute** (`ncu --metrics sm__throughput…,gpu__dram_throughput…`):
| kernel | SM (compute) % | DRAM (memory) % | dur | verdict |
|---|---|---|---|---|
| GEMM `nvjet_…` | **74%** | 19% | 36 µs | COMPUTE-bound |
| add `vectorized_elementwise…` | 4.7% | **90%** | 81 µs | MEMORY-bound |

Rule confirmed: the throughput near 100% = the bottleneck. ncu kernel-name gotcha: match the *real*
kernels (`nvjet` for cuBLAS GEMM, `vectorized_elementwise` for the add) — `elementwise` alone also
matches `randn`'s init kernel. Files: `roofline_diagnose.py`, `ncu_target.py`, `roofline_diagnose.png`.

---

# Part 2 — Session addendum: op taxonomy, batching economics, warp/latency internals

New figures (`h100_setup/`): `batched_forward.png` (prefill packing vs decode + op taxonomy),
`bandwidth_util.png` (warp latency-hiding: under-utilized ~20% vs saturated ~72%).

## 10. Per-token (position-wise) vs cross-token ops — only attention mixes tokens
Classify by whether an op combines **different token positions** (the token axis) — NOT by whether it has
an internal reduction:
- **PER-TOKEN (batches for free — one GEMM/elementwise over all tokens):** embedding; RMSNorm/LayerNorm
  (*reduces over FEATURES within a token*); all Linears/GEMM; RoPE (*uses each token's own position*);
  activations SiLU/GELU/**sigmoid**/SwiGLU (*elementwise*); residual add; sampling softmax (*over VOCAB*).
- **CROSS-TOKEN (needs seq boundaries):** **ATTENTION only** — `QKᵀ → softmax over KEYS → ·V`.
- **Rule: an internal reduction ≠ cross-token.** Softmax mixes only along the axis applied — over **KEYS**
  ⇒ attention (cross-token); over **VOCAB** ⇒ sampling (per-token). RMSNorm reduces over **FEATURES** ⇒
  per-token. This is why **only attention** consumes `cu_seqlens`/`block_tables`; everything else sees a
  flat `[tokens, hidden]` batch and batches trivially.

## 11. How the batched input is built (fig `batched_forward.png`)
- **PREFILL = flat "varlen" packing**: all scheduled tokens of all seqs **concatenated into ONE 1-D
  `input_ids [total_tokens]`** (no padding); `positions` restart per seq; **`cu_seqlens_q/k`** mark seq
  boundaries → `flash_attn_varlen_func` applies a **block-diagonal (per-seq causal) mask** (cross-seq blocks
  are *skipped*, not computed-then-masked).
- **DECODE = 1 token/seq**: `input_ids [B]`; `positions=[len-1]`; `context_lens`; **`block_tables`** (which
  paged blocks hold each seq's KV); `slot_mapping` (1 slot/seq for the new K,V) → `flash_attn_with_kvcache`
  gathers each seq's own KV.

## 12. Two levers of batching + the shared-vs-per-seq reconciliation
Two independent throughput levers:
- **① arithmetic intensity** — amortize a **SHARED** HBM read → raises FLOP/byte → can flip memory→compute bound.
- **② GPU utilization** — fill SMs + hide latency + saturate bandwidth → bottleneck *stays*, but is fully used.

Decisive question: **is the data read from HBM SHARED across the batch, or PER-SEQ?**
| op | reads | shared? | intensity vs batch | batching mechanism |
|---|---|---|---|---|
| **Linear** | weights | **YES** | **rises ∝ tokens** (W read once) | **① intensity** (memory→compute); depends on **TOTAL TOKEN count, seq-agnostic** (a big prefill amortizes as well as many decode seqs); occupancy usually already fine |
| **Attention** | KV cache | **NO** (per-seq) | **flat** (bytes scale with batch too) | **② utilization only** — stays memory-bound |
- Reconciles the apparent contradiction: batching raises intensity for **linear** (shared weights) but **not
  attention** (per-seq KV) — for attention it only *fills the machine*.
- Even when attention stops gaining (long context saturates), **linears keep gaining from ① at any context**
  → batching decode stays worthwhile overall.
- **Long context** tilts total cost toward the **non-amortizable per-step KV read** (can exceed weight bytes)
  → per-token throughput degrades → why KV-size opts (paging, quantized KV, GQA/MQA) matter.

## 13. Why `max_num_batched_tokens` exists
(a) **Activation memory** — intermediates ~`total_tokens×hidden` must fit in HBM left after weights+KV
(measured in `warmup_model`); (b) **compute saturation** — past the roofline ridge, more tokens add
**latency**, not throughput/token.

## 14. Warp / SIMT internals (refines §1)
- **HARDWARE: SM, warp.  SOFTWARE: thread block, thread, grid, tile.** CPU analogies: thread block ≈ "a task
  pinned to a core"; warp ≈ a **32-wide SIMD bundle** (no clean process analogy); tile = the data-chunk a
  block owns.
- A **thread runs the WHOLE kernel on its OWN data element** (not "one line of code"; "lane" ≠ "line").
- **SIMT**: the 32 threads of a warp execute the **same instruction at the same time (lockstep parallel)**,
  each on its own data — **NOT one-after-another**. Only the instruction stream advances over time, in sync
  across all 32 lanes.
- **Warp divergence**: if lanes branch differently, hardware **serializes** the paths (masked) → lost
  parallelism. Uniform branch = full parallel.
- Two levels: **within a warp** (32 lanes lockstep) + **across warps** (4 schedulers/SM issue several
  concurrently; many resident warps time-sliced for latency hiding).

## 15. Memory-latency hiding, quantified (fig `bandwidth_util.png`)
- HBM = **high bandwidth BUT high latency**. To *use* the bandwidth you need **many reads in flight**
  (Little's law: achieved BW ≈ concurrency ÷ latency).
- **1 seq / few warps** → 1–2 KV reads in flight → bandwidth **under-utilized (~20%)**; SM mostly waits.
- **Many seqs / many warps** → whenever a warp stalls on a load, another computes → ~6–8 reads in flight →
  bandwidth **saturated (~72%)**; same per-seq work finishes far sooner.
- This is the **same mechanism as "more tiles"** — more independent warps fill compute **and** the memory
  pipeline at once. Computation never waits for "all memory read first" (flash-attn **streams KV block-by-block**).

## 16. Why one seq can't fill the GPU, and why KV-splitting is bounded
- One-seq decode-attention parallelism ≈ **(query heads) × (KV-splits)**. The **missing axis vs prefill**:
  **query tokens = 1** (prefill parallelizes over N query rows; decode lost it).
- Flash-Decoding manufactures KV-splits but **can't split infinitely**: (a) each split needs a big-enough KV
  **tile** (~128–256 keys) to amortize per-split setup; (b) softmax reduces over **ALL** keys → more splits =
  bigger online-softmax **combine** step; (c) partials go through **HBM** → more splits = more partial traffic
  (bad for memory-bound). Useful splits ≈ `L / tile_size` → few for short context.
- So: **Flash-Decoding** fills the GPU from ONE seq when **context is LONG** (small batch); **batching** adds
  back the **query-token axis** (`heads × splits × B`, cleaner parallelism) when **context is SHORT**. Both
  aim for `#work-units ≫ #SMs` with enough resident warps.

## 17. Async scheduling — NOT in nano-vllm
- `llm_engine.step` is **fully synchronous**: `schedule() → run() → postprocess()`, serial; `run()` ends in
  `.tolist()` (D2H sync) → the CPU can't overlap the next `schedule()` with the current forward.
- **Async scheduling** (vLLM V1) builds step N+1's batch on the CPU **while the GPU runs step N** — hides
  CPU schedule/prepare/postprocess behind compute. Needs deferred output handling + speculative next-batch
  (corrected on EOS). **Backlog** for nano-vllm; independent of the other optimizations.

## 18. Paged KV (storage) vs KV-split (Flash-Decoding) — independent concepts
Two "chop up the KV" ideas at **different layers**:
| | **KV page / block** (paged attn) | **KV-split** (Flash-Decoding) |
|---|---|---|
| what | **storage/addressing** unit in HBM | **parallelization** unit (context slice → one SM) |
| set by | engine (`kvcache_block_size`=256), **fixed** | kernel, **per-launch heuristic** |
| for | **memory management** (fragmentation vs metadata) | **occupancy** (fill SMs) |
| carrier | `block_table` (logical→physical); write via `slot_mapping` | kernel inner loop / `num_splits` |
- **No 1:1 requirement.** A split is a **range of the context spanning many pages**; it gathers its pages via `block_table`. Split *count* is occupancy-driven, **independent of page size** (split boundaries are usually page-aligned for gather cleanliness, but that's convenience). Paging = *how KV is stored & addressed*; splitting = *how the kernel reads & computes over it in parallel*.

### Paged attention: benefit vs cost (it's a trade-off, not free locality)
- **Benefit (the win): memory efficiency, not compression.** Contiguous per-seq allocation reserves `max_seq_len` → **60–80% waste**; paging allocates blocks **on demand** (~few-% waste) + allows **prefix sharing** → **more concurrent seqs → bigger batch → more throughput.**
- **Cost:** paging **scatters** pages + adds `block_table` **indirection** → *slightly less* locality than contiguous, **not more**. Locality is kept **"good enough"**: within a block KV is contiguous (**coalesced**), and the block (256) is big enough that the per-block lookup is negligible. A split's read = **"contiguous within each page, hopping between scattered pages"** (a gather of coalesced chunks).
- **`block_size` trade-off:** bigger → more coalescing + less metadata but **more tail fragmentation**; smaller → less waste but **more indirection**. 256 balances.

### Coalescing granularity — what the warp actually cares about
Size hierarchy: **`load ⊆ tile ⊆ page ⊆ split`**.
- A **load** (one instruction, 32 lanes → consecutive addresses) is the **coalescing unit** — tens/hundreds of bytes, **far smaller than a page** → it **always lands inside one page**.
- The warp **cares about within-page contiguity** (for coalesced loads); it is **indifferent to inter-page adjacency** — scattered pages are resolved by `block_table` (each page read from its own base addr).
- **Multiple pages per warp/block is normal** (a split = many pages): the kernel loops **tile-by-tile**, switching page base address **between** loads — never straddling a page boundary within a load. So scattered pages cost only a tiny **per-block lookup**, no coalescing penalty.
- **Real coupling: page size ↔ kernel TILE size**, NOT page ↔ split size. The page must be a **multiple of / aligned to the kernel's `BLOCK_N`** so tiles never cross a page boundary.

### flash-attn is a **paged-aware** kernel (imposes the block_size rule)
- Vanilla flash-attn (contiguous `[B,L,H,D]`) is **not** paged-aware. nano-vllm's paths **are**: `flash_attn_with_kvcache(..., block_table=…, cache_seqlens=…)` (decode) and `flash_attn_varlen_func(..., block_table=…)` (prefill cache-hit) — they **gather scattered pages** via `block_table`.
- Because it's **tile-based**, it **requires `block_size` to be a multiple of its internal `BLOCK_N`** (16 or 256 by version) so tiles stay inside a page. **The kernel dictates the rule; the engine picks a compatible `block_size`** — nano-vllm's **256** satisfies it *and* is memory-efficient.
- **Co-design:** paging (storage) needs a paged-aware kernel to read it — options: **vLLM's own `paged_attention` CUDA kernel**, **flash-attn's paged path** (nano-vllm), **FlashInfer**. Each takes `block_table` and imposes its own alignment rule.
