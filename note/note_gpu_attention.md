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
