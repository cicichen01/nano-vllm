# Learning read — torch.compile vs CUDA graphs

## The problem both attack: eager-mode overhead
Each eager op pays: (1) Python/dispatch overhead, (2) kernel launch (~µs CPU each),
(3) no fusion (each op writes to HBM, next reads back). Two problems → two tools:
- lack of fusion + dispatch → **torch.compile** (fewer, bigger, fused kernels)
- per-kernel launch overhead → **CUDA graphs** (replay many launches as one)

## 1. torch.compile (high-level JIT compiler, PyTorch 2.x)
`cmodel = torch.compile(model)`  — call as usual.
Pipeline: **TorchDynamo** (Python→FX graph; "graph breaks" on untraceable code) →
**AOTAutograd** (backward) → **TorchInductor** (generates fused Triton/C++ kernels).
Get: fusion, less dispatch, sometimes better algos. First call slow (compile), then fast.
Modes:
- `torch.compile(model)` — Inductor fusion
- `torch.compile(model, mode="reduce-overhead")` — + CUDA graphs (launch overhead too)
- `torch.compile(model, mode="max-autotune")` — autotune; slowest compile, fastest run
Gotchas: first-call compile latency; **recompiles on shape/dtype change** (`dynamic=True` to
generalize); **graph breaks** cut benefit (`TORCH_LOGS="graph_breaks,recompiles"`); opaque debugging
(`torch._dynamo.explain`).

## 2. CUDA graphs (low-level launch-overhead eliminator)
Record a kernel sequence ONCE, **replay as ONE launch** → removes per-kernel CPU dispatch.
Constraints: static shapes (one graph/shape → bucketing), **static input addresses** (copy new data
into the SAME buffers), no CPU syncs inside, no dynamic alloc inside, deterministic control flow.
Manual API (nano-vllm):
```
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): static_out = model(static_in)   # capture
static_in.copy_(new); g.replay(); use(static_out)         # per step
```
Helper: `torch.cuda.make_graphed_callables(fn, sample_args)` (does the buffer dance for you).
Best for: many tiny kernels (decode), fixed shapes, repeated calls. Gotchas: static-buffer
copy-in/out, capture warmup, per-shape graphs (bucketing+padding).

## 3. How they relate (compose)
- torch.compile = fewer/bigger FUSED kernels (dispatch + HBM traffic)
- CUDA graphs = replay launches as ONE (launch overhead)
- `torch.compile(model, mode="reduce-overhead")` = Inductor fusion + CUDA graphs (both, one line)
- Or use separately: nano-vllm uses `@torch.compile` on RMSNorm/Sampler (fusion) AND explicit
  `torch.cuda.graph` for the decode forward (launch overhead).

## 4. When to use which
| situation | tool |
|---|---|
| general speedup, easy | torch.compile(model) |
| also kill launch overhead | torch.compile(model, mode="reduce-overhead") |
| tiny-kernel launch-bound forward, need control | explicit torch.cuda.graph (bucketing+static buffers) |
| fuse a few hot ops | @torch.compile on those fns |

## 5. Mental model
- eager = interpreter (flexible, per-op overhead)
- torch.compile = compiler (trace→optimize→fused kernels; compile time; recompiles on change)
- CUDA graph = macro recording of GPU launches, replayed with no CPU in the loop (fixed shapes/addresses)

## 6a. Are they the same thing? (they are NOT)
- **torch.compile (Inductor) CHANGES the kernels that run** — traces the graph, fuses ops, generates new
  Triton/C++ kernels. Fewer, bigger kernels; less HBM traffic; sometimes better algos.
- **CUDA graph does NOT change any kernel** — same kernels, same order; it only removes the *per-kernel CPU
  launch* by replaying the whole sequence as one `cudaGraphLaunch`.
- `default` mode = Inductor fusion **only** (no CUDA graph). `reduce-overhead` = Inductor fusion **+** CUDA
  graphs. `max-autotune` = fusion + autotuned kernel/template picks (+ graphs), slowest compile.
- **max-autotune can pick different kernels per input shape** (it benchmarks templates for the shapes it
  sees → recompiles/re-tunes on new shapes). Epilogue fusion: it folds a trailing pointwise (bias, gelu,
  scale) *into the matmul kernel's HBM write* — the matmul writes the already-activated result, saving a
  full read+write round-trip to HBM.

## 6b. If there's only ONE big op, does CUDA graph still help?
Barely. Graph's win = eliminating N launches; with 1 op there's ~1 launch to eliminate. Fusion's win comes
from a different place (HBM traffic + kernel count). Proven with `torch_compile_hbm.py` (chain of 20
pointwise ops):

| size | mode | ms/iter | GPU kernels | reading |
|---|---|---|---|---|
| LARGE (16M elems, memory-bound) | eager | 4.03 | 20 | 20 kernels, each reads+writes 64 MB |
| | graph-only | 3.92 | 20 | launches gone, **but still 20 kernels × HBM → ~no help** |
| | fused (compile) | 0.068 | 1 | **59× — one kernel, data stays in registers, 1 HBM round-trip** |

→ **Fusion wins on 3 fronts** (kernel count ↓, HBM traffic ↓, launches ↓). **Graph-only wins on 1 front**
(launches) — useless when you're memory-bound, decisive when you're launch-bound (many tiny kernels).

## 6c. Toy transformer block: all 4 modes (`torch_compile_playground.py`)
Small dims + small batch (launch-bound on purpose). Measured pattern:
- **eager**: most kernels, most `cudaLaunchKernel`, 0 `cudaGraphLaunch`.
- **default**: fewer kernels (fusion), fewer launches, 0 graph launches.
- **reduce-overhead**: fused kernels **replayed as a graph** → `cudaGraphLaunch` appears, `cudaLaunchKernel`
  collapses to ~0 in the steady state.
- **max-autotune**: similar kernel count to default but tuned; longest compile+warmup time.

## 6d. Real Qwen3-0.6B, eager vs CUDA-graph decode (`qwen_eager_vs_graph.py`)
nano-vllm's actual decode path (graph captures `self.model(...)`; lm_head + sampler run eager after):

| mode | ms/decode-step | ~kernels | cudaLaunchKernel | cudaGraphLaunch |
|---|---|---|---|---|
| eager | 34.34 | ~430 | ~86 | 0 |
| CUDA graph | 5.26 | ~430 | ~3 | 1 |

→ **6.5× faster** with the SAME ~430 kernels — pure launch-overhead elimination. Decode is launch-bound
(tiny per-step work, ~430 sequential launches), exactly the case graphs are built for.

## 6e. Reading the graph-mode decode trace (per-step CPU pattern)
`run()` = `prepare_decode` → `run_model` (graph replay) → `sampler(...).tolist()`. On the CPU track each
decode step repeats:
```
[prepare_decode + copy inputs into graph_vars]  →  cudaGraphLaunch  →  [lm_head matmul]
   →  "Torch-Compiled Region" (the Sampler)  →  aten::to (.tolist() D2H copy + implicit sync)
```
- **cudaGraphLaunch** = the whole decoder stack (embed + 28 layers + final norm) replayed as ONE launch.
  RMSNorm is `@torch.compile`'d but lives *inside* the graph → swallowed into this one launch (no separate
  CPU marker).
- **"Torch-Compiled Region"** = an Inductor-compiled region running eagerly *outside* the graph — here the
  `@torch.compile` **Sampler** (temperature ÷ → softmax → exponential/argmax). Shows on CPU because the
  sampler is not captured in the graph.
- **aten::to** = the `.tolist()` copying the sampled token IDs **device→host** so Python can read them; this
  also forces the step's implicit GPU sync.

## 6f. Graph breaks (`graphbreak_trace.py` / graphbreak_with.json)
A break = Dynamo hits untraceable code (`.item()`, `.tolist()`, data-dependent control flow, opaque custom
ops like flash_attn), stops the current subgraph, runs that op **eagerly**, then resumes compiling.
- Measured: 1 break → **2 "Torch-Compiled Region" markers/call** (vs 1 with no break); the eager op sits in
  the gap between them.
- **The model still runs correctly** — this is why nano-vllm can `@torch.compile` around the custom
  flash_attn op. You just lose fusion *across* the break and re-materialize to HBM there.
- More breaks = more, smaller compiled islands = less benefit. Diagnose with `TORCH_LOGS="graph_breaks,recompiles"`
  or `torch._dynamo.explain(fn)(args)` (`.graph_break_count`, `.graph_count`).

## 6g. Why nano-vllm resets the Context (CUDA-graph correctness)
`reset_context()` after each `run()` matters *because of* CUDA graphs: a captured graph reads from the
**frozen buffer addresses** it saw at capture time. If stale Context (slot_mapping/block_tables) leaked into
a replay, the graph would read wrong KV addresses. Eager mode is far more forgiving; the discipline exists
to keep replay inputs well-defined. Graphs are captured per **bucketed batch size** `[1,2,4,…,512]`; a step
rounds `bs` up to the nearest bucket, copies live inputs into the static `graph_vars`, zero-fills unused
rows, and replays.

## 6h. Is per-model `capture_cudagraph` / explicit graphs standard?
- **Explicit `torch.cuda.graph` is manual by design** — there's no one-liner "run this eager or not" for
  arbitrary models because graphs need static shapes/addresses. The one-liner *is* `mode="reduce-overhead"`
  (torch.compile does the buffer dance via `make_graphed_callables`). Inference engines (vLLM, SGLang,
  TensorRT-LLM) mostly hand-roll capture like nano-vllm to control bucketing, padding, and the
  static-buffer layout for the decode forward.
- So yes, a bespoke `capture_cudagraph` per engine (often per model family) is normal for high-perf serving.

---
# Part 2 — Trace-reading & internals deep-dive

## 7. FX graph vs CUDA graph (different things, same word "graph")
| | **FX graph** | **CUDA graph** |
|---|---|---|
| what | a description of tensor *ops* (matmul, add, gelu) | a recording of GPU *kernel launches* + fixed addresses |
| made by | TorchDynamo (torch.compile front-end) | CUDA driver (`torch.cuda.graph`) |
| when/where | compile time, on CPU (Python objects) | runtime, on GPU/driver |
| purpose | let Inductor optimize/fuse → generate kernels | replay many launches as one `cudaGraphLaunch` |
| a "break" | splits into 2 FX graphs (untraceable code) | N/A |
Pipeline: Python → **FX graph** (recipe) → Inductor **fused kernels** (how) → optionally record launches → **CUDA graph** (replay cheaply). FX graph = plan; CUDA graph = recording of the execution.

## 8. `explain` vs `compile`; neither "builds the model"
- `make(fn)` builds the *function* (weights are pre-existing globals).
- `torch._dynamo.explain(fn)(x)` — runs Dynamo trace for a **report** (`graph_break_count`, `graph_count`), no Inductor, throwaway.
- `torch.compile(fn)` — runs Dynamo **+ Inductor** → cached **fast callable**. Both trace (so both see the same breaks); only `compile` generates runnable kernels. `torch._dynamo.reset()` between them clears the cache so `compile` recompiles cleanly.

## 9. Compile-id `X/Y` (the "Torch-Compiled Region: X/Y" marker)
- **X = frame id** — which compiled function.
- **Y = frame_compile_id** — **which cached version (specialization) is executing here**, numbered in creation order. NOT a running "total compiles" counter — it's a *version tag* chosen per-call by shape via guards.
- torch.compile **specializes on shape by default** → a new shape fails the guard → a **new version** (higher Y) is compiled. Different functions meet different #s of shapes → different max Y.
- **Nested markers** (`0/0` wraps whole call, `1/0` nested) = a **graph break**: `0/0` is the entry frame (ran subgraph1 + the eager break op + *called* the resume continuation), `1/0` is the synthesized **resume function** (subgraph2), called from inside `0/0`.
- **Reuse vs recompile in a trace** (proven on `qwen_eager.json`, frame 0 ran both `0/4`×20 and `0/5`×1120): (a) **interleaved** old+new versions → reuse (a recompile is one-way; the old version can't resurface); (b) **no ms-scale spike** on any first call (all µs) → nothing compiled inside the trace. A *real* recompile = a brand-new Y appearing with a **ms-scale first call** + Dynamo/Inductor CPU activity. `cache_size_limit` (default 8) caps versions; beyond it → eager/dynamic fallback.

## 10. Graph breaks: `.item()` = host-sync AND break (separable costs)
`.item()`/`.tolist()`/`.cpu()`/data-dependent `if`/opaque custom ops trigger a break. `.item()` bundles **two independent costs**:
- **sync bubble** (from the *host read*): CPU must wait for the GPU value → can't run ahead → GPU/CPU idle bubble. Happens even in eager. It's a *serialization/overlap* loss, **not** extra per-kernel launch overhead.
- **lost fusion** (from the *graph split*): 2 subgraphs → intermediates go to HBM, no cross-break fusion. Happens for **any** break.
They're separable: a **non-syncing** break (custom `flash_attn`) → fusion loss only, overlap preserved; `.item()` → both. Demo `graphbreak_with.json` (1 break → **2** compiled regions) vs `graphbreak_without.json` (**1** region). **The model still runs correctly** — this is why nano-vllm can `@torch.compile` around flash_attn.

## 11. Reading kernels in a trace (kernels ≠ Python ops)
From `graphbreak_without.json` (`g(x)=gelu(x@W); (a*1.0)@W` → 5 kernels):
- **each matmul can be 2 kernels**: `cutlass sgemm` + `cublasLt splitKreduce_kernel` — cuBLAS picked a **split-K** algo (small M, large K → split the contraction across blocks → partial sums, then a reduce kernel). fp32 → `simt_sgemm` (CUDA cores, not tensor cores).
- **pointwise ops fuse**: `gelu` + `*1.0` → one `triton_poi_fused_gelu_mul_0`.
- **fusion follows the subgraph boundary**: in the break case, `gelu` fused with the `sum` feeding `.item()` (`triton_red_fused_gelu_sum`) instead of with `mul` — where you break changes what welds together.
- **Pointwise op** = elementwise, `out[i]=f(in[i])`, no cross-position mixing (gelu, silu, add, mul, scale). Memory-bound alone; prime fusion candidates. NOT pointwise: matmul (mixes row×col), reductions (sum/mean/softmax/norm), attention.

## 12. Where nano-vllm applies `@torch.compile` (5 functions → 5 frames)
`grep @torch.compile nanovllm/layers/*.py`:
| frame | function (file:line) | kernel(s) | calls/decode-step |
|---|---|---|---|
| 0 | `RMSNorm.add_rms_forward` (layernorm.py:28) | `..._add_mean_mul_pow_rsqrt` | 56 = 2×28 (input+post norms, +final) |
| 1 | `RotaryEmbedding.forward` (rotary_embedding.py:37) | `..._cat_0/_cat_1` | 28 |
| 2 | `RMSNorm.rms_forward` (layernorm.py:16) | `..._mean_mul_pow_rsqrt` | 56 = 2×28 (q_norm, k_norm) |
| 3 | `SiluAndMul.forward` (activation.py:8) | `..._mul_silu` | 28 |
| 4 | `Sampler.forward` (sampler.py:7) | softmax/argmax | 1 |
- Decorators are on the **branch sub-methods** (`rms_forward`/`add_rms_forward`) not `forward()`, to avoid breaking on the `if residual is None`. The kernel name literally lists the fused ops (`_to_copy`=dtype cast, `add`=residual, `pow/mean/rsqrt/mul`=RMSNorm math) — ~7 eager ops → **1 fused Triton kernel**.
- **`enforce_eager=True` only disables CUDA graphs, NOT `@torch.compile`** → these 5 still run as compiled Triton kernels with "Torch-Compiled Region" markers even in the eager trace.
- **No marker ⇒ not torch.compiled**: `store_kvcache_kernel` (hand-written `@triton.jit`), `flash_fwd_*` (flash-attn library CUDA), `nvjet` GEMM (cuBLAS), `splitKreduce`.

## 13. Decode kernel anatomy: 15 kernels = ONE decoder layer
The `cudaGraphLaunch` replays 28× this 15-kernel block (28 layers). Per layer:
```
ATTENTION block (1–10):
  rmsnorm(input) → qkv_proj[GEMM] → q_norm,k_norm[2×rmsnorm] → RoPE q,k[2×cat]
  → store_kvcache → flash_fwd_splitkv + combine (Flash-Decoding: split KV across SMs, then merge)
  → o_proj[GEMM]
MLP/FFN block (11–15):   (= FFN = SwiGLU, gated)
  rmsnorm(post) → gate_up_proj[GEMM] → SiluAndMul → down_proj[GEMM+splitKreduce]
```
Notes: **4 RMSNorms/layer** incl. Qwen3's distinctive **QK-norm** (q_norm/k_norm); **4 GEMMs/layer** (`nvjet`=cuBLAS tensor-core); decode attention = **2 kernels** (Flash-Decoding split+combine). 15×28 + embed + final norm + lm_head ≈ **~432 kernels/step** (matches measured).

## 14. Memcpy / pinned memory / H2D physics
Around `cudaGraphLaunch` in `qwen_graph.json`: **6× HtoD** (`prepare_decode` uploads per-step metadata from pinned CPU) + **5× DtoD** (stage into the **static `graph_vars` buffers** the graph reads from — graph reads *frozen addresses*, must refresh each step) + **1× DtoH** (`.tolist()` tokens back to CPU). **Normal + mandatory** (input staging must live *outside* the graph).
- **Not worth optimizing**: real GPU-side transfer = **13 µs/step = 0.85%** of the 1561 µs/step compute. The big CPU numbers (`cudaMemcpyAsync` 1477 µs, `cudaGraphLaunch` 427 µs) are **API wait/blocking, not bytes** — dominated by the `.tolist()` **sync** (inherent to reading a token/step).
- **Pinned memory** = page-locked host RAM (OS can't move/swap it). DMA needs a **fixed physical address**. **Pageable** source → CUDA must first copy to a hidden pinned **staging buffer (still in host RAM)** → DMA to HBM = **2 moves**. **Pinned** source → DMA straight to HBM = **1 move**. The staging buffer is **host memory, not HBM**; every H2D ends with bytes physically in **HBM** (GPU cores read HBM, not host). `.cuda()` always materializes in HBM (zero-copy/mapped memory is a separate opt-in, slow, not this path).
- **`non_blocking=True`** = the copy call returns immediately (CPU races ahead); the DMA runs in background. Only meaningful **on pinned memory** (pageable forces the sync staging copy anyway). Pinned + non_blocking = async DMA overlapping CPU work.

## 15. Custom kernels & how they compose with torch.compile
"Custom kernel" = anything you didn't let Inductor generate: **`@triton.jit`** (easiest hand-written), **raw CUDA C++** (`cpp_extension`), **CUTLASS/CuTe**, or **prebuilt libraries** (cuBLAS, cuDNN, flash-attn). "Triton" isn't the dividing line — **Inductor itself emits Triton**; the line is *compiler-generated* vs *hand-written*.
- **Why hand-write**: algorithms Inductor can't invent (flash-attn online-softmax, paged-KV scatter, MoE, quant), layout/tensor-core control, or beating suboptimal generated code. nano-vllm mixes all: generated-Triton norms/act, hand-`@triton.jit` `store_kvcache`, library flash-attn + cuBLAS.
- **torch.compile + a custom kernel**: since ~2.3 it can **include a user `@triton.jit` in the graph without a break**, but treats it as an **opaque black box** — it **fuses *around* it, never *into* it**. Your kernel stays its own launch, unchanged. (Prologue/epilogue fusion into a kernel is only for Inductor's *own* templates, e.g. its matmul.) To avoid a break on a CUDA/opaque op, register it with `torch.library.custom_op` + `register_fake`.

## 16. nano-vllm vs vLLM CUDA graphs; cost of going "torch.compile-only"
- **vLLM has manual CUDA graphs too**: V0 `CUDAGraphRunner`/`capture_model` (same bucketed-static-buffer pattern nano-vllm mirrors); V1 (torch.compile era) uses a **custom Inductor backend for fusion + manually-captured "piecewise CUDA graphs"** around the attention op. It does **not** rely on `mode="reduce-overhead"` alone.
- **Making nano-vllm torch.compile-only** (`model_runner.py`): mechanically small — ~5 edits, −50 lines (drop `capture_cudagraph` L222–257, the `run_model` graph branch L199–212, init/exit hooks; add `torch.compile(self.model, mode="reduce-overhead")`). **But to keep the 6.5× decode win** you must also: (1) **kill the global `Context`**, threading slot_mapping/block_tables/context_lens as **tensor args** through model→layer→attention (Dynamo can't trace mutable globals; cudagraph needs stable-address tensors) — touches context.py/attention.py/qwen3.py; (2) **register `store_kvcache`+`flash_attn` as custom ops**; (3) **re-add bucket padding** (cudagraphs need static shapes; varying `bs` else recompiles/recaptures per size). → complexity **relocates, not disappears**. That's why both nano-vllm and vLLM hand-roll cudagraphs: serving hits all 3 of cudagraph's hard constraints at once (dynamic batch, custom attention op that breaks the graph, in-place KV writes via metadata tensors).

## 6. Read more
- PyTorch: "Introduction to torch.compile"; torch.compiler docs; CUDA semantics → "CUDA Graphs";
  torch.cuda.graph / make_graphed_callables API; blog "Accelerating PyTorch with CUDA Graphs".
- Deeper: TorchDynamo/TorchInductor dev-blog posts.
- Worked example: nano-vllm `model_runner.capture_cudagraph` (explicit graphs) + `@torch.compile`
  in `sampler.py`/`layernorm.py` (fusion).
