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

## 6. Read more
- PyTorch: "Introduction to torch.compile"; torch.compiler docs; CUDA semantics → "CUDA Graphs";
  torch.cuda.graph / make_graphed_callables API; blog "Accelerating PyTorch with CUDA Graphs".
- Deeper: TorchDynamo/TorchInductor dev-blog posts.
- Worked example: nano-vllm `model_runner.capture_cudagraph` (explicit graphs) + `@torch.compile`
  in `sampler.py`/`layernorm.py` (fusion).
