# Note — Designing an inference engine: how the engine↔model boundary emerges

Conceptual companion to the code walkthrough. Answers "how does an engine author know, from zero,
what belongs in the engine vs the model?" — and the mindset behind an inference system.

## 1. Start from the goal + the scarce resource
Goal: **serve autoregressive LLM inference (prompts → tokens) maximizing throughput (tok/$/s) while meeting
latency SLOs (TTFT, inter-token), on fixed GPUs.** The **scarce resource is the GPU (compute + HBM)**, so the whole
design is one repeated question: **"where is the GPU wasted, and how do I stop it — for *any* model?"** Every engine
abstraction is an answer to that.

## 2. The key move: separate INVARIANT (across models) from VARIABLE (per model)
- **Invariant** across all autoregressive transformers → the **ENGINE (mechanism)**: request lifecycle
  (prompt→prefill→decode→EOS), batching, KV caching, scheduling, memory management, sampling.
- **Variable** per model → the **MODEL FILE (the math)**: which layers, attention variant, norm placement,
  activation, MoE, positional encoding.
**That split *is* the boundary.** Engine = "how to run *any* AR transformer efficiently on a GPU"; model = "what
*this* transformer computes." You find it by asking **"what's common vs what differs,"** not by guessing.

## 3. How it's discovered from zero: build naive → profile → factor
Nobody knows the abstractions up front. You build the dumbest thing, profile, and **factor each GPU-wasting
bottleneck's reusable fix out of the model into the engine.** The boundary is the accumulation of those factorings:
```
1. Naive: loop forward() one token/step, one request           → works, slow
2. Profile → recomputes KV, memory-bound decode, GPU idle
3. add KV CACHE (stop recomputing)                             → engine
4. serve many reqs: batch them; they arrive/finish at diff
   times → CONTINUOUS BATCHING → a SCHEDULER                   → engine
5. KV fragments / can't fit many → PAGED KV → BLOCKMANAGER     → engine
6. decode = many tiny kernels, launch-bound → CUDA GRAPHS →
   a RUNNER (+ a shape-determinism contract on the model)      → engine
7. model too big → TENSOR PARALLELISM → parallel layer prims   → engine mechanism
8. many models → arithmetic in a MODEL FILE using a layer lib  → model side
```
Each step **factors a reusable, model-agnostic solution into the engine and leaves the arithmetic in the model.**
(This mirrors the field's history: ORCA continuous batching, vLLM PagedAttention.)

## 4. Where to draw the line: a narrow, stable model-facing contract
Good boundary test: **the interface the engine uses to call the model is small and doesn't change per model.**
nano-vllm's contract is tiny: `model(input_ids, positions) → hidden`; `compute_logits(hidden)`; read metadata from
`Context`; paged KV via `store_kvcache` + flash-attn; provide `weight_loader`; build layers from
`ColumnParallelLinear`/etc. **If adding a model would force an engine-interface change, the boundary is wrong** —
push the varying part to the model side. Iterate until the contract is narrow and stable across Llama/Qwen/Mixtral.

## 5. Invariants come from the DOMAIN + the HARDWARE (not invention)
- **Domain structure**: every AR transformer has prefill (parallel) + decode (sequential, memory-bound), a growing
  KV, causal attention, a vocab sample → dictates scheduler phases, paged KV, sampler.
- **GPU laws**: roofline (compute vs memory bound), HBM scarcity, launch overhead, collective cost → *shape* the
  abstractions (HBM scarce → paging+quant; launch-bound decode → graphs; weight amortization → batching).
The abstractions are **forced by "what all models do" × "what the GPU rewards."**

## 6. Design principles (the architect's mindset)
- **Separate mechanism (engine) from policy/math (model)** — framework vs plugin.
- **Design to the scarce resource and its laws** — every core abstraction stops wasting the GPU.
- **Centralize the hard, shared parts; externalize the variable parts** — every model inherits the GPU optimizations.
- **Keep the model-facing contract narrow + stable** — adding a model shouldn't touch the engine.
- **Make the common case fast, and the model easy to add** — two audiences (perf engineer vs model porter).
- **Correctness-preserving transformations** — fuse/shard/graph/quant must keep the math → build in verification.

## One-paragraph mindset
> Design from **"maximize GPU utilization under latency SLOs, for any model,"** then split **invariant-across-models
> (engine) from per-model (model file)**. Don't guess the abstractions — **build the naive loop, profile, and factor
> each GPU-wasting bottleneck's reusable fix into the engine**, letting the **GPU's laws** and the **domain's
> structure** dictate their shape. Place the **boundary where the model-facing contract is narrowest/most stable**;
> if a new model forces an engine change, the boundary is wrong. Result: the engine centralizes the hard,
> model-agnostic GPU optimizations so every model benefits, and adding a model = writing only its math against a
> small, stable contract.
