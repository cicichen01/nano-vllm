# Note — Porting a model into an inference engine (mindset + contracts)

How to take a trained model (which has a *training* forward) and enable it in a serving engine.
Companion to `note_engine_design.md` (how the engine is designed) and the Step 1–7 walkthrough.

## 1. Core reframe: training-forward ≠ serving-forward
Same function, same weights, **different contracts**:
| | training forward | serving forward |
|---|---|---|
| direction | forward+backward (grads) | forward only |
| sequence | whole seq at once | **autoregressive**: prefill then **decode** (1 tok/step) |
| KV | recomputed | **cached & reused** (paged) |
| shapes | dynamic OK (eager) | often **static** (CUDA graphs) |
| batch | dense `[B,L]` padded | **flat/packed** varlen + per-request state |
| metric | throughput over a dataset | **latency + throughput + memory** on live mixed traffic |
| memory | activations for backward | **weights + KV** dominate |
| precision | bf16/fp32 | often **quantized** |
So you **re-express the architecture** to fit serving — same math, new implementation. **Re-express, don't re-derive.**

## 2. Two hard invariants (everything else is optimization)
1. **Numerical equivalence** — match a **reference** (HF/training) output within tolerance. *Correctness is a gate.*
2. **Fit the engine's execution contracts** — shape/KV/parallelism/graph rules (below).

## 3. Know the boundary: how much work is it?
The engine is generic; you write a **model file (+ maybe new layers/kernels)**. First assess:
- **Just a model file** (reuse layers): different norm placement/activation/dims, tied embeddings, standard attention.
- **Needs new layers/kernels/engine features**: novel attention (sliding-window, MLA, cross-attn), new positional
  encoding (ALiBi), **MoE** (expert layers + EP + routing), multimodal (encoder+fusion), new quant format (kernels).

## 4. The contracts checklist (for each: "how does my model meet this?")
- **Correctness/weights** — match architecture; map checkpoint→your (fused/TP-sharded) params **math-equivalently**
  (`weight_loader` + `packed_modules_mapping`); have a **correctness oracle** (compare logits/tokens to HF).
- **KV + attention** — read/write a **paged KV cache**; split **prefill** (varlen, full causal) vs **decode**
  (1 query × cached KV) with a **graph-friendly** kernel (fixed launch, length as a value).
- **Batching + metadata** — flat packed batch; position-wise layers batch free; **only attention** needs boundary
  metadata (`cu_seqlens`/`block_tables`) via the engine's Context.
- **Static-shape / graph-ability (decode)** — shape-deterministic + sync-free; value-dependent shapes (e.g. MoE
  routing) need padding/graph-breaks.
- **Parallelism** — column/row-parallel layers, norms replicated, embed/LM-head vocab-parallel, collectives placed
  to minimize comms (column→row = 1 all_reduce/block); MoE → EP + all-to-all.
- **Precision** — pick dtype; if quantizing (weight/KV/activation) pick kernels and **validate quality**.

## 5. Which contracts are FIXED vs CHANGEABLE (know what to fit vs improve)
Contracts come from different **sources** with different mutability:
| source | mutability | examples |
|---|---|---|
| **hardware/platform law** | **immutable** (design around) | graphs need static shapes+addresses; value→shape needs a host sync; DMA fixed addresses; collective semantics |
| **math/correctness** | **immutable** | TP row must all_reduce; causality; weight equivalence |
| **kernel/library** | fixed *while using that kernel* | flash-attn `block_size % BLOCK_N`; dtype support |
| **engine design choice** | **changeable** (edit engine; know blast radius) | block_size value; graph buckets; one global Context; sync scheduler; TP-only; offline-only |

**Test for any constraint:** *is the GPU/math telling me this, or the engine's author?* Former → obey; latter →
renegotiable. **Classify by:** what breaks if violated + is there an alternative — no alt (platform/math) = FIXED;
alt = swap kernel = kernel-fixed; alt = modify engine = CHANGEABLE.
**Discover the contracts** by reading the **seams** (where engine calls model): forward signature, `Context`,
`store_kvcache`/paged attn, `weight_loader`/`packed_modules_mapping`, parallel layer base classes, `capture_cudagraph`;
plus **asserts** (= invariants) and **config fields** (= tunable knobs).
**Two tools:** (a) **diff against a fuller engine (vLLM)** — a difference ⇒ it's a *choice*, not a law;
(b) **trace the blast radius** — a config/layer swap = easy improvement; a new abstraction = architecture change
(this *is* the `NANO_VLLM_NOTES.md` "architecture gap vs incremental" taxonomy).

## 6. Workflow — correctness gate at every step
```
1. Reference oracle (HF logits on a fixed input)             ← the truth to match
2. Implement architecture in engine layers; map weights      → verify match  ✅gate
3. Wire paged KV + prefill/decode attention                  → verify match  ✅gate
4. Add TP sharding                                           → verify match  ✅gate
5. Enable CUDA graphs for decode (shape-determinism)         → verify match  ✅gate
6. Measure → find bound → optimize the right axis
7. (optional) Quantize                                       → validate quality ✅gate
```
Each transformation (fuse/shard/graph/quant) can **silently corrupt** outputs → re-verify after each.

## 7. Measure → diagnose → optimize (don't guess)
- **Measure**: prefill TTFT, decode ITL, throughput, memory (weights/KV/activation), per-kernel (roofline, ncu).
- **Diagnose the bound**: prefill = compute-bound; decode = memory-bound (weights+KV).
- **Optimize the right axis** (hierarchy): (1) keep GPU busy — CUDA graphs, pinned/async H2D, batch more;
  (2) make each kernel efficient — fuse / quantize / tensor-cores (per the bound); (3) do less work — KV cache,
  prefix caching, chunked prefill, spec decoding, sparsity/MoE. *Effort on the wrong axis is wasted.*

## One-paragraph mindset
> Porting a model = **re-expressing a known architecture (same weights/math) to satisfy the engine's serving
> contracts** — autoregressive prefill/decode over paged KV, flat batching with attention-only boundary metadata,
> static shapes for graphs, TP/EP sharding with minimal collectives, optional quantization — while **verifying
> numerical equivalence at every transformation** and finally **measuring, diagnosing the bound (compute vs memory),
> and optimizing that axis.** Obey the **laws** (GPU/math) and **current kernels**; treat **engine design choices** as
> your improvement backlog (change them only when a choice truly blocks your model, and you've judged the ripple cost).
