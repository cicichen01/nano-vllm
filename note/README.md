# nano-vllm study notes — master index

Two tracks: a **code walkthrough** (Steps 1–6, read the codebase piece by piece) and an
**optimization roadmap** (what's supported + what to add). Cross-cutting deep-dives sit alongside.

## Track A — Code walkthrough (Steps 1–6)  ✅ complete
| Step | Note | Covers | Code |
|---|---|---|---|
| 1 | `note_1.md` | Data model, Sequence lifecycle, sampling (temp/softmax/Gumbel), determinism | `sequence.py`, `sampling_params.py`, `sampler.py` |
| 2 | `note2.md` | Engine loop, `step()`, generate(), tokenizer | `llm_engine.py`, `llm.py` |
| 3 | `note3.md` | Scheduler: continuous batching, prefill-priority, chunked prefill, preemption, `max_num_*` | `scheduler.py` |
| 4 | `note4.md` | BlockManager: paged KV, block_table, prefix caching (xxhash chain), ref-count, eviction | `block_manager.py` |
| 5 | `note5.md` | ModelRunner: tensor prep, slot_mapping, Context, warmup/KV sizing, eager-vs-graph dispatch, TP IPC | `model_runner.py`, `context.py` |
| 6 | `note6.md` | Model & layers: hierarchy, **fused residual stream**, **pre- vs post-norm (+ gradient math)**, embed_head tricks, **weight loading** (`weight_loader`/`packed_modules_mapping`), **TP data-flow**, **parallelism (DP/PP/EP)**, **mmap/pinned/H2D data movement** | `qwen3.py`, `layers/`, `loader.py`, `linear.py`, `embed_head.py`, `attention.py`, `activation.py`, `layernorm.py`, `rotary_embedding.py` |

Extra Q&A deep-dive: `note5_scheduling_and_prefix_cache.md` (scheduling / chunked prefill / prefix-cache reuse).

## Cross-cutting deep-dives (not tied to one step)
| Note | Covers |
|---|---|
| `note_gpu_attention.md` | GPU execution model (SM/warp/tile), batched attention prefill-vs-decode, roofline, Flash-Decoding, **batching economics (arithmetic-intensity vs utilization)**, TP data-flow |
| `note_torch_compile_cudagraph.md` | torch.compile (Inductor) vs CUDA graphs; fusion vs launch-overhead; graph breaks; trace reading; where `@torch.compile` lives; decode kernel anatomy |
| `note_cudagraph_capture.md` | What can/can't be CUDA-graphed (**shape-frozen, value-free-within-shape**), capture-with-zeros, in-kernel loops, paged-attention co-design — all empirically verified |
| `ideas.md` | nano-vllm vs vLLM: Context/forward-context differences |
| `note_engine_design.md` | **How an inference engine is designed** — engine↔model boundary emerges from goal + scarce resource; invariant-vs-variable; discover by naive→profile→factor; narrow stable contract |
| `note_porting_a_model.md` | **How to port a model into an engine** — training-vs-serving forward; the contracts checklist; **fixed-vs-changeable contracts** (law/kernel/design); correctness-gated workflow; measure→diagnose→optimize |

## Track B — Optimization roadmap  → `h100_setup/NANO_VLLM_NOTES.md`
Architecture map, execution flow, **11 supported optimizations**, **candidate backlog (A–N)**, gap taxonomy
(architecture gaps vs incremental). This is the answer to the original "optimizations supported + possible to add."

## Runnable demos & figures → `h100_setup/`
`explore_step*.py`, `profile_*.py`, `roofline_diagnose.py`, `tp_parallel_demo.py`, `torch_compile_*.py`,
`qwen_eager_vs_graph.py`, `plot_*.py` (→ PNGs), and the capture/reshape verification scripts
(`capture_zeros_attn.py`, `reshape_*.py`, `loop_in_kernel.py`). Setup: `setup_h100.sh`, `test_h100.py`,
`VERSIONS_EXPLAINED.md`, `push_to_fork.md`.

## Step 7
- **A. End-to-end capstone** ✅ done → `note7.md` (+ figure `h100_setup/end_to_end.png`): one `generate()` traced
  through every component (submit → step-loop [schedule→run→postprocess] → return), with a concrete 2-seq trace and
  how each optimization surfaces.
- **B. Optimization roadmap deep-dive** *(next candidate)* — expand `NANO_VLLM_NOTES.md` backlog into design sketches
  (spec decoding, async scheduling, chunked-prefill/decode overlap, quantization, prefix-cache eviction,
  disaggregation, MoE/EP).
- **C. One advanced topic in full** — e.g. speculative decoding end-to-end, or the distributed worker/IPC (shm+NCCL) path.
