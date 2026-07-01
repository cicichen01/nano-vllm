
**  nano vllm vs vllm on context:
  1. Context manager with save/restore. set_forward_context is a with-block that stashes the previous context and restores
  it in a finally. That makes it exception-safe and nestable, versus nano-vllm's manual set_context / reset_context pair.
  2. Per-layer-keyed metadata. nano-vllm has one Context for all attention layers (every layer is the same kind of
  attention). In vLLM, attn_metadata is typically a dict keyed by layer name (each Attention registers a unique layer_name
  at init). That's needed because different layers can need different metadata — full vs sliding-window attention,
  encoder/decoder cross-attention, multiple KV-cache groups. Each layer does attn_metadata[self.layer_name].
  3. Richer payload than just attention. ForwardContext also carries things nano-vllm doesn't have: data-parallel metadata,
  the vllm_config, num_tokens (used to pick the right CUDA-graph / padded batch size), and torch.compile bookkeeping. The
  forward context is in fact what lets captured CUDA graphs and compiled regions find the correct attention metadata at
  replay time — a piece nano-vllm sidesteps because its graph path is simpler.
  4. Process-scoped, not thread-local. It's a plain module global. vLLM runs each (TP/PP) worker as its own process with a
  single model-execution thread, so a process global is unambiguous — there's no concurrent forward in the same process to
  collide with it. (Same assumption nano-vllm makes with its singleton _CONTEXT.)
